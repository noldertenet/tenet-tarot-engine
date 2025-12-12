import os
import json
import random
import re
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


APP_VERSION = "2.0.0-tenet-context"

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
OPENAI_MODEL = os.getenv("TENET_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
TENET_API_KEYS_ENV = os.getenv("TENET_API_KEYS", "").strip()  # es: "TENET-AAAA-001,TENET-BBBB-002"

KNOWLEDGE_FILE = os.getenv("TENET_KNOWLEDGE_FILE", "tarot_knowledge_base.json")
DEFAULT_DECK_NAME = os.getenv("TENET_DEFAULT_DECK", "tarot_demo")

# Spread “three_cards_v1” usato dal plugin
SPREAD_THREE_CARDS = [
    ("Stato", "Dove si trova la situazione ora, il clima e la sostanza."),
    ("Dinamica", "Cosa la muove, cosa facilita o blocca, che cosa chiede di essere agito."),
    ("Direzione", "Verso cosa tende, quale verità emerge, quale passaggio è richiesto."),
]

# -------------------------------------------------------------------
# CONTEXT (C) - classificazione semplice e robusta
# -------------------------------------------------------------------
CONTEXTS = ["relazioni", "lavoro", "crescita", "scelta", "neutro"]

CONTEXT_KEYWORDS = {
    "relazioni": [
    "amore", "relazione", "coppia", "partner",
    "fidanz", "marito", "moglie",
    "gelosia", "tradimento", "ex", "sentimenti"
],
    "lavoro": [
        "lavoro", "carriera", "colloquio", "azienda", "capo", "progetto", "business",
        "clienti", "soldi", "denaro", "stipendio", "fatturato", "contratto", "studio"
    ],
    "crescita": [
        "ansia", "paura", "blocco", "crescita", "consapevolezza", "autostima",
        "energia", "chakra", "percorso", "guarigione", "stress", "equilibrio"
    ],
    "scelta": [
        "scegliere", "scelta", "bivio", "decisione", "decidere", "lasciare", "cambiare",
        "trasferir", "rinuncia", "opzione", "strada", "direzione"
    ],
}

def detect_context(question: str) -> str:
    q = (question or "").lower().strip()
    if not q:
        return "neutro"

    scores = {c: 0 for c in CONTEXTS}
    for ctx, kws in CONTEXT_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                scores[ctx] += 1

    best_ctx = max(scores, key=lambda k: scores[k])
    if scores[best_ctx] == 0:
        return "neutro"
    return best_ctx


# -------------------------------------------------------------------
# PROMPTS per contesto (C)
# -------------------------------------------------------------------
BASE_STYLE_RULES = """
Sei TENET, un motore di lettura simbolica. Usi un tono chiaro, responsabile e non deterministico.
Non “sentenzi” e non predici eventi certi: descrivi dinamiche, possibilità e passaggi utili.
Evita frasi tradotte letteralmente o costruzioni rigide. Italiano naturale, scorrevole.
Non usare toni fatalisti. Se emerge criticità, formulala come invito a chiarire/riconoscere/decidere.
Struttura: una frase introduttiva generale + 3 sezioni (Stato/Dinamica/Direzione) + Sintesi finale.
"""

CONTEXT_INSTRUCTIONS = {
    "relazioni": """
Cornice: relazione/legame. Parla di comunicazione, fiducia, confini, bisogni e maturazione del legame.
Evita “torna/non torna” o verdetti. Dai indicazioni su che cosa va chiarito e cosa può far crescere.
""",
    "lavoro": """
Cornice: lavoro/progetto/denaro. Parla di ruoli, processi, strategia, scelte pratiche e tempistiche.
Evita frasi vaghe. Mantieni concretezza senza promettere risultati.
""",
    "crescita": """
Cornice: crescita personale/stato interiore. Parla di risorse interne, blocchi, consapevolezze, integrazione.
Evita diagnosi cliniche. Offri lettura come orientamento e comprensione.
""",
    "scelta": """
Cornice: decisione/bivio. Metti a fuoco opzioni, criteri, conseguenze probabili e passo successivo.
Evita “devi fare X”. Usa “può essere utile”, “se scegli A allora…”.
""",
    "neutro": """
Cornice: generale. Mantieni la lettura applicabile a più aspetti. Non forzare in amore o lavoro.
""",
}


def build_system_prompt(context: str) -> str:
    ctx = context if context in CONTEXT_INSTRUCTIONS else "neutro"
    return (BASE_STYLE_RULES + "\n" + CONTEXT_INSTRUCTIONS[ctx]).strip()


def ensure_openai_client():
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK non disponibile. Verifica requirements.txt.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY mancante nelle env di Render.")
    return OpenAI(api_key=api_key)


# -------------------------------------------------------------------
# KNOWLEDGE BASE
# -------------------------------------------------------------------
def load_knowledge() -> Dict[str, Any]:
    if not os.path.exists(KNOWLEDGE_FILE):
        # Fallback minimo: se manca il file, teniamo un set ridotto per non crashare.
        return {
            "cards": [
                {"id": "la_papessa", "name": "La Papessa", "keywords": ["interiorità", "silenzio", "non detto"]},
                {"id": "il_mago", "name": "Il Mago", "keywords": ["inizio", "potenziale", "azione"]},
                {"id": "la_torre", "name": "La Torre", "keywords": ["verità", "scossa", "ristrutturazione"]},
            ]
        }
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


KB = load_knowledge()

def all_cards() -> List[Dict[str, Any]]:
    # Supporta vari formati: {"cards":[...]} oppure lista diretta
    if isinstance(KB, dict) and isinstance(KB.get("cards"), list):
        return KB["cards"]
    if isinstance(KB, list):
        return KB
    return []

def draw_cards(n: int = 3) -> List[Dict[str, Any]]:
    cards = all_cards()
    if len(cards) < n:
        return cards
    return random.sample(cards, n)


def card_brief(card: Dict[str, Any]) -> str:
    name = card.get("name") or card.get("title") or "Carta"
    # prendiamo campi comuni senza pretendere uno schema unico
    keywords = card.get("keywords") or card.get("tags") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    kw = ", ".join([str(x) for x in keywords[:6]]) if keywords else ""
    base = card.get("meaning") or card.get("base_meaning") or card.get("description") or ""
    base = re.sub(r"\s+", " ", str(base)).strip()
    parts = [f"- {name}"]
    if kw:
        parts.append(f"  parole chiave: {kw}")
    if base:
        parts.append(f"  nota: {base[:220]}")
    return "\n".join(parts)


# -------------------------------------------------------------------
# AUTH
# -------------------------------------------------------------------
def parse_allowed_keys() -> List[str]:
    if not TENET_API_KEYS_ENV:
        return []
    # supporta separatori comuni
    raw = re.split(r"[,\n; ]+", TENET_API_KEYS_ENV)
    return [x.strip() for x in raw if x.strip()]

ALLOWED_KEYS = parse_allowed_keys()

def extract_bearer_key(req) -> str:
    h = req.headers.get("Authorization", "")
    if not h:
        return ""
    m = re.match(r"Bearer\s+(.+)", h, re.IGNORECASE)
    return m.group(1).strip() if m else ""

def unauthorized(message: str = "Invalid TENET API key"):
    return jsonify({
        "ok": False,
        "error": {"code": "UNAUTHORIZED", "message": message}
    }), 401

def require_key():
    key = extract_bearer_key(request)
    if not key:
        return False
    if ALLOWED_KEYS and key not in ALLOWED_KEYS:
        return False
    # se ALLOWED_KEYS è vuoto, per sicurezza blocchiamo comunque
    return bool(ALLOWED_KEYS)


# -------------------------------------------------------------------
# FLASK APP
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.get("/")
def root():
    return jsonify({"ok": True, "service": "tenet-core", "version": APP_VERSION})


@app.get("/health")
def health():
    if not require_key():
        return unauthorized()
    return jsonify({
        "ok": True,
        "service": "tenet-core",
        "version": APP_VERSION,
        "model": OPENAI_MODEL
    })


def generate_reading(question: str, cards: List[Dict[str, Any]], context: str, language: str = "it") -> Dict[str, Any]:
    client = ensure_openai_client()

    system_prompt = build_system_prompt(context)

    # Prepariamo input pulito per il modello: domanda + carte + posizioni
    positions = "\n".join([f"{i+1}) {title}: {hint}" for i, (title, hint) in enumerate(SPREAD_THREE_CARDS)])
    cards_text = "\n\n".join([card_brief(c) for c in cards])

    user_prompt = f"""
DOMANDA:
{question}

CONTESTO RILEVATO:
{context}

STESA A 3 CARTE:
{positions}

CARTE ESTRATTE (usa i nomi esatti):
{cards_text}

Richiesta:
Scrivi una lettura con:
- intro (1–2 frasi)
- 3 sezioni: "Stato – <NomeCarta>", "Dinamica – <NomeCarta>", "Direzione – <NomeCarta>" (ognuna 2–4 frasi)
- Sintesi finale (2–3 frasi)

Evita assoluti e profezie. Mantieni coerenza e leggibilità.
""".strip()

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6
    )

    text = (resp.output_text or "").strip()

    # Parser leggero: ricaviamo le sezioni dal testo (se il modello non segue, comunque mostriamo raw)
    intro = ""
    synthesis = ""
    sections = []

    # Tentiamo split per titoli principali
    # Cerchiamo "Sintesi" come marker
    if "Sintesi" in text:
        before, after = text.split("Sintesi", 1)
        synthesis = after.strip(" :\n\t-")
        main = before.strip()
    else:
        main = text

    # Intro = prima riga/paragrafo fino al primo "Stato –"
    m_intro = re.split(r"\n\s*\n|Stato\s*[–-]", main, maxsplit=1)
    if m_intro:
        intro = m_intro[0].strip()

    # Sezioni: cerchiamo righe che iniziano con Stato/Dinamica/Direzione
    def find_block(label: str) -> Tuple[str, str]:
        pattern = rf"{label}\s*[–-]\s*(.+)"
        m = re.search(pattern, main)
        if not m:
            return "", ""
        title = m.group(0).strip()
        start = m.start()
        return title, str(start)

    # Parsing semplice con regex a blocchi
    block_pattern = re.compile(r"(Stato|Dinamica|Direzione)\s*[–-]\s*([^\n]+)\n(.*?)(?=\n(Stato|Dinamica|Direzione)\s*[–-]|\Z)", re.S)
    for m in block_pattern.finditer(main):
        label = m.group(1).strip()
        card_name = m.group(2).strip()
        body = re.sub(r"\s+", " ", m.group(3)).strip()
        sections.append({
            "title": f"{label} – {card_name}",
            "body": body
        })

    if not sections:
        # fallback: se non parsabile, impacchettiamo tutto come sezione unica
        sections = [{"title": "Lettura", "body": text}]

    return {
        "ok": True,
        "context": context,
        "intro": intro,
        "card_sections": sections,
        "synthesis": synthesis
    }


@app.post("/v1/reading")
def v1_reading():
    if not require_key():
        return unauthorized()

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if len(question) < 5:
        return jsonify({"ok": False, "error": {"code": "BAD_REQUEST", "message": "Domanda troppo breve"}}), 400

    language = (payload.get("language") or "it").strip().lower()
    spread = (payload.get("spread") or "").strip()
    deck = (payload.get("deck") or DEFAULT_DECK_NAME).strip()

    # Per ora supportiamo la demo spread three_cards_v1
    if spread and spread != "three_cards_v1":
        return jsonify({"ok": False, "error": {"code": "BAD_REQUEST", "message": "Spread non supportato in demo"}}), 400

    ctx = detect_context(question)
    cards = draw_cards(3)

    try:
        out = generate_reading(question=question, cards=cards, context=ctx, language=language)
        # Aggiungiamo metadati utili per debug/demo
        out["meta"] = {
            "version": APP_VERSION,
            "deck": deck,
            "spread": "three_cards_v1",
            "cards": [c.get("name") or c.get("title") for c in cards],
        }
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": {"code": "SERVER_ERROR", "message": str(e)}}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
