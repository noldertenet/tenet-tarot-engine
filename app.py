from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

# OpenAI client (usa OPENAI_API_KEY da env)
client = OpenAI()

# TENET API Keys (comma-separated)
VALID_KEYS = [k.strip() for k in os.environ.get("TENET_API_KEYS", "").split(",") if k.strip()]

def _unauthorized():
    return jsonify({
        "ok": False,
        "error": {"code": "UNAUTHORIZED", "message": "Invalid TENET API key"}
    }), 401

def _check_key(req):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    key = auth.replace("Bearer ", "").strip()
    return key in VALID_KEYS

@app.get("/health")
def health():
    if not _check_key(request):
        return _unauthorized()

    return jsonify({
        "ok": True,
        "service": "tenet-core-demo",
        "mode": "demo"
    })

@app.post("/v1/reading")
def reading():
    if not _check_key(request):
        return _unauthorized()

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if len(question) < 5:
        return jsonify({
            "ok": False,
            "error": {"code": "INPUT_INVALID", "message": "Inserisci una domanda più completa."}
        }), 400

    # Demo: estrazione fissa (per testare flusso end-to-end)
    cards = [
        {"name": "La Papessa", "role": "Stato"},
        {"name": "Il Mago", "role": "Dinamica"},
        {"name": "La Torre", "role": "Direzione"},
    ]

    reading = {
        "intro": "Qui si vede una relazione che vive molto più dentro che fuori: emozioni presenti, ma trattenute.",
        "card_sections": [
            {
                "title": "Stato – La Papessa",
                "body": "In questo momento il legame si muove soprattutto sul piano interiore. Ci sono cose sentite, ma non dette. Non per mancanza di sentimento, ma per timore di esporsi."
            },
            {
                "title": "Dinamica – Il Mago",
                "body": "La possibilità di cambiare passo c’è. Il potenziale non manca, manca un gesto chiaro. Si pensa molto, si immagina, ma l’azione viene rimandata."
            },
            {
                "title": "Direzione – La Torre",
                "body": "Questo equilibrio non regge a lungo. Non significa per forza fine, ma significa che una verità dovrà emergere: o si chiarisce, o il non detto diventa distanza."
            }
        ],
        "synthesis": "La relazione può maturare solo con una scelta chiara e visibile. Se resta tutto sospeso, sarà il silenzio a spezzare l’equilibrio."
    }

    return jsonify({
        "ok": True,
        "deck": data.get("deck", "tarot_demo"),
        "spread": data.get("spread", "three_cards_v1"),
        "language": data.get("language", "it"),
        "style": data.get("style", "tenet_clear"),
        "cards": cards,
        "reading": reading,
        "safety": {
            "no_predictions": True,
            "no_certainty": True,
            "no_prescriptions": True
        }
    })
