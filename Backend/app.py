"""
backend/app.py
FastAPI backend for PHI De-identification System.

Run from the phi_deidentification/ root directory:
    python backend/app.py
"""

import sys
import os
import tempfile
from pathlib import Path

# ── Auto-detect project root ──────────────────────────────────────────────────
CURRENT_FILE = Path(__file__).resolve()

def find_project_root(start: Path) -> Path:
    for candidate in [start, start.parent, start.parent.parent]:
        if (candidate / "inference").exists() and (candidate / "config").exists():
            return candidate
    return start.parent

PROJECT_ROOT = find_project_root(CURRENT_FILE)
sys.path.insert(0, str(PROJECT_ROOT))
print(f"[Backend] Project root : {PROJECT_ROOT}")

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="PHI De-identification API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model loading ────────────────────────────────────────────────────────
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            from inference.predict import PHIPredictor
            _predictor = PHIPredictor()
            print("[Backend] PHIPredictor loaded successfully.")
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=f"Model checkpoint not found. Run training first. Detail: {e}")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")
    return _predictor


# ── Core redaction helper ─────────────────────────────────────────────────────

def run_redaction(text: str) -> dict:
    predictor = get_predictor()

    from inference.redact_text import hybrid_redact
    from utils.phi_extractor import extract_phi_from_text

    # Redacted text via model + regex hybrid
    redacted = hybrid_redact(text, predictor=predictor, use_model=True)

    # Entity list from model
    raw_entities = predictor.predict_entities(text)

    # Normalize entity format for frontend
    entities = [
        {
            "entity_type": e["entity_type"],
            "text":        e["text"],
            "start_token": e.get("start_token", 0),
            "end_token":   e.get("end_token",   0),
        }
        for e in raw_entities
    ]

    counts = {}
    for e in entities:
        counts[e["entity_type"]] = counts.get(e["entity_type"], 0) + 1

    return {
        "original":      text,
        "redacted":      redacted,
        "entities":      entities,
        "entity_counts": counts,
    }


# ── Schemas ───────────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        get_predictor()
        loaded = True
    except Exception:
        loaded = False
    return {
        "status":       "ok",
        "model_loaded": loaded,
        "project_root": str(PROJECT_ROOT),
        "version":      "1.0.0",
    }


@app.post("/redact/text")
def redact_text_endpoint(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return run_redaction(req.text)


@app.post("/redact/pdf")
async def redact_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    predictor = get_predictor()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from utils.pdf_parser import extract_text_from_pdf
        from inference.redact_text import redact_document

        raw_text = extract_text_from_pdf(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail=(
                "Could not extract text from this PDF. "
                "It may be scanned — ensure Tesseract and Poppler are installed."
            ))

        result   = redact_document(raw_text, predictor=predictor, use_model=True)
        entities = result.get("entities", [])
        counts   = {}
        for e in entities:
            counts[e["entity_type"]] = counts.get(e["entity_type"], 0) + 1
        result["entity_counts"] = counts
        return result

    finally:
        os.unlink(tmp_path)


@app.post("/predict/text")
def predict_tokens(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    predictor    = get_predictor()
    tokens, tags = predictor.predict_tags(req.text)
    return {"tokens": [{"token": t, "tag": tag} for t, tag in zip(tokens, tags)]}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)