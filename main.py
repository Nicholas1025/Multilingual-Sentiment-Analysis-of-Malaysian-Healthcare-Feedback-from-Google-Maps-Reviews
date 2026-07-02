"""
FastAPI entry point for the Healthcare Feedback Sentiment Analysis System.

Routes:
    GET  /                  -> single-review UI
    GET  /batch             -> batch CSV upload UI
    GET  /dashboard         -> dataset statistics UI
    GET  /about             -> project info
    POST /api/predict       -> single review prediction
    POST /api/predict-batch -> CSV upload, returns CSV with predictions
    GET  /api/stats         -> dataset statistics for the dashboard
    GET  /api/health        -> health check

Start: python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
"""

import csv
import io
import os
import sys
from collections import Counter

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# allow importing the inference service
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from backend import predict_service             # noqa: E402
from backend import hospital_stats               # noqa: E402
from backend.schemas import (                   # noqa: E402
    PredictRequest, PredictResponse, StatsResponse,
    RankingResponse, RankingRow, AspectScore,
    ChatRequest, ChatResponse, CitedReview,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── APP SETUP ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Healthcare Feedback Sentiment Analysis",
    description="Malaysia-focused sentiment analysis (Melaka hospitals).",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = os.path.join(HERE, "static")
DATA_DIR   = os.path.join(HERE, "Data")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ─── HTML ROUTES (serve the SPA pages) ──────────────────────────────────────

def _page(name: str):
    return FileResponse(os.path.join(STATIC_DIR, name))


@app.get("/")
def home():           return _page("index.html")

@app.get("/batch")
def batch_page():     return _page("batch.html")

@app.get("/dashboard")
def dashboard_page(): return _page("dashboard.html")

@app.get("/ranking")
def ranking_page():   return _page("ranking.html")

@app.get("/chat")
def chat_page():      return _page("chat.html")

@app.get("/about")
def about_page():     return _page("about.html")

# ─── API ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "transformer_loaded": predict_service._transformer is not None,
        "baseline_loaded":    predict_service._baseline is not None,
    }


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="text too long (>5000)")

    result = predict_service.predict(text)
    return PredictResponse(**result)


@app.post("/api/predict-batch")
async def api_predict_batch(file: UploadFile = File(...)):
    """Accept a CSV file with a 'text' column and return a labelled CSV."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="upload a .csv file")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not read CSV: {e}")

    if "text" not in df.columns:
        raise HTTPException(status_code=400,
                            detail="CSV must contain a 'text' column")

    df = df.head(1000)         # safety cap
    preds = [predict_service.predict(str(t)) for t in df["text"]]
    df["predicted_label"]      = [p["label"] for p in preds]
    df["confidence"]           = [round(p["confidence"], 3) for p in preds]
    df["language"]             = [p["language"] for p in preds]
    df["aspects"]              = ["; ".join(f"{a}={s}" for a, s in p["aspects"].items())
                                  for p in preds]

    out = io.StringIO()
    df.to_csv(out, index=False, encoding="utf-8")
    out.seek(0)
    return StreamingResponse(
        iter([out.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.get("/api/stats", response_model=StatsResponse)
def api_stats():
    """Read the cleaned dataset (Stage 2 output) and return its stats."""
    candidates = [
        os.path.join(DATA_DIR, "processed", "healthcare_cleaned.csv"),
        os.path.join(DATA_DIR, "raw",       "healthcare_final.csv"),
        os.path.join(DATA_DIR, "raw",       "healthcare_merged.csv"),
        os.path.join(DATA_DIR, "raw",       "healthcare_raw.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return JSONResponse({"detail": "no dataset on disk"}, status_code=404)

    df = pd.read_csv(path, encoding="utf-8-sig")

    label_dist  = Counter(df["label"].dropna()) if "label" in df.columns else Counter()
    source_dist = Counter(df.get("source", []))

    if "language" in df.columns:
        lang_dist = Counter(df["language"].dropna())
    else:
        lang_dist = Counter()

    text_col = "text_cleaned" if "text_cleaned" in df.columns else "text"
    avg_len = float(df[text_col].astype(str).str.len().mean()) if text_col in df.columns else 0.0

    # quick aspect frequency (uses the lexicon from Stage 5)
    # aspect_sentiment is on sys.path via backend.predict_service
    import aspect_sentiment as _asp
    asp_counter = Counter()
    # the cleaned CSV stores the raw text in text_original
    raw_col = ("text" if "text" in df.columns
               else "text_original" if "text_original" in df.columns
               else None)
    sample = df[raw_col].astype(str).tolist() if raw_col else []
    for t in sample[:2000]:
        for a in _asp.detect_aspects(t):
            asp_counter[a] += 1

    return StatsResponse(
        total_reviews        = int(len(df)),
        label_distribution   = dict(label_dist),
        language_distribution= dict(lang_dist),
        source_distribution  = dict(source_dist),
        avg_text_length      = round(avg_len, 1),
        top_aspects          = dict(asp_counter),
    )


# ─── RANKING API ────────────────────────────────────────────────────────────

@app.get("/api/ranking", response_model=RankingResponse)
def api_ranking(sort_by: str = "overall",
                location: str = "",
                order: str = "desc"):
    """Return the hospital leaderboard.

    sort_by:   "overall" or an aspect key (doctor / waiting_time / ...)
    location:  ""  ->  all,   "KL",  "Melaka"
    order:     "desc" -> best first, "asc" -> worst first
    """
    loc_filter = location.strip() or None
    reverse    = (order != "asc")

    if sort_by == "overall":
        ranked = hospital_stats.rank_by(
            aspect=None, location=loc_filter,
            metric="positive_rate", reverse=reverse,
        )
    else:
        ranked = hospital_stats.rank_by(
            aspect=sort_by, location=loc_filter,
            metric="positive_rate", reverse=reverse,
        )

    rows = []
    for i, h in enumerate(ranked, start=1):
        aspect_scores = [
            AspectScore(
                aspect        = a,
                positive_rate = v["positive_rate"],
                negative_rate = v["negative_rate"],
                mentions      = v["mentions"],
            )
            for a, v in h["aspects"].items()
        ]
        rows.append(RankingRow(
            rank          = i,
            hospital      = h["hospital"],
            location      = h["location"],
            total_reviews = h["total_reviews"],
            positive_rate = h["overall_sentiment"]["positive"],
            negative_rate = h["overall_sentiment"]["negative"],
            neutral_rate  = h["overall_sentiment"]["neutral"],
            aspects       = aspect_scores,
        ))

    return RankingResponse(
        sort_by         = sort_by,
        total_hospitals = len(rows),
        rows            = rows,
    )


# ─── CHAT / RAG API ────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    from backend import chatbot                              # lazy import
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is empty")
    if len(q) > 500:
        raise HTTPException(status_code=400, detail="question too long (>500)")

    result = chatbot.answer(q)
    cited = [CitedReview(**c) for c in result.get("cited_reviews", [])]
    return ChatResponse(
        question         = q,
        answer           = result["answer"],
        cited_reviews    = cited,
        model_used       = result.get("model_used", "unknown"),
        sources_count    = result.get("sources_count", len(cited)),
        intent           = result.get("intent"),
        fallback_reason  = result.get("fallback_reason"),
    )
