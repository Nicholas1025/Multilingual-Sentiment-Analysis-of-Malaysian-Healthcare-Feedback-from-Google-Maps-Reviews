"""Pydantic request / response schemas for the FastAPI app."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class PredictRequest(BaseModel):
    text: str


class TopWord(BaseModel):
    word: str
    weight: float


class PredictResponse(BaseModel):
    label: str                         # positive / neutral / negative
    confidence: float                  # 0.0 - 1.0
    language: str                      # english / malay / chinese / mixed / other
    emojis: List[str]                  # original emoji characters found
    aspects: Dict[str, str]            # aspect -> sentiment
    text_cleaned: str                  # the preprocessed text (for transparency)
    model_used: str                    # "baseline" or "transformer"
    class_scores: Optional[Dict[str, float]] = None  # full distribution
    top_words: Optional[List[TopWord]] = None        # explainability


class BatchPredictItem(BaseModel):
    text: str
    label: str
    confidence: float
    language: str


class StatsResponse(BaseModel):
    total_reviews: int
    label_distribution: Dict[str, int]
    language_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    avg_text_length: float
    top_aspects: Dict[str, int]


# ─── Ranking page ──────────────────────────────────────────────────────────

class AspectScore(BaseModel):
    aspect: str
    positive_rate: float
    negative_rate: float
    mentions: int


class RankingRow(BaseModel):
    rank: int
    hospital: str
    location: str
    total_reviews: int
    positive_rate: float
    negative_rate: float
    neutral_rate: float
    aspects: List[AspectScore]


class RankingResponse(BaseModel):
    sort_by: str
    total_hospitals: int
    rows: List[RankingRow]


# ─── Chat / RAG ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


class CitedReview(BaseModel):
    hospital: str
    text: str
    date: str
    stars: int
    similarity: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    cited_reviews: List[CitedReview]
    model_used: str                   # "gemini-1.5-flash" or "pattern-based (fallback)"
    sources_count: int                # total retrieved (may be > shown)
    intent: Optional[str] = None      # detected intent, for debugging
    fallback_reason: Optional[str] = None
