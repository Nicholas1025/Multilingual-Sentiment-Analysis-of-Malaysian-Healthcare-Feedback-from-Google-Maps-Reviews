"""
TF-IDF-based review retriever - the R in our RAG pipeline.

Reuses the pre-trained TF-IDF vectorizer from Stage 4 (models/tfidf_vectorizer.pkl)
so we don't need any new embedding dependency. All eligible reviews are
vectorized once at startup and cached; retrieval on a query then takes < 10 ms.

Retrieval augmentations to combat bag-of-words semantic gap:
  - Aspect-consistency filter: reviews must mention the queried aspect via
    the multilingual lexicon (aspect_sentiment.ASPECTS).
  - Sentiment-intent alignment: BEST queries prefer positive-labeled reviews;
    WORST queries prefer negative.
  - Corpus restricted to hospitals present in hospital_stats (removes tiny-n
    hospitals like UMMC with only 9 reviews).
"""

import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Source Codes", "5_advanced"))

import aspect_sentiment as asp   # noqa: E402

RAW_CSV        = os.path.join(ROOT, "Data",   "raw", "healthcare_raw.csv")
TFIDF_VEC_PATH = os.path.join(ROOT, "models", "tfidf_vectorizer.pkl")

# Import location classifier + eligible-hospital set from hospital_stats
from backend.hospital_stats import classify_location, load as _load_stats  # noqa: E402


class ReviewRetriever:
    """TF-IDF + cosine similarity retriever over the review corpus."""

    def __init__(self, vectorizer, reviews_df: pd.DataFrame):
        self.vec     = vectorizer
        self.reviews = reviews_df.reset_index(drop=True)
        self.matrix  = self.vec.transform(self.reviews["text"].astype(str))
        self.reviews["_location"] = self.reviews["hospital"].apply(classify_location)
        # Precompute aspect sets per review (once, at startup).
        # detect_aspects returns a list; store as a set for O(1) membership.
        self._aspects = [set(asp.detect_aspects(t))
                         for t in self.reviews["text"].astype(str)]

    def retrieve(self,
                 query: str,
                 k: int = 10,
                 filter_location: Optional[str] = None,
                 filter_hospital: Optional[str] = None,
                 filter_aspect:   Optional[str] = None,
                 filter_label:    Optional[str] = None,
                 min_similarity:  float = 0.05) -> List[Dict]:
        """Return top-k most similar reviews, each with hospital, text,
        review_date, stars, similarity score.

        Filters (all AND-ed):
          filter_location: 'KL' or 'Melaka'
          filter_hospital: exact hospital name
          filter_aspect:   restrict to reviews mentioning this aspect
          filter_label:    'positive' / 'neutral' / 'negative'
        """
        if not query or not query.strip():
            return []

        q_vec = self.vec.transform([query])
        sims  = cosine_similarity(q_vec, self.matrix).flatten()

        mask = np.ones(len(sims), dtype=bool)
        if filter_location:
            mask &= (self.reviews["_location"].values == filter_location)
        if filter_hospital:
            mask &= (self.reviews["hospital"].values == filter_hospital)
        if filter_aspect:
            asp_mask = np.array([filter_aspect in s for s in self._aspects])
            mask &= asp_mask
        if filter_label and "label" in self.reviews.columns:
            mask &= (self.reviews["label"].values == filter_label)

        sims = np.where(mask, sims, -1.0)
        top_idx = sims.argsort()[-k:][::-1]

        results: List[Dict] = []
        for i in top_idx:
            score = float(sims[i])
            if score < min_similarity:
                continue
            row = self.reviews.iloc[i]
            results.append({
                "hospital":   str(row["hospital"]),
                "text":       str(row["text"]),
                "date":       str(row.get("review_date", "")),
                "stars":      int(row["stars"]) if pd.notna(row.get("stars")) else 0,
                "similarity": round(score, 3),
                "location":   row["_location"],
            })
        return results


# ─── Module-level singleton ─────────────────────────────────────────────────

_retriever: Optional[ReviewRetriever] = None


def load() -> Optional[ReviewRetriever]:
    global _retriever
    if _retriever is not None:
        return _retriever

    if not (os.path.exists(TFIDF_VEC_PATH) and os.path.exists(RAW_CSV)):
        print(f"[retriever] WARN: missing vectorizer or dataset "
              f"(vec={os.path.exists(TFIDF_VEC_PATH)}, csv={os.path.exists(RAW_CSV)})")
        return None

    try:
        with open(TFIDF_VEC_PATH, "rb") as f:
            vec = pickle.load(f)
        df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
        df = df.dropna(subset=["hospital", "text"]).reset_index(drop=True)

        # Restrict to hospitals that appear in the aggregate stats (>=10 reviews).
        # This drops UMMC (n=9) from the RAG corpus for consistency with the
        # leaderboard - the LLM would otherwise cite reviews from a hospital
        # that has no summary statistics.
        eligible = set(_load_stats().keys())
        before = len(df)
        df = df[df["hospital"].isin(eligible)].reset_index(drop=True)
        dropped = before - len(df)

        _retriever = ReviewRetriever(vec, df)
        print(f"[retriever] loaded TF-IDF retriever over {len(df)} reviews "
              f"(dropped {dropped} from tiny-n hospitals)")
        return _retriever
    except Exception as e:
        print(f"[retriever] WARN: failed to load: {e}")
        return None


def retrieve(query: str,
             k: int = 10,
             filter_location: Optional[str] = None,
             filter_hospital: Optional[str] = None,
             filter_aspect:   Optional[str] = None,
             filter_label:    Optional[str] = None) -> List[Dict]:
    r = load()
    if r is None:
        return []
    return r.retrieve(query, k=k,
                      filter_location=filter_location,
                      filter_hospital=filter_hospital,
                      filter_aspect=filter_aspect,
                      filter_label=filter_label)


# Load at import time (once)
load()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for q, kwargs in [
        ("best doctors",   dict(filter_aspect="doctor", filter_label="positive")),
        ("long wait",      dict(filter_aspect="waiting_time", filter_label="negative")),
        ("clean rooms",    dict(filter_aspect="cleanliness")),
        ("expensive fees", dict(filter_aspect="cost", filter_label="negative")),
    ]:
        print(f"\n[Q] {q}  {kwargs}")
        for r in retrieve(q, k=3, **kwargs):
            print(f"  ({r['similarity']:.2f}) {r['hospital'][:35]:35s}  "
                  f"{r['text'][:80]}")
