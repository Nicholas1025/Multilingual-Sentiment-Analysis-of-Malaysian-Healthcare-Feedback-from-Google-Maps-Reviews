"""
Stage 5 (Advanced Feature) - Aspect-Based Sentiment Analysis.

Healthcare reviews mention many *aspects* (doctor, waiting time, cost...) and
the overall sentiment hides the per-aspect picture. This module breaks a
review into aspect-bearing sentences and reports a sentiment per aspect.

Improves over R11 (patient feedback framework + topic modelling) because
topics are *unsupervised* and not tied to a fixed service vocabulary - here
we use a defined healthcare aspect lexicon mapped directly to actionable
service dimensions.

Designed as a library, used by Stage 6 (FastAPI deployment).
"""

import re
from collections import Counter
from typing import Callable, Dict, List


# Healthcare aspects with multilingual keyword lexicon.
# Keys are lowercase substrings; matched as whole-word for Latin script.
ASPECTS: Dict[str, List[str]] = {
    "doctor": [
        "doctor", "doctors", "dr", "dr.", "physician", "specialist",
        "consultant", "surgeon", "doktor", "pakar", "sinseh",
        "医生", "医师", "大夫",
    ],
    "nurse_staff": [
        "nurse", "nurses", "staff", "receptionist", "attendant",
        "jururawat", "kakitangan", "pekerja", "kaunter",
        "护士", "员工", "护理", "服务员",
    ],
    "waiting_time": [
        "wait", "waiting", "waited", "queue", "slow", "fast", "quick",
        "delay", "delayed", "long", "hours", "minutes",
        "tunggu", "menunggu", "lambat", "cepat", "lama",
        "等", "等候", "等待", "时间", "排队", "慢", "快",
    ],
    "facilities": [
        "facility", "facilities", "room", "rooms", "ward", "bed",
        "clean", "dirty", "comfortable", "parking", "lift", "elevator",
        "toilet", "washroom", "aircond", "ac", "wifi",
        "kemudahan", "bilik", "wad", "bersih", "kotor", "tempat",
        "设施", "房间", "病房", "床", "干净", "脏",
    ],
    "cost": [
        "cost", "price", "prices", "expensive", "cheap", "affordable",
        "fee", "fees", "charge", "charges", "bill", "payment", "money",
        "kos", "harga", "mahal", "murah", "bayar", "yuran",
        "费用", "价钱", "价格", "贵", "便宜", "钱", "收费",
    ],
    "treatment": [
        "treatment", "medicine", "medication", "drug", "drugs", "therapy",
        "operation", "surgery", "procedure", "diagnosis", "consultation",
        "rawatan", "ubat", "operasi", "rawatan", "diagnos",
        "治疗", "药", "药物", "手术", "诊断", "诊治",
    ],
    "cleanliness": [
        "clean", "cleanliness", "dirty", "hygienic", "hygiene", "smell",
        "tidy", "messy", "bersih", "kotor", "kebersihan",
        "干净", "卫生", "脏", "整洁",
    ],
}

LATIN_RE   = re.compile(r"[A-Za-z]")
CJK_RE     = re.compile(r"[一-鿿]")
SENT_SPLIT = re.compile(r"(?<=[\.\!\?。！？])\s+|\n+")

# Clause-boundary markers across our four languages.  Splitting here lets
# "Doctor good, nurse bad, but very clean" be analysed as three clauses
# with different sentiment instead of one mixed blob.
CONTRAST_RE = re.compile(
    r"(?:[,;，；]"                                                # ASCII + CJK comma/semicolon
    r"|\bbut\b|\bhowever\b|\bthough\b|\balthough\b|\byet\b"        # English contrastives
    r"|\btapi\b|\btetapi\b|\bnamun\b|\bwalaupun\b"                 # Malay contrastives
    r"|但是|可是|不过|然而|而|却)",                                   # Chinese contrastives
    re.IGNORECASE,
)


# ─── HELPERS ────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    """Split a review into sentences in a language-agnostic way."""
    parts = SENT_SPLIT.split(str(text))
    return [p.strip() for p in parts if p and p.strip()]


def split_clauses(text: str) -> List[str]:
    """Like split_sentences, but also splits each sentence on contrastive
    conjunctions (but, however, tapi, 但是, ...) so that each aspect gets
    only its own clause for sentiment scoring."""
    out = []
    for sent in split_sentences(text):
        parts = CONTRAST_RE.split(sent)
        out.extend(p.strip() for p in parts if p and p.strip())
    return out


def _contains_keyword(sentence: str, keyword: str) -> bool:
    """Whole-word for Latin keywords, substring for CJK keywords."""
    if CJK_RE.search(keyword):
        return keyword in sentence
    pattern = r"(?<![A-Za-z])" + re.escape(keyword) + r"(?![A-Za-z])"
    return re.search(pattern, sentence, flags=re.IGNORECASE) is not None


def detect_aspects(text: str) -> List[str]:
    """Return the set of aspects mentioned anywhere in the text."""
    text = str(text)
    out = []
    for aspect, words in ASPECTS.items():
        if any(_contains_keyword(text, w) for w in words):
            out.append(aspect)
    return out


def sentences_for_aspect(text: str, aspect: str) -> List[str]:
    """Return only the clauses that mention `aspect`. Uses split_clauses so
    contrastive sentences ("doctor good but nurse bad") split correctly."""
    words = ASPECTS[aspect]
    return [c for c in split_clauses(text)
            if any(_contains_keyword(c, w) for w in words)]


# ─── MAIN ENTRY POINT (used by the FastAPI app) ─────────────────────────────

def aspect_sentiments(text: str,
                      classify_fn: Callable[[str], str]) -> Dict[str, str]:
    """Return {aspect -> 'positive'/'neutral'/'negative'} for every aspect
    mentioned in `text`. `classify_fn` is the sentiment model from Stage 4
    (it takes a string and returns one of the three labels)."""
    out: Dict[str, str] = {}
    for aspect in detect_aspects(text):
        sents = sentences_for_aspect(text, aspect)
        if not sents:
            continue
        labels = [classify_fn(s) for s in sents]
        # majority vote, with negative > positive > neutral on ties
        tally = Counter(labels)
        priority = ["negative", "positive", "neutral"]
        ranked = sorted(tally.items(),
                        key=lambda kv: (-kv[1], priority.index(kv[0])
                                         if kv[0] in priority else 99))
        out[aspect] = ranked[0][0]
    return out


# ─── CLI DEMO ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = (
        "The doctor was very friendly and explained everything clearly. "
        "However, the waiting time was terrible - we waited almost 3 hours. "
        "The facilities are clean and modern but parking is limited. "
        "Quite expensive but worth it."
    )

    def dummy_classify(s: str) -> str:
        sl = s.lower()
        if any(w in sl for w in ["bad","terrible","worst","slow","limited",
                                 "expensive","dirty"]):
            return "negative"
        if any(w in sl for w in ["good","great","friendly","clean","worth"]):
            return "positive"
        return "neutral"

    print("Aspects detected:", detect_aspects(demo))
    print("Per-aspect sentiment:", aspect_sentiments(demo, dummy_classify))
