"""
Stage 2 - Data Preprocessing.

Builds a clean, tokenised, language-aware version of the dataset, with emoji
sentiment markers preserved as features. Handles four language modes:
English (NLTK), Malay (Sastrawi), Chinese (jieba), and Manglish / Rojak.

Improvements over R5/R6/R16 (referenced in References/references.md):
  - Multi-language pipeline instead of English-only stopwords/stemming
  - Emojis converted to sentiment tokens instead of being stripped out
  - Original and cleaned text preserved side-by-side for honest reporting

Input  (auto-detected, preferring the final labelled dataset):
    Data/raw/healthcare_final.csv     (preferred)
    Data/raw/healthcare_raw.csv       (fallback for development)

Output:
    Data/processed/healthcare_cleaned.csv
    Data/processed/preprocessing_stats.txt
"""

import os
import re
import sys
from collections import Counter

import pandas as pd
import emoji as emojilib

# ─── OPTIONAL LANGUAGE-SPECIFIC IMPORTS ──────────────────────────────────────

try:
    import nltk
    from nltk.corpus import stopwords as nltk_sw
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    HAVE_NLTK = True
except ImportError:
    HAVE_NLTK = False

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    HAVE_SASTRAWI = True
except ImportError:
    HAVE_SASTRAWI = False

try:
    import jieba
    HAVE_JIEBA = True
except ImportError:
    HAVE_JIEBA = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    HAVE_LANGDETECT = True
except ImportError:
    HAVE_LANGDETECT = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── PATHS ───────────────────────────────────────────────────────────────────

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
INPUT_PREFERRED = os.path.join(BASE, "Data", "raw", "healthcare_final.csv")
INPUT_FALLBACK  = os.path.join(BASE, "Data", "raw", "healthcare_raw.csv")
PROC            = os.path.join(BASE, "Data", "processed")
OUT_CSV         = os.path.join(PROC, "healthcare_cleaned.csv")
OUT_STATS       = os.path.join(PROC, "preprocessing_stats.txt")

# ─── NLTK SETUP ──────────────────────────────────────────────────────────────

def ensure_nltk_data():
    if not HAVE_NLTK:
        return
    needed = [
        ("corpora/stopwords",      "stopwords"),
        ("tokenizers/punkt",       "punkt"),
        ("tokenizers/punkt_tab",   "punkt_tab"),
        ("corpora/wordnet",        "wordnet"),
        ("corpora/omw-1.4",        "omw-1.4"),
    ]
    for path, name in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  downloading nltk:{name}")
            nltk.download(name, quiet=True)

# ─── CLEANING REGEX ──────────────────────────────────────────────────────────

URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_RE    = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"@\w+")
EMAIL_RE   = re.compile(r"\S+@\S+\.\S+")
WS_RE      = re.compile(r"\s+")

def basic_clean(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()

# ─── EMOJI -> SENTIMENT TOKEN ────────────────────────────────────────────────

NEG_EMOJI_KEYS = {
    "angry", "rage", "cry", "sob", "weep", "tear", "frown", "tired",
    "weary", "disappoint", "broken", "vomit", "sick", "skull",
    "pleading", "thumbs_down", "poo", "middle_finger", "anguish",
    "confounded", "persevering", "loudly_crying",
}
POS_EMOJI_KEYS = {
    "smile", "grin", "joy", "laugh", "heart", "love", "kiss", "hug",
    "thumbs_up", "star", "sparkle", "fire", "party", "beaming",
    "blush", "ok_hand", "clap", "wave", "raised_hands",
    "hundred_points", "smiling", "slightly_smiling", "pray",
    "rolling_on_the_floor",
}

def emoji_sentiment(ch: str) -> str:
    name = emojilib.demojize(ch).strip(":").lower()
    for kw in NEG_EMOJI_KEYS:
        if kw in name:
            return "negemoji"
    for kw in POS_EMOJI_KEYS:
        if kw in name:
            return "posemoji"
    return "neuemoji"


def extract_and_convert_emojis(text: str):
    """Return (text_with_emoji_tokens, list_of_original_emojis)."""
    found = []
    chars = []
    for ch in text:
        if ch in emojilib.EMOJI_DATA:
            found.append(ch)
            chars.append(" " + emoji_sentiment(ch) + " ")
        else:
            chars.append(ch)
    return WS_RE.sub(" ", "".join(chars)).strip(), found

# ─── LANGUAGE DETECTION ──────────────────────────────────────────────────────

MALAY_HINTS = {
    "yang", "dan", "tak", "tidak", "saya", "untuk", "dengan", "ini", "itu",
    "sangat", "boleh", "ada", "kena", "doktor", "hospital", "klinik",
    "ubat", "sakit", "baik", "buruk", "lambat", "cepat", "ramah", "mesra",
    "tunggu", "rawatan", "pesakit", "jururawat", "puas", "hati", "teruk",
    "bagus", "lah", "je", "dah", "nak", "kalau", "macam", "memang",
    "tetapi", "kerana", "saja",
}

def detect_language(text: str) -> str:
    if len(re.findall(r"[一-鿿]", text)) >= 3:
        return "chinese"
    words = set(re.findall(r"[a-z']+", text.lower()))
    overlap = len(words & MALAY_HINTS)

    base = "unknown"
    if HAVE_LANGDETECT:
        try:
            code = detect(text)
            base = {
                "en": "english",
                "id": "malay", "ms": "malay",
                "zh-cn": "chinese", "zh-tw": "chinese", "zh": "chinese",
            }.get(code, code)
        except Exception:
            base = "unknown"

    if base == "english" and overlap >= 2:
        return "mixed"
    if base in ("malay", "id"):
        return "malay"
    if base == "unknown" and overlap >= 2:
        return "malay"
    if base in ("english", "chinese", "malay"):
        return base
    return "other"

# ─── STOPWORDS  (lazy-loaded singletons) ─────────────────────────────────────

ZH_STOPWORDS = set("""
的 了 在 是 我 有 和 就 不 人 都 一 上 也 很 到 说 要 去 你 会 着 看 好
自己 这 那 它 他 她 我们 你们 他们 但是 因为 所以 而且 而 又 然后 已经
还有 还是 这个 那个 这些 那些 怎么 什么 哪里 为什么 可以 应该 给 跟
把 对 让 才 都是 还 比较 真的 觉得 知道 想 一些 时候 现在 就是 没有
吗 呢 啊 吧 嗯 哦 喔 哈 哈哈 嗨 哇 嘛 嘞 啦 噢 唉 哎 呀 哟 一个
""".split())

_en_sw = None
_ms_sw = None

def en_stopwords():
    global _en_sw
    if _en_sw is None and HAVE_NLTK:
        _en_sw = set(nltk_sw.words("english"))
    return _en_sw or set()

def ms_stopwords():
    global _ms_sw
    if _ms_sw is None and HAVE_SASTRAWI:
        _ms_sw = set(StopWordRemoverFactory().get_stop_words())
    return _ms_sw or set()

# ─── STEMMER / LEMMATIZER  (lazy-loaded singletons) ──────────────────────────

_lemmatizer = None
_ms_stemmer = None

def en_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None and HAVE_NLTK:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer

def ms_stemmer():
    global _ms_stemmer
    if _ms_stemmer is None and HAVE_SASTRAWI:
        _ms_stemmer = StemmerFactory().create_stemmer()
    return _ms_stemmer

# ─── PER-LANGUAGE PIPELINES ──────────────────────────────────────────────────

EMOJI_TOKENS = {"posemoji", "negemoji", "neuemoji"}

def _is_keep_token(t: str) -> bool:
    # drop punctuation-only tokens and 1-character non-CJK noise
    if not re.search(r"[a-z0-9一-鿿]", t.lower()):
        return False
    # single Chinese characters are meaningful; single Latin letters usually
    # are not (drops residue like "u", "s", "'s", "a")
    if len(t) < 2 and not re.search(r"[一-鿿]", t):
        return False
    return True


def process_english(text: str):
    text = text.lower()
    tokens = word_tokenize(text) if HAVE_NLTK else re.findall(r"[a-z']+|posemoji|negemoji|neuemoji", text)
    sw = en_stopwords()
    lem = en_lemmatizer()
    out = []
    for t in tokens:
        if t in EMOJI_TOKENS:
            out.append(t); continue
        if not _is_keep_token(t):
            continue
        if t in sw:
            continue
        # WordNetLemmatizer can over-stem ("us" -> "u"). Re-check the result
        # so lemmatised single-letter / stopword forms are dropped.
        if lem:
            t = lem.lemmatize(t)
            if not _is_keep_token(t) or t in sw:
                continue
        out.append(t)
    return out


def process_malay(text: str):
    text = text.lower()
    if HAVE_SASTRAWI:
        text = ms_stemmer().stem(text)
    tokens = re.findall(r"[a-z']+|posemoji|negemoji|neuemoji", text)
    sw = ms_stopwords() | en_stopwords()
    return [t for t in tokens if (t in EMOJI_TOKENS) or (_is_keep_token(t) and t not in sw)]


def process_chinese(text: str):
    if HAVE_JIEBA:
        tokens = list(jieba.cut(text))
    else:
        tokens = re.findall(r"[一-鿿]+|[a-z']+|posemoji|negemoji|neuemoji",
                             text.lower())
    out = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t in EMOJI_TOKENS:
            out.append(t); continue
        if t in ZH_STOPWORDS:
            continue
        if not _is_keep_token(t):
            continue
        out.append(t)
    return out


def process_mixed(text: str):
    # Manglish: route through the Malay pipeline (Sastrawi tolerates English
    # tokens; English stopwords are also removed in process_malay)
    return process_malay(text)


def process_other(text: str):
    text = text.lower()
    tokens = re.findall(r"[a-z']+|[一-鿿]+|posemoji|negemoji|neuemoji",
                        text)
    sw = en_stopwords()
    return [t for t in tokens if (t in EMOJI_TOKENS) or (_is_keep_token(t) and t not in sw)]

# ─── FULL PIPELINE ───────────────────────────────────────────────────────────

def process_text(text: str) -> dict:
    text = basic_clean(str(text))
    text, emojis = extract_and_convert_emojis(text)
    lang = detect_language(text)

    if   lang == "english": tokens = process_english(text)
    elif lang == "malay":   tokens = process_malay(text)
    elif lang == "chinese": tokens = process_chinese(text)
    elif lang == "mixed":   tokens = process_mixed(text)
    else:                   tokens = process_other(text)

    return {
        "text_cleaned": " ".join(tokens),
        "tokens":       tokens,
        "language":     lang,
        "emojis_found": emojis,
    }

# ─── MAIN ────────────────────────────────────────────────────────────────────

def find_input():
    if os.path.exists(INPUT_PREFERRED):
        return INPUT_PREFERRED, os.path.basename(INPUT_PREFERRED)
    if os.path.exists(INPUT_FALLBACK):
        return INPUT_FALLBACK, os.path.basename(INPUT_FALLBACK)
    sys.exit("[!] No input CSV found. Expected one of:\n"
             f"    {INPUT_PREFERRED}\n    {INPUT_FALLBACK}")


def main():
    if not HAVE_NLTK:
        print("[!] nltk missing -> pip install nltk")
    if not HAVE_SASTRAWI:
        print("[!] Sastrawi missing -> pip install Sastrawi")
    if not HAVE_JIEBA:
        print("[!] jieba missing -> pip install jieba")

    ensure_nltk_data()

    path, name = find_input()
    print(f"[1] Loading {name}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"    {len(df)} rows")

    if "text" not in df.columns:
        sys.exit("[!] Input CSV needs a 'text' column")

    print("[2] Preprocessing...")
    results = []
    for i, t in enumerate(df["text"].astype(str), 1):
        results.append(process_text(t))
        if i % 200 == 0 or i == len(df):
            print(f"    {i}/{len(df)}", flush=True)

    df["text_original"] = df["text"]
    df["text_cleaned"]  = [r["text_cleaned"] for r in results]
    df["tokens"]        = [" ".join(r["tokens"]) for r in results]
    df["language"]      = [r["language"] for r in results]
    df["emojis_found"]  = ["".join(r["emojis_found"]) for r in results]

    keep_cols = [c for c in [
        "source", "hospital", "stars", "label", "verified",
        "text_original", "text_cleaned", "tokens",
        "language", "emojis_found",
    ] if c in df.columns]
    out = df[keep_cols]

    os.makedirs(PROC, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[3] Saved -> {OUT_CSV}  ({len(out)} rows)")

    # ── stats report ──
    print("[4] Building stats report")
    lines = [f"Input file        : {name}",
             f"Total rows        : {len(out)}"]
    lines.append("\nLanguage distribution:")
    for lang, c in Counter(out["language"]).most_common():
        pct = 100 * c / len(out)
        lines.append(f"  {lang:<10} {c:>5}  ({pct:.1f}%)")

    orig_len  = out["text_original"].astype(str).str.len()
    clean_len = out["text_cleaned"].astype(str).str.len()
    lines.append("\nText length (characters):")
    lines.append(f"  original  mean={orig_len.mean():.0f}  max={orig_len.max()}")
    lines.append(f"  cleaned   mean={clean_len.mean():.0f}  max={clean_len.max()}")

    n_emoji = (out["emojis_found"].astype(str).str.len() > 0).sum()
    lines.append(f"\nReviews containing emoji: {n_emoji}")

    all_tokens = Counter()
    for t in out["tokens"].astype(str):
        all_tokens.update(t.split())
    lines.append("\nTop 30 tokens overall (after cleaning):")
    for tok, c in all_tokens.most_common(30):
        lines.append(f"  {tok:<22} {c}")

    if "label" in out.columns:
        for lab in ("positive", "neutral", "negative"):
            sub = out[out["label"] == lab]
            if len(sub) == 0:
                continue
            cnt = Counter()
            for t in sub["tokens"].astype(str):
                cnt.update(t.split())
            lines.append(f"\nTop 15 tokens for [{lab}] ({len(sub)} reviews):")
            for tok, c in cnt.most_common(15):
                lines.append(f"  {tok:<22} {c}")

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    saved -> {OUT_STATS}")

    print("\nDone. Next: Stage 3 (Knowledge Representation).")


if __name__ == "__main__":
    main()
