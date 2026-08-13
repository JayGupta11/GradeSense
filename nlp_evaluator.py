import re
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# lazy-loaded sentence-transformers model
_embedder = None
_fallback_warned = False

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def semantic_similarity(student_answer: str, model_answer: str) -> Tuple[float, str]:
    try:
        model = _get_embedder()
        embeddings = model.encode([student_answer, model_answer])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(max(0.0, min(1.0, sim))), "sentence-transformers"
    except ImportError:
        pass
    except Exception as e:
        global _fallback_warned
        if not _fallback_warned:
            print(f"[nlp_evaluator] sentence-transformers unavailable ({type(e).__name__}), using TF-IDF fallback for this session...")
            _fallback_warned = True

    docs = [clean_text(student_answer), clean_text(model_answer)]
    if not docs[0] or not docs[1]:
        return 0.0, "tfidf"
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return float(sim), "tfidf"

def keyword_match(student_answer: str, keywords: List[str]) -> dict:
    student_clean = clean_text(student_answer)

    matched, missing = [], []
    for kw in keywords:
        kw_clean = clean_text(kw)
        if kw_clean and kw_clean in student_clean:
            matched.append(kw)
        else:
            missing.append(kw)

    coverage = len(matched) / len(keywords) if keywords else 1.0
    return {
        "matched": matched,
        "missing": missing,
        "coverage": round(coverage, 3),
    }

def extract_keywords_from_model_answer(model_answer: str, top_n: int = 8) -> List[str]:
    cleaned = clean_text(model_answer)
    if not cleaned:
        return []

    vectorizer = TfidfVectorizer(max_features=top_n)
    vectorizer.fit([cleaned])
    return list(vectorizer.get_feature_names_out())

def answer_length_ratio(student_answer: str, model_answer: str) -> float:
    s_len = len(student_answer.split())
    m_len = len(model_answer.split()) or 1
    return round(min(s_len / m_len, 1.5), 3)  # capped so verbosity doesn't over-reward