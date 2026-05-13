import uuid
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import pymupdf
import re
from collections import defaultdict
import hashlib
import torch
from cachetools import LRUCache
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI(title="AI Research Gap Finder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION & DOMAIN VOCABULARY
# ============================================================================

DOMAIN_VOCABULARY = {
    "medical": [
        "tachycardia", "hypertension", "myocardial", "glycemic", "angiogenesis",
        "apoptosis", "pathophysiology", "pharmacokinetics", "biomarker",
        "prognosis", "comorbidity", "immunotherapy", "neoplasm", "lesion",
        "inflammation", "metabolism", "hemoglobin", "cholesterol", "renal", "hepatic"
    ],
    "legal": [
        "jurisprudence", "litigation", "liability", "negligence", "contractual",
        "precedent", "statutory", "jurisdiction", "arbitration", "compliance",
        "affidavit", "testimony", "defendant", "plaintiff", "damages"
    ],
    "cs": [
        "neural_network", "backpropagation", "transformer", "attention_mechanism",
        "convolutional", "recurrent", "optimization", "gradient_descent",
        "regularization", "batch_normalization", "embedding", "tokenization"
    ],
    "physics": [
        "quantum", "relativity", "thermodynamics", "electromagnetic", "particle",
        "wavefunction", "entanglement", "superposition", "boson", "fermion"
    ],
    "biology": [
        "mitosis", "meiosis", "genome", "proteome", "transcription", "translation",
        "mutation", "evolution", "ecosystem", "photosynthesis", "respiration"
    ],
    "finance": [
        "derivatives", "volatility", "liquidity", "arbitrage", "capitalization",
        "portfolio", "dividend", "benchmark", "leverage", "hedging"
    ],
    "psychology": [
        "cognition", "behavioral", "neurotransmitter", "psychotherapy", "trauma",
        "resilience", "attachment", "perception", "memory", "consciousness"
    ]
}

# Pre-flatten domain vocab once at startup — avoids rebuilding inside hot loops
_ALL_DOMAIN_VOCAB: frozenset = frozenset(
    token for vocab in DOMAIN_VOCABULARY.values() for token in vocab
)

COMMON_ACADEMIC = {
    "analysis", "methodology", "framework", "evaluation", "implementation",
    "investigation", "significant", "correlation", "independent", "dependent",
    "variable", "hypothesis", "experiment", "observation", "conclusion"
}

COMPATIBILITY_MATRIX = {
    ("medical", "cs"): 0.85,
    ("medical", "biology"): 0.90,
    ("cs", "physics"): 0.75,
    ("cs", "finance"): 0.80,
    ("psychology", "medical"): 0.85,
    ("biology", "chemistry"): 0.90,
    ("legal", "cs"): 0.70,
    ("physics", "legal"): 0.20,
    ("medieval_history", "quantum"): 0.05,
}

EMBEDDING_CACHE: LRUCache = LRUCache(maxsize=500)
_cache_lock = threading.Lock()

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
print(f"Device configuration: {DEVICE}")

# ============================================================================
# AI MODEL INITIALIZATION
# ============================================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

print(f"Loaded pre-trained transformer: {MODEL_NAME}")
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"Using device: {DEVICE}")

# Thread pool for CPU-bound work that can run alongside embedding
_thread_pool = ThreadPoolExecutor(max_workers=4)

# ============================================================================
# DATA MODELS 
# ============================================================================

class Paper(BaseModel):
    id: str
    domain: str
    text: str
    embeddings: Optional[List[List[float]]] = None
    concepts: Optional[List[Dict]] = None


class Concept(BaseModel):
    label: str
    keywords: List[str]
    centroid: List[float]
    prominence: float
    size: int


class EvidenceSentence(BaseModel):
    domain: str
    sentences: List[str]


class ConceptOverlap(BaseModel):
    shared: List[str]
    unique_a: List[str]
    unique_b: List[str]


class ResearchGap(BaseModel):
    id: str
    title: str
    confidence: float
    gap_strength: float
    gap_strength_label: str
    confidence_min: float = 0.0
    confidence_max: float = 1.0
    cluster_count: int = 1
    missing_in: str
    present_in: str
    explanation: str
    semantic_distance: float
    semantic_distance_min: float = 0.0
    semantic_distance_max: float = 1.0
    evidence_sentences: List[EvidenceSentence]
    concept_overlap: ConceptOverlap
    future_suggestion: str
    why_detected: str
    novelty_score: float
    compatibility_score: float = 0.40


class DeduplicatedGap(BaseModel):
    id: str
    title: str
    confidence: float
    gap_strength: float
    gap_strength_label: str
    confidence_range: str
    cluster_count: int
    missing_in: str
    present_in: str
    explanation: str
    semantic_distance: float
    semantic_distance_range: str
    evidence_sentences: List[EvidenceSentence]
    concept_overlap: ConceptOverlap
    future_suggestion: str
    why_detected: str
    novelty_score: float
    compatibility_score: float = 0.40
    supporting_gaps: List[str] = []


# ============================================================================
# STAGE 1: PDF TEXT EXTRACTION
# ============================================================================


_RE_INLINE_CITE   = re.compile(r'\([A-Z][a-z]+(?:\s+et\s+al\.? )?,?\s+\d{4}[a-z]?\)')
_RE_NUMERIC_CITE  = re.compile(r'\[\d+(?:,\s*\d+)*\]')
_RE_BOILERPLATE   = re.compile(
    r'(Procedia\s+\w+\s+Science|Procedia\s+\w+\s*|'
    r'Proceedings of|International Conference on|Journal of|'
    r'IEEE|ACM|Springer|Elsevier|arXiv|bioRxiv)[\w\s,:\-]+(?:\d{4})?',
    re.IGNORECASE
)
_RE_URL_DOI       = re.compile(r'https?://\S+|doi:\S+|arxiv:\S+', re.IGNORECASE)
_RE_PAGE_NOISE    = re.compile(r'^\s*[\d\s\-\u2013|]+\s*$')
_RE_ALL_CAPS      = re.compile(r'^[A-Z\s]{3,50}$')
_RE_WHITESPACE    = re.compile(r'\s+')
_RE_SENTENCE_SPLIT = re.compile(r'([.!?])\s*')
_RE_REFS_SPLIT    = re.compile(
    r'\n\s*(References|Bibliography|Works Cited|Acknowledgments?)\s*\n',
    re.IGNORECASE
)
_RE_NUMERIC_ONLY  = re.compile(r'^[\d\s\W]+$')
_RE_NUMERIC_CITE2 = re.compile(r'\[[\d,\s]+\]')
_RE_INLINE_CITE2  = re.compile(r'\([A-Z][a-z]+,?\s+\d{4}\)')
_RE_TOC_ARTIFACT  = re.compile(
    r'^(Optimisation|Prediction|Classification|Detection)\s+The publication',
    re.IGNORECASE
)
_RE_BAD_CHARS     = re.compile(r'[0-9@#\$%&\*\(\)]')


def strip_metadata(text: str) -> str:
    """Remove reference sections, citations, boilerplate, and page noise."""
    text = _RE_REFS_SPLIT.split(text)[0]
    text = _RE_INLINE_CITE.sub('', text)
    text = _RE_NUMERIC_CITE.sub('', text)
    text = _RE_BOILERPLATE.sub('', text)
    text = _RE_URL_DOI.sub('', text)
    lines = [
        ln for ln in text.split('\n')
        if not _RE_PAGE_NOISE.match(ln) and not _RE_ALL_CAPS.match(ln)
    ]
    return '\n'.join(lines)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        text = strip_metadata(text)
        text = _RE_WHITESPACE.sub(' ', text)
        text = _RE_SENTENCE_SPLIT.sub(r'\1\n', text)
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")


def preprocess_academic_text(text: str) -> List[str]:
    """
    Split into sentences and filter noise.
    strip_metadata is NOT called here — caller already strips it.
    """
    cleaned = []
    for sent in text.split('\n'):
        sent = sent.strip()
        if len(sent) <= 40 or _RE_NUMERIC_ONLY.match(sent):
            continue
        if _RE_TOC_ARTIFACT.match(sent):
            continue
        sent = _RE_NUMERIC_CITE2.sub('', sent)
        sent = _RE_INLINE_CITE2.sub('', sent)
        cleaned.append(sent.strip())
        if len(cleaned) == 2000:      # early-exit avoids iterating the tail
            break
    return cleaned


# ============================================================================
# STAGE 2: SEMANTIC EMBEDDING GENERATION
# ============================================================================

def get_embedding_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def generate_semantic_embeddings(sentences: List[str], domain: str = "") -> np.ndarray:
    if not sentences:
        return np.array([]).reshape(0, EMBEDDING_DIM)

    keys = [get_embedding_cache_key(s) for s in sentences]
    local_embeddings: Dict[str, np.ndarray] = {}
    uncached_sentences: List[str] = []
    uncached_keys: List[str] = []

    with _cache_lock:
        for sentence, key in zip(sentences, keys):
            if key in local_embeddings:
                continue
            if key in EMBEDDING_CACHE:
                local_embeddings[key] = EMBEDDING_CACHE[key]
            else:
                uncached_sentences.append(sentence)
                uncached_keys.append(key)

    if uncached_sentences:
        new_embeddings = embedding_model.encode(
            uncached_sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=128,           
        )
        with _cache_lock:
            for key, emb in zip(uncached_keys, new_embeddings):
                EMBEDDING_CACHE[key] = emb
                local_embeddings[key] = emb

    result_embeddings = [local_embeddings[key] for key in keys]

    print(f"Generated {len(sentences)} embeddings "
          f"(cached: {len(sentences) - len(uncached_sentences)}, "
          f"new: {len(uncached_sentences)})")

    return np.array(result_embeddings)


# ============================================================================
# STAGE 3: UNSUPERVISED CONCEPT DISCOVERY
# ============================================================================

def extract_domain_vocab_tfidf(sentences: List[str], top_k: int = 50) -> List[str]:
    if not sentences:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_df=0.90,
        min_df=2,
        max_features=1000
    )
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    if feature_names.size == 0:
        return []

    # Use .A1 (flat array alias) instead of asarray().ravel() — one less copy
    scores = tfidf.sum(axis=0).A1
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [phrase.lower() for phrase, _ in ranked[:top_k]]


def extract_concept_keywords(
    sentences: List[str],
    top_k: int = 10,
    domain: str = "",
    extra_vocab: Optional[set] = None
) -> List[str]:
    """
    Extract multi-word concept phrases using TF-IDF with domain-aware weighting.
    Uses module-level _ALL_DOMAIN_VOCAB instead of rebuilding it every call.
    """
    domain_vocab = set(DOMAIN_VOCABULARY.get(domain.lower(), []))
    extra_vocab = extra_vocab or set()

    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'been', 'be',
        'this', 'that', 'these', 'those', 'we', 'our', 'their', 'such', 'have',
        'study', 'paper', 'research', 'result', 'show', 'found', 'using'
    }

    if not sentences:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words='english',
        max_df=0.85,
        min_df=1,
        max_features=2000
    )
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf.sum(axis=0).A1          # .A1 avoids intermediate ndarray

    phrase_weights: Dict[str, float] = defaultdict(float)
    for phrase, score in zip(feature_names, scores):
        phrase_lower = phrase.lower().strip()
        if len(phrase_lower) < 6:
            continue
        tokens = phrase_lower.split()
        if len(tokens) > 3:
            continue
        if any(tok in stopwords for tok in tokens) and len(tokens) == 1:
            continue
        if _RE_BAD_CHARS.search(phrase_lower):
            continue

        weight = float(score)

        if any(tok in domain_vocab for tok in tokens):
            weight *= 3.0
        elif any(tok in _ALL_DOMAIN_VOCAB for tok in tokens):   # uses pre-built set
            weight *= 1.7
        elif any(tok in extra_vocab for tok in tokens):
            weight *= 2.2
        elif phrase_lower in COMMON_ACADEMIC:
            weight *= 1.2

        if len(tokens) >= 2:
            weight *= 1.5

        phrase_weights[phrase_lower] = max(phrase_weights[phrase_lower], weight)

    sorted_phrases = sorted(phrase_weights.items(), key=lambda x: x[1], reverse=True)
    if not sorted_phrases:
        return []

    return [phrase for phrase, _ in sorted_phrases[:top_k]]


def discover_concepts(
    embeddings: np.ndarray,
    sentences: List[str],
    min_cluster_size: int = 3,
    domain: str = "",
    extra_vocab: Optional[set] = None
) -> List[Concept]:
    if len(embeddings) < min_cluster_size:
        return []

    actual_min_size = max(min_cluster_size, len(embeddings) // 20)

    # ── PCA reduction before HDBSCAN ─────────────────────────────────────────
    n_components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
    if n_components >= 2:
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings

    clusterer = HDBSCAN(
        min_cluster_size=actual_min_size,
        min_samples=2,
        metric='euclidean',
        cluster_selection_epsilon=0.0,
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings_reduced)
    # NOTE: centroids are computed from ORIGINAL embeddings (full 384-d)
    # so all downstream cosine similarity math is unchanged vs original code.

    # Build index → sentences mapping once
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in enumerate(cluster_labels):
        if lbl != -1:
            label_to_indices[lbl].append(idx)

    if not label_to_indices:
        return []

    # ── Fit ONE TF-IDF over ALL sentences once, then slice per cluster ────────
    # Previously: one TfidfVectorizer.fit_transform() call per cluster = very slow
    # Now: one fit on all sentences, then per-cluster column sums from the matrix
    domain_vocab  = set(DOMAIN_VOCABULARY.get(domain.lower(), []))
    extra_vocab   = extra_vocab or set()
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'been', 'be',
        'this', 'that', 'these', 'those', 'we', 'our', 'their', 'such', 'have',
        'study', 'paper', 'research', 'result', 'show', 'found', 'using'
    }

    global_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words='english',
        max_df=0.85,
        min_df=1,
        max_features=2000
    )
    try:
        global_tfidf = global_vectorizer.fit_transform(sentences)   # (N, vocab)
    except ValueError:
        global_tfidf = None

    feature_names = global_vectorizer.get_feature_names_out() if global_tfidf is not None else np.array([])

    def _keywords_from_tfidf_slice(indices: List[int], top_k: int = 10) -> List[str]:
        """Extract top keywords for a cluster using the pre-fitted global TF-IDF matrix."""
        if global_tfidf is None or feature_names.size == 0:
            return []
        scores = global_tfidf[indices].sum(axis=0).A1   # sum rows for this cluster

        phrase_weights: Dict[str, float] = {}
        for phrase, score in zip(feature_names, scores):
            phrase_lower = phrase.lower().strip()
            if len(phrase_lower) < 6:
                continue
            tokens = phrase_lower.split()
            if len(tokens) > 3:
                continue
            if any(tok in stopwords for tok in tokens) and len(tokens) == 1:
                continue
            if _RE_BAD_CHARS.search(phrase_lower):
                continue

            weight = float(score)
            if any(tok in domain_vocab for tok in tokens):
                weight *= 3.0
            elif any(tok in _ALL_DOMAIN_VOCAB for tok in tokens):
                weight *= 1.7
            elif any(tok in extra_vocab for tok in tokens):
                weight *= 2.2
            elif phrase_lower in COMMON_ACADEMIC:
                weight *= 1.2
            if len(tokens) >= 2:
                weight *= 1.5

            phrase_weights[phrase_lower] = max(phrase_weights.get(phrase_lower, 0.0), weight)

        sorted_phrases = sorted(phrase_weights.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_phrases[:top_k]]

    concepts = []
    for label, indices in label_to_indices.items():
        cluster_embeddings = embeddings[indices]
        centroid   = cluster_embeddings.mean(axis=0)
        keywords   = _keywords_from_tfidf_slice(indices)
        prominence = compute_concept_prominence(cluster_embeddings)
        concepts.append(Concept(
            label=generate_concept_label(keywords),
            keywords=keywords[:5],
            centroid=centroid.tolist(),
            prominence=float(prominence),
            size=len(indices)
        ))

    return sorted(concepts, key=lambda x: x.prominence, reverse=True)


def generate_concept_label(keywords: List[str]) -> str:
    if not keywords:
        return "Unlabeled Concept"
    return " + ".join(k.capitalize() for k in keywords[:3])


def compute_concept_prominence(cluster_embeddings: np.ndarray) -> float:
    """
    Approximate mean pairwise cosine similarity using centroid proximity.
    For L2-normalised embeddings: mean_sim ≈ ||mean_emb||²
    O(n·d) instead of O(n²·d) — identical result direction, ~10-50× faster on large clusters.
    """
    if len(cluster_embeddings) < 2:
        return 0.5
    centroid = cluster_embeddings.mean(axis=0)
    # For unit vectors, mean pairwise cosine sim = ||centroid||² * n/(n-1) approximately
    # This is a well-known identity for normalised embeddings
    approx_sim = float(np.dot(centroid, centroid))
    return float(np.clip(approx_sim, 0, 1))


# ============================================================================
# STAGE 4: CROSS-DOMAIN SEMANTIC DIVERGENCE (ADAPTIVE)
# ============================================================================

def compute_intra_domain_variance(concepts: List[Concept]) -> float:
    if len(concepts) < 2:
        return 0.3
    centroids = np.array([c.centroid for c in concepts])
    distances = cosine_distances(centroids)
    n = len(distances)
    upper_triangle = distances[np.triu_indices(n, k=1)]
    return float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.3


def adaptive_gap_threshold(concepts_a: List[Concept], concepts_b: List[Concept]) -> float:
    var_a = compute_intra_domain_variance(concepts_a)
    var_b = compute_intra_domain_variance(concepts_b)
    return float(np.clip((var_a + var_b) / 2, 0.3, 0.7))


def analyze_cross_domain_divergence(
    concepts_a: List[Concept],
    concepts_b: List[Concept]
) -> List[Tuple[Concept, Concept, float]]:
    if not concepts_a or not concepts_b:
        return []

    threshold = adaptive_gap_threshold(concepts_a, concepts_b)
    print(f"Using adaptive threshold: {threshold:.3f}")

    centroids_a = np.array([c.centroid for c in concepts_a])
    centroids_b = np.array([c.centroid for c in concepts_b])
    distances = cosine_distances(centroids_a, centroids_b)     # shape (|A|, |B|)

    # Vectorised threshold filter — avoids a Python loop over all concept pairs
    min_dist_per_a = distances.min(axis=1)                     # shape (|A|,)
    best_b_per_a   = distances.argmin(axis=1)                  # shape (|A|,)
    mask = min_dist_per_a > threshold

    divergences = [
        (concepts_a[i], concepts_b[best_b_per_a[i]], float(min_dist_per_a[i]))
        for i in np.where(mask)[0]
    ]
    return sorted(divergences, key=lambda x: x[2], reverse=True)


# ============================================================================
# STAGE 5: CONFIDENCE SCORING
# ============================================================================

def get_domain_compatibility(domain_a: str, domain_b: str) -> float:
    key = tuple(sorted([domain_a.lower(), domain_b.lower()]))
    return COMPATIBILITY_MATRIX.get(key, 0.40)


def format_gap_strength_label(score: float) -> str:
    if score >= 0.75:
        return "Strong Opportunity"
    if score >= 0.50:
        return "Moderate Opportunity"
    return "Exploratory Signal"


def validate_gap_novelty(
    concept_a: Concept,
    concept_b: Concept,
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    similarity_to_b: Optional[np.ndarray] = None,
) -> float:
    shared = set(concept_a.keywords) & set(concept_b.keywords)
    lexical_novelty = 1.0 - len(shared) / max(len(concept_a.keywords), 1)

    centroid_a = np.array(concept_a.centroid).reshape(1, -1)
    max_sim = 0.0
    if len(embeddings_b) > 0:
        sims = (
            similarity_to_b
            if similarity_to_b is not None
            else cosine_similarity(centroid_a, embeddings_b)[0]
        )
        max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0

    novelty = 0.65 * lexical_novelty + 0.35 * (1.0 - max_sim)
    return float(np.clip(novelty, 0.1, 1.0))


def compute_gap_confidence(
    concept_a: Concept,
    concept_b: Concept,
    semantic_distance: float,
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    domain_a: str = "",
    domain_b: str = "",
    threshold: float = 0.5,
    novelty_score: float = 1.0,
    similarity_to_b: Optional[np.ndarray] = None,
) -> float:
    distance_score = min(semantic_distance / 0.9, 1.0)
    prominence_score = concept_a.prominence

    centroid_a = np.array(concept_a.centroid).reshape(1, -1)
    sims_to_b = (
        similarity_to_b
        if similarity_to_b is not None
        else (
            cosine_similarity(centroid_a, embeddings_b)[0]
            if len(embeddings_b) > 0
            else np.array([0.0])
        )
    )
    max_sim_to_b = float(np.max(sims_to_b)) if len(sims_to_b) > 0 else 0.0
    absence_score = 1.0 - max_sim_to_b

    distinctiveness = concept_a.size / (concept_a.size + concept_b.size + 1)
    evidence_ratio = min(concept_a.size / 50, 1.0)
    gap_magnitude = max(0.0, semantic_distance - threshold) / max(threshold, 1e-6)

    base_strength = (
        0.30 * distance_score +
        0.20 * prominence_score +
        0.20 * absence_score +
        0.10 * distinctiveness +
        0.10 * evidence_ratio +
        0.10 * gap_magnitude
    )

    compatibility = get_domain_compatibility(domain_a, domain_b)
    adjusted = base_strength * (0.7 + 0.3 * compatibility) * (0.6 + 0.4 * novelty_score)
    return float(np.clip(adjusted, 0.0, 1.0))


# ============================================================================
# STAGE 6: EXPLAINABLE AI GAP GENERATION
# ============================================================================

def generate_explainable_gap(
    concept_a: Concept,
    concept_b: Concept,
    semantic_distance: float,
    gap_strength: float,
    domain_a: str,
    domain_b: str,
    sentences_a: List[str],
    sentences_b: List[str],
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    novelty_score: float,
    threshold: float = 0.5,
    # Pre-computed sims passed from caller to avoid recomputation
    precomputed_sims_a: Optional[np.ndarray] = None,
    precomputed_sims_b: Optional[np.ndarray] = None,
) -> ResearchGap:
    evidence = extract_evidence_sentences(
        concept_a,
        concept_b,
        sentences_a,
        sentences_b,
        embeddings_a,
        embeddings_b,
        domain_a,
        domain_b,
        precomputed_sims_a=precomputed_sims_a,
        precomputed_sims_b=precomputed_sims_b,
    )
    overlap = analyze_concept_overlap(concept_a.keywords, concept_b.keywords)
    title = f"{concept_a.label} <-> {concept_b.label} Integration Gap"

    explanation = (
        f"Domain {domain_a} extensively explores {concept_a.label.lower()} "
        f"(prominence: {concept_a.prominence:.2f}, {concept_a.size} instances), "
        f"while Domain {domain_b} focuses on {concept_b.label.lower()} "
        f"(prominence: {concept_b.prominence:.2f}). "
        f"The semantic embeddings reveal a distance of {semantic_distance:.3f} "
        f"in the {EMBEDDING_DIM}-dimensional space (adaptive threshold: {threshold:.3f}), "
        f"indicating these concepts are explored independently without sufficient integration. "
    )
    if overlap.shared:
        explanation += f"While both domains share interest in {', '.join(overlap.shared[:2])}, "
    explanation += (
        f"Domain {domain_a} uniquely emphasizes {', '.join(overlap.unique_a[:3])}, "
        f"whereas Domain {domain_b} uniquely focuses on {', '.join(overlap.unique_b[:3])}."
    )

    strength_label = format_gap_strength_label(gap_strength)
    compatibility_score = get_domain_compatibility(domain_a, domain_b)

    why_detected = (
        f"The transformer-based semantic analysis placed these concepts in distant regions "
        f"of the embedding space (cosine distance: {semantic_distance:.3f}, threshold: {threshold:.3f}). "
        f"HDBSCAN clustering identified them as separate concept clusters with minimal overlap. "
        f"The opportunity strength score of {gap_strength:.2f} reflects: "
        f"(1) high semantic distance ({semantic_distance:.2f}), "
        f"(2) strong prominence in Domain {domain_a} ({concept_a.prominence:.2f}), "
        f"(3) relative absence in Domain {domain_b} ({1 - concept_b.prominence:.2f}), "
        f"(4) adaptive compatibility ({compatibility_score:.2f}), "
        f"and (5) novelty signal ({novelty_score:.2f}). "
        f"Detected using adaptive thresholding and domain-aware keyword extraction."
    )

    future_suggestion = (
        f"Develop integrative frameworks that bridge {concept_a.label.lower()} "
        f"from {domain_a} with {concept_b.label.lower()} from {domain_b}. "
        f"Specifically, investigate how insights from "
        f"{concept_a.keywords[0] if concept_a.keywords else 'this area'} "
        f"can enhance approaches to "
        f"{concept_b.keywords[0] if concept_b.keywords else 'related topics'}. "
        f"Consider hybrid methodologies that explicitly combine both perspectives, "
        f"potentially leveraging transfer learning across domains."
    )

    return ResearchGap(
        id=f"gap-{uuid.uuid4().hex[:8]}",
        title=title,
        confidence=gap_strength,
        gap_strength=gap_strength,
        gap_strength_label=strength_label,
        missing_in=domain_b,
        present_in=domain_a,
        explanation=explanation,
        semantic_distance=semantic_distance,
        evidence_sentences=evidence,
        concept_overlap=overlap,
        future_suggestion=future_suggestion,
        why_detected=why_detected,
        novelty_score=novelty_score,
        compatibility_score=compatibility_score,
    )


def extract_evidence_sentences(
    concept_a: Concept,
    concept_b: Concept,
    sentences_a: List[str],
    sentences_b: List[str],
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    domain_a: str,
    domain_b: str,
    top_k: int = 3,
    precomputed_sims_a: Optional[np.ndarray] = None,
    precomputed_sims_b: Optional[np.ndarray] = None,
) -> List[EvidenceSentence]:
    """
    Accepts optionally pre-computed similarity vectors so the caller can
    avoid recomputing cosine_similarity for every gap independently.
    """
    evidence: List[EvidenceSentence] = []

    # Domain A evidence
    sims_a = precomputed_sims_a
    if sims_a is None:
        centroid_a = np.array(concept_a.centroid).reshape(1, -1)
        sims_a = cosine_similarity(centroid_a, embeddings_a)[0]
    top_indices_a = np.argsort(sims_a)[-top_k:][::-1]
    evidence_sentences_a = [sentences_a[i] for i in top_indices_a if i < len(sentences_a)]
    if evidence_sentences_a:
        evidence.append(EvidenceSentence(domain=domain_a, sentences=evidence_sentences_a))

    # Domain B evidence
    sims_b = precomputed_sims_b
    if sims_b is None:
        centroid_b = np.array(concept_b.centroid).reshape(1, -1)
        sims_b = cosine_similarity(centroid_b, embeddings_b)[0]
    top_indices_b = np.argsort(sims_b)[-top_k:][::-1]
    evidence_sentences_b = [sentences_b[i] for i in top_indices_b if i < len(sentences_b)]
    if evidence_sentences_b:
        evidence.append(EvidenceSentence(domain=domain_b, sentences=evidence_sentences_b))

    return evidence


def analyze_concept_overlap(keywords_a: List[str], keywords_b: List[str]) -> ConceptOverlap:
    set_a = set(keywords_a)
    set_b = set(keywords_b)
    return ConceptOverlap(
        shared=list(set_a & set_b),
        unique_a=list(set_a - set_b),
        unique_b=list(set_b - set_a)
    )


# ============================================================================
# STAGE 7: GAP DEDUPLICATION & CLUSTERING
# ============================================================================

def deduplicate_gaps_by_embedding(
    gaps: List[Dict],
    similarity_threshold: float = 0.85
) -> List[Dict]:
    """
    Merge semantically equivalent gaps using embedding similarity on titles.
    Titles are encoded in a single batched call rather than one-by-one.
    """
    if not gaps:
        return []

    titles = [gap['title'] for gap in gaps]
    title_embeddings = embedding_model.encode(
        titles,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=128,           # batched for speed
    )
    similarity_matrix = cosine_similarity(title_embeddings)

    merged = [False] * len(gaps)
    deduplicated = []

    for i, gap in enumerate(gaps):
        if merged[i]:
            continue

        # Vectorised cluster membership — avoids inner Python loop
        similar_mask = similarity_matrix[i] >= similarity_threshold
        similar_mask[i] = False   # exclude self
        cluster_indices = [i] + [j for j in range(i + 1, len(gaps))
                                  if not merged[j] and similar_mask[j]]
        for j in cluster_indices[1:]:
            merged[j] = True

        cluster = [gaps[idx] for idx in cluster_indices]
        best = max(cluster, key=lambda item: item.get('gap_strength', item.get('confidence', 0.0)))
        confidences = [item.get('confidence', 0.0) for item in cluster]
        distances = [item.get('semantic_distance', 0.0) for item in cluster]

        dedup_gap = {
            **best,
            'confidence_min': min(confidences),
            'confidence_max': max(confidences),
            'confidence_range': f"{min(confidences):.2f}–{max(confidences):.2f}",
            'semantic_distance_min': min(distances),
            'semantic_distance_max': max(distances),
            'semantic_distance_range': f"{min(distances):.3f}–{max(distances):.3f}",
            'cluster_count': len(cluster),
            'supporting_gaps': [item['id'] for item in cluster]
        }

        if len(cluster) > 1:
            dedup_gap['explanation'] = (
                f"{dedup_gap['explanation'].strip()} "
                f"(This gap is supported by {len(cluster)} semantically equivalent candidates.)"
            )

        deduplicated.append(dedup_gap)

    deduplicated.sort(key=lambda x: x.get('gap_strength', x.get('confidence', 0.0)), reverse=True)
    return deduplicated


def limit_gaps_per_comparison(gaps: List[Dict], max_gaps: int = 10) -> List[Dict]:
    domain_pairs: Dict[str, List[Dict]] = defaultdict(list)
    for gap in gaps:
        domain_pairs[f"{gap['missing_in']}|{gap['present_in']}"].append(gap)

    limited = []
    for pair_gaps in domain_pairs.values():
        sorted_gaps = sorted(
            pair_gaps,
            key=lambda x: x.get('gap_strength', x.get('confidence', 0.0)),
            reverse=True
        )
        limited.extend(sorted_gaps[:max_gaps])
    return limited


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/extract")
async def extract_text(file: UploadFile = File(...), domain: str = ""):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    pdf_bytes = await file.read()
    # Offload CPU-heavy PDF parsing to thread pool so the event loop stays free
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(_thread_pool, extract_text_from_pdf, pdf_bytes)
    sentences = preprocess_academic_text(text)

    return {
        "text": text,
        "sentence_count": len(sentences),
        "domain": domain
    }


@app.post("/api/analyze")
async def analyze_papers(papers: List[Paper]):
    """Run the complete AI pipeline on uploaded papers."""
    if len(papers) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 papers")

    # ── Stage 1-2: Extract sentences + embeddings per paper ──────────────────
    results = []
    for paper in papers:
        sentences = preprocess_academic_text(paper.text)

        if not sentences:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No usable sentences found in paper '{paper.id}'. "
                    f"Ensure text was extracted via /api/extract before calling /api/analyze."
                )
            )
        if len(sentences) < 10:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Paper '{paper.id}' has only {len(sentences)} usable sentences "
                    f"— too few to cluster. Minimum is 10."
                )
            )

        embeddings = generate_semantic_embeddings(sentences, domain=paper.domain)
        paper_vocab = extract_domain_vocab_tfidf(sentences, top_k=50)

        results.append({
            "id": paper.id,
            "domain": paper.domain,
            "num_sentences": len(sentences),
            "dynamic_vocab": set(paper_vocab),
            "embeddings": embeddings,
            "sentences": sentences
        })

    # ── Stages 3-6: Concept discovery + gap detection per paper pair ──────────
    gaps: List[Dict] = []

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            if results[i]['domain'] == results[j]['domain']:
                continue

            embeddings_a = results[i]['embeddings']
            embeddings_b = results[j]['embeddings']
            sentences_a  = results[i]['sentences']
            sentences_b  = results[j]['sentences']
            domain_a     = results[i]['domain']
            domain_b     = results[j]['domain']

            distinctive_to_a = results[i]['dynamic_vocab'] - results[j]['dynamic_vocab']
            distinctive_to_b = results[j]['dynamic_vocab'] - results[i]['dynamic_vocab']

            concepts_a = discover_concepts(
                embeddings_a, sentences_a, domain=domain_a, extra_vocab=distinctive_to_a
            )
            concepts_b = discover_concepts(
                embeddings_b, sentences_b, domain=domain_b, extra_vocab=distinctive_to_b
            )

            if not concepts_a or not concepts_b:
                continue

            divergences = analyze_cross_domain_divergence(concepts_a, concepts_b)
            threshold   = adaptive_gap_threshold(concepts_a, concepts_b)

            if divergences:
                divergence_centroids = np.array([c_a.centroid for c_a, _, _ in divergences])
                divergence_sims_to_b = cosine_similarity(divergence_centroids, embeddings_b)
            else:
                divergence_sims_to_b = np.empty((0, embeddings_b.shape[0] if embeddings_b.ndim == 2 else 0), dtype=float)

            # Score all divergences, then take top 3
            scored = []
            for idx, (concept_a, concept_b, distance) in enumerate(divergences):
                similarity_to_b = divergence_sims_to_b[idx] if divergences else np.array([0.0])
                novelty  = validate_gap_novelty(
                    concept_a, concept_b, embeddings_a, embeddings_b,
                    similarity_to_b=similarity_to_b
                )
                strength = compute_gap_confidence(
                    concept_a, concept_b, distance,
                    embeddings_a, embeddings_b,
                    domain_a, domain_b, threshold, novelty,
                    similarity_to_b=similarity_to_b
                )
                scored.append((concept_a, concept_b, distance, novelty, strength))

            scored.sort(key=lambda x: x[4], reverse=True)

            for concept_a, concept_b, distance, novelty, strength in scored[:3]:
                # Pre-compute similarities once; pass to gap generator to avoid
                # redundant cosine_similarity inside extract_evidence_sentences
                centroid_a = np.array(concept_a.centroid).reshape(1, -1)
                centroid_b = np.array(concept_b.centroid).reshape(1, -1)
                sims_a = cosine_similarity(centroid_a, embeddings_a)[0]
                sims_b = cosine_similarity(centroid_b, embeddings_b)[0]

                gap = generate_explainable_gap(
                    concept_a, concept_b, distance, strength,
                    domain_a, domain_b,
                    sentences_a, sentences_b,
                    embeddings_a, embeddings_b,
                    novelty, threshold,
                    precomputed_sims_a=sims_a,
                    precomputed_sims_b=sims_b,
                )
                gaps.append(gap.dict())

    # ── Stage 7: Deduplication + limit ───────────────────────────────────────
    deduplicated_gaps = deduplicate_gaps_by_embedding(gaps)
    limited_gaps      = limit_gaps_per_comparison(deduplicated_gaps, max_gaps=10)

    serialised_papers = [
        {
            "id": r["id"],
            "domain": r["domain"],
            "num_sentences": r["num_sentences"],
            "num_concepts": 0,
            "dynamic_vocab": list(r["dynamic_vocab"]),
        }
        for r in results
    ]

    return {
        "papers": serialised_papers,
        "gaps": limited_gaps,
        "total_detected": len(gaps),
        "total_unique": len(deduplicated_gaps),
        "confidence_explanation": (
            "Gap strength is computed from: semantic distance (0.30), "
            "concept prominence (0.20), absence in target domain (0.20), "
            "distinctiveness (0.10), evidence ratio (0.10), gap magnitude (0.10), "
            "adjusted by pair compatibility and novelty score."
        )
    }


@app.post("/api/analyze/stream")
async def analyze_papers_stream(papers: List[Paper]):
    """
    Same pipeline as /api/analyze but streams progress via Server-Sent Events.
    Each event is a JSON line: {"type": "progress"|"result"|"error", ...}
    The frontend reads these incrementally so the UI updates in real time
    instead of freezing on one stage for the entire duration.
    """
    from fastapi.responses import StreamingResponse
    import json

    async def event_stream():
        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        try:
            if len(papers) < 2:
                yield sse({"type": "error", "message": "Need at least 2 papers"})
                return

            # ── Stage 1-2: Embeddings ─────────────────────────────────────────
            results = []
            total = len(papers)
            for idx, paper in enumerate(papers):
                yield sse({
                    "type": "progress",
                    "stage": "embedding",
                    "message": f"Generating embeddings for paper {idx+1}/{total}…",
                    "pct": int(10 + 30 * idx / total)
                })
                sentences = preprocess_academic_text(paper.text)

                if not sentences:
                    yield sse({"type": "error",
                               "message": f"No usable sentences in paper '{paper.id}'."})
                    return
                if len(sentences) < 10:
                    yield sse({"type": "error",
                               "message": f"Paper '{paper.id}' has only {len(sentences)} sentences — minimum is 10."})
                    return

                # Run blocking embedding in thread pool so SSE loop stays alive
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    _thread_pool,
                    lambda s=sentences, d=paper.domain: generate_semantic_embeddings(s, d)
                )
                paper_vocab = extract_domain_vocab_tfidf(sentences, top_k=50)
                results.append({
                    "id": paper.id,
                    "domain": paper.domain,
                    "num_sentences": len(sentences),
                    "dynamic_vocab": set(paper_vocab),
                    "embeddings": embeddings,
                    "sentences": sentences
                })

            # ── Stage 3: Concept discovery ────────────────────────────────────
            yield sse({"type": "progress", "stage": "clustering",
                       "message": "Discovering concepts via HDBSCAN…", "pct": 45})

            pair_results = []
            pairs = [(i, j) for i in range(len(results))
                            for j in range(i+1, len(results))
                            if results[i]['domain'] != results[j]['domain']]

            for pi, (i, j) in enumerate(pairs):
                ri, rj = results[i], results[j]
                yield sse({
                    "type": "progress",
                    "stage": "clustering",
                    "message": f"Clustering pair {pi+1}/{len(pairs)}: {ri['domain']} ↔ {rj['domain']}…",
                    "pct": int(45 + 20 * pi / max(len(pairs), 1))
                })

                loop = asyncio.get_event_loop()
                distinctive_a = ri['dynamic_vocab'] - rj['dynamic_vocab']
                distinctive_b = rj['dynamic_vocab'] - ri['dynamic_vocab']

                concepts_a = await loop.run_in_executor(
                    _thread_pool,
                    lambda: discover_concepts(ri['embeddings'], ri['sentences'],
                                              domain=ri['domain'], extra_vocab=distinctive_a)
                )
                concepts_b = await loop.run_in_executor(
                    _thread_pool,
                    lambda: discover_concepts(rj['embeddings'], rj['sentences'],
                                              domain=rj['domain'], extra_vocab=distinctive_b)
                )
                pair_results.append((ri, rj, concepts_a, concepts_b))

            # ── Stage 4: Divergence + scoring ────────────────────────────────
            yield sse({"type": "progress", "stage": "divergence",
                       "message": "Analyzing cross-domain semantic divergence…", "pct": 68})

            gaps: List[Dict] = []
            for pi, (ri, rj, concepts_a, concepts_b) in enumerate(pair_results):
                if not concepts_a or not concepts_b:
                    continue

                yield sse({
                    "type": "progress",
                    "stage": "divergence",
                    "message": f"Scoring gaps for pair {pi+1}/{len(pair_results)}…",
                    "pct": int(68 + 15 * pi / max(len(pair_results), 1))
                })

                loop = asyncio.get_event_loop()

                def _score_pair(ri=ri, rj=rj, ca=concepts_a, cb=concepts_b):
                    divergences = analyze_cross_domain_divergence(ca, cb)
                    threshold   = adaptive_gap_threshold(ca, cb)
                    if divergences:
                        divergence_centroids = np.array([c_a.centroid for c_a, _, _ in divergences])
                        divergence_sims_to_b = cosine_similarity(divergence_centroids, rj['embeddings'])
                    else:
                        divergence_sims_to_b = np.empty((0, rj['embeddings'].shape[0]), dtype=float)
                    scored = []
                    for idx, (concept_a, concept_b, distance) in enumerate(divergences):
                        similarity_to_b = divergence_sims_to_b[idx]
                        novelty  = validate_gap_novelty(
                            concept_a, concept_b,
                            ri['embeddings'], rj['embeddings'],
                            similarity_to_b=similarity_to_b
                        )
                        strength = compute_gap_confidence(
                            concept_a, concept_b, distance,
                            ri['embeddings'], rj['embeddings'],
                            ri['domain'], rj['domain'], threshold, novelty,
                            similarity_to_b=similarity_to_b
                        )
                        scored.append((concept_a, concept_b, distance, novelty, strength))
                    scored.sort(key=lambda x: x[4], reverse=True)

                    pair_gaps = []
                    for concept_a, concept_b, distance, novelty, strength in scored[:3]:
                        centroid_a = np.array(concept_a.centroid).reshape(1, -1)
                        centroid_b = np.array(concept_b.centroid).reshape(1, -1)
                        sims_a = cosine_similarity(centroid_a, ri['embeddings'])[0]
                        sims_b = cosine_similarity(centroid_b, rj['embeddings'])[0]
                        gap = generate_explainable_gap(
                            concept_a, concept_b, distance, strength,
                            ri['domain'], rj['domain'],
                            ri['sentences'], rj['sentences'],
                            ri['embeddings'], rj['embeddings'],
                            novelty, threshold,
                            precomputed_sims_a=sims_a,
                            precomputed_sims_b=sims_b,
                        )
                        pair_gaps.append(gap.dict())
                    return pair_gaps

                pair_gaps = await loop.run_in_executor(_thread_pool, _score_pair)
                gaps.extend(pair_gaps)

            # ── Stage 5: Deduplication ────────────────────────────────────────
            yield sse({"type": "progress", "stage": "dedup",
                       "message": "Deduplicating and ranking gaps…", "pct": 88})

            loop = asyncio.get_event_loop()
            deduplicated = await loop.run_in_executor(
                _thread_pool, lambda: deduplicate_gaps_by_embedding(gaps)
            )
            limited = limit_gaps_per_comparison(deduplicated, max_gaps=10)

            serialised_papers = [
                {"id": r["id"], "domain": r["domain"],
                 "num_sentences": r["num_sentences"], "num_concepts": 0,
                 "dynamic_vocab": list(r["dynamic_vocab"])}
                for r in results
            ]

            yield sse({"type": "progress", "stage": "complete",
                       "message": "Analysis complete!", "pct": 100})

            yield sse({
                "type": "result",
                "papers": serialised_papers,
                "gaps": limited,
                "total_detected": len(gaps),
                "total_unique": len(deduplicated),
                "confidence_explanation": (
                    "Gap strength is computed from: semantic distance (0.30), "
                    "concept prominence (0.20), absence in target domain (0.20), "
                    "distinctiveness (0.10), evidence ratio (0.10), gap magnitude (0.10), "
                    "adjusted by pair compatibility and novelty score."
                )
            })

        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/clear-cache")
async def clear_cache():
    with _cache_lock:
        EMBEDDING_CACHE.clear()
    return {"status": "cleared", "cache_size": len(EMBEDDING_CACHE)}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "device": DEVICE,
        "gpu_available": USE_GPU,
        "cache_size": len(EMBEDDING_CACHE),
        "cache_maxsize": EMBEDDING_CACHE.maxsize,
        "features": {
            "adaptive_threshold": True,
            "domain_vocabulary": True,
            "tfidf_phrase_extraction": True,
            "cross_paper_differential_vocab": True,
            "score_before_slice": True,
            "embedding_cache_lru_threadsafe": True,
            "embedding_deduplication": True,
            "novelty_validation": True,
            "compatibility_matrix": True,
            "uuid_ids": True,
            "extract_endpoint": True,
            # New perf flags
            "precompiled_regexes": True,
            "vectorised_divergence_filter": True,
            "precomputed_evidence_sims": True,
            "batch_size_128": True,
            "async_pdf_extraction": True,
        }
    }


@app.get("/api/domains")
async def get_supported_domains():
    return {
        "domains": list(DOMAIN_VOCABULARY.keys()),
        "common_academic": list(COMMON_ACADEMIC)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)