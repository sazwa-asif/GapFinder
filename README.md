# AI Research Gap Finder

An explainable cross-domain research gap finder that uses transformer-based semantic embeddings to identify research gaps between different academic domains.

## What is GapFinder?

**GapFinder** identifies research opportunities by comparing papers from different domains. For example:
- Upload medical research papers in "Domain A"
- Upload AI/ML papers in "Domain B"  
- GapFinder discovers concepts discussed in medicine but missing in AI research (and vice versa)
- This helps researchers find opportunities for cross-disciplinary collaboration

## How to Run

### Start Backend (Terminal 1)
```powershell
powershell -Command "Set-Location 'e:/CODING PROJECTS/GapFinder/backend'; python -m uvicorn main:app --host 0.0.0.0 --port 8000"
```
- Runs at: http://localhost:8000

### Start Frontend (Terminal 2)
```powershell
powershell -Command "Set-Location 'e:/CODING PROJECTS/GapFinder/frontend'; npx vite"
```
- Runs at: http://localhost:3000

**Access the app:** Open http://localhost:3000 in your browser

## AI Components Explained

### 1. Sentence-BERT Embeddings (Semantic Understanding)
- Converts paper sentences into 384-dimensional vectors
- Uses pre-trained `all-MiniLM-L6-v2` model
- Similar concepts have similar vectors (cosine similarity)

### 2. HDBSCAN Clustering (Concept Discovery)
- Groups semantically similar sentences into clusters
- Each cluster = one research concept
- Automatic - no need to specify number of clusters

### 3. Cross-Domain Divergence Analysis
- Measures semantic distance between concepts from different domains
- High distance (> 0.5) = research gap
- Identifies what's present in one field but missing in another

### 4. Confidence Scoring
Combines multiple factors:
- **40%** - Semantic distance between concepts
- **30%** - Prominence of concept in source domain
- **20%** - Absence in target domain
- **10%** - Distinctiveness of the gap

## Output Strength & Quality

### How Strong is the Output?

The output quality depends on several factors:

| Factor | Impact on Quality |
|--------|------------------|
| Paper quantity (5+ per domain) | **High** - More data = better clustering |
| Paper relevance | **High** - Similar topic areas produce better results |
| Text length per paper | **Medium** - Longer papers provide more concepts |
| Domain similarity | **Medium** - Very different domains show more gaps |

**Expected Output Quality:**
- **Good** with 5+ relevant papers per domain
- **Moderate** with 2-4 papers (fewer concepts detected)
- **Basic** with 1 paper per domain (limited comparison)

### Does it Work on Trained Models?

**Yes!** This system uses **transfer learning**:

1. **Sentence-BERT (all-MiniLM-L6-v2)**
   - Pre-trained on 1 billion+ sentence pairs
   - No training required for this application
   - Downloads automatically from HuggingFace

2. **HDBSCAN Clustering**
   - Unsupervised algorithm - no training data needed
   - Works directly on embedding vectors

**What does NOT require training:**
- Semantic embedding generation (pre-trained model)
- Concept clustering (unsupervised)
- Gap detection (distance-based threshold)
- Confidence scoring (formula-based)

### Is This a Better Approach?

**Advantages of this approach:**
| Benefit | Explanation |
|---------|-------------|
| **No Training Required** | Uses pre-trained transformers - ready to use |
| **Explainable AI** | Shows WHY a gap was detected (distance, prominence, evidence) |
| **Domain Agnostic** | Works on any academic field |
| **Scalable** | Add more papers, get more concepts |
| **Transparent** | Confidence scores + evidence sentences |

**Limitations compared to custom models:**
| Limitation | Impact |
|------------|--------|
| General-purpose embeddings | May miss domain-specific terminology |
| Fixed distance threshold | 0.5 may not be optimal for all domains |
| No fine-tuning | Cannot specialize for specific fields |

### Comparison to Alternatives

| Method | Training Required | Explainability | Best For |
|--------|-------------------|----------------|----------|
| **GapFinder (this)** | None | High | Quick analysis, general domains |
| Custom BERT fine-tuning | High GPU + data | Medium | Specialized fields |
| LLM-based analysis (GPT) | None (API) | Medium | Qualitative insights |
| Keyword matching | None | Low | Surface-level analysis |

### When is this approach BEST?
- Exploring new cross-domain opportunities
- Initial literature review phase
- When you need explainable results
- When training data is unavailable

### When might other approaches be BETTER?
- Highly specialized medical/legal fields → Fine-tuned models
- Need quantitative precision → Custom trained classifiers
- Unlimited budget/resources → LLM API-based analysis

## How to Fix Limitations

### 1. Domain-Specific Terminology Issue

**Problem:** General embeddings miss specialized terms (e.g., "tachycardia" in medicine)

**Solutions:**

| Approach | Difficulty | Effectiveness |
|----------|------------|---------------|
| **Add domain vocabulary** | Easy | Medium |
| **Fine-tune Sentence-BERT** | Hard | High |
| **Use domain-specific embeddings** | Medium | High |
| **Combine with keyword boosting** | Easy | Medium |

**Implementation:**
```python
# Add custom vocabulary to keyword extraction
DOMAIN_VOCABULARY = {
    "medical": ["tachycardia", "hypertension", "myocardial", "glycemic", ...],
    "legal": ["jurisprudence", "litigation", "liability", ...],
    "cs": ["neural_network", "backpropagation", "transformer", ...]
}

# Boost scores for domain-specific terms
def compute_domain_adjusted_score(embedding, domain):
    # Check if domain vocabulary terms appear in nearby sentences
    # Add bonus to confidence scores
```

### 2. Fixed Threshold (0.5) Issue

**Problem:** 0.5 may not be optimal for all domain pairs

**Solutions:**

| Approach | Description |
|----------|-------------|
| **Adaptive threshold** | Calculate threshold based on average concept distance |
| **Per-domain calibration** | Store optimized thresholds per domain pair |
| **Confidence intervals** | Show range instead of binary gap/no-gap |

**Implementation:**
```python
def adaptive_gap_threshold(concepts_a, concepts_b):
    # Calculate average intra-domain distance
    avg_intra_a = compute_avg_cluster_distance(concepts_a)
    avg_intra_b = compute_avg_cluster_distance(concepts_b)
    
    # Set threshold relative to intra-domain variance
    return max(0.3, min(0.7, (avg_intra_a + avg_intra_b) / 2))
```

### 3. Lack of Fine-Tuning

**Problem:** Cannot specialize for specific fields

**Solutions:**

| Solution | When to Use |
|----------|-------------|
| **Domain-adaptive pre-training** | Have 10K+ domain sentences |
| **Contrastive learning** | Have labeled gap examples |
| **Hybrid with LLM** | Budget allows API calls |

**Fine-tuning approach:**
```python
from sentence_transformers import SentenceTransformer, InputExample

# Create domain-specific training data
train_examples = [
    InputExample(texts=["Deep learning optimizes neural networks", 
                        "Gradient descent minimizes loss function"],
                 label=0.8),  # Similar concepts
    InputExample(texts=["CRISPR edits DNA sequences",
                        "Neural networks recognize images"],
                 label=0.2),  # Different concepts
]

# Fine-tune the model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.fit(train_examples, epochs=1)
```

### 4. Performance Optimizations

| Optimization | Speedup | Difficulty |
|--------------|---------|------------|
| **Batch processing** | 5-10x | Easy |
| **GPU acceleration** | 10-50x | Medium |
| **Caching embeddings** | 100x | Easy |
| **Parallel API requests** | 3-5x | Medium |

### 5. Enhanced Confidence Scoring

**Current:** 4-factor weighted average

**Enhanced version:**
```python
def enhanced_confidence_score(
    semantic_distance,
    prominence_a,
    prominence_b,
    absence_score,
    distinctiveness,
    evidence_count,
    source_quality,  # New: Journal impact factor, citations
    temporal_factor  # New: Is concept growing or declining?
):
    weights = {
        'distance': 0.25,
        'prominence_a': 0.20,
        'prominence_b': 0.10,
        'absence': 0.15,
        'distinctiveness': 0.10,
        'evidence': 0.10,
        'quality': 0.05,
        'temporal': 0.05
    }
    # Calculate weighted score
```

### 6. Recommended Enhancements by Use Case

| Use Case | Recommended Fix |
|----------|-----------------|
| Medical research | Fine-tune on PubMed abstracts |
| Legal analysis | Add legal vocabulary dictionary |
| Real-time analysis | Add embedding caching |
| High precision needs | Add human-in-the-loop validation |
| Cross-domain exploration | Keep current approach |

### 7. Future Improvements Roadmap

| Priority | Improvement | Effort |
|----------|-------------|--------|
| High | Adaptive threshold | Low |
| Medium | Domain vocabulary injection | Low |
| Medium | GPU acceleration support | Medium |
| Low | Fine-tuning pipeline | High |
| Low | LLM hybrid integration | Medium |

## Technical Stack

- **Backend:** FastAPI + Sentence-BERT + HDBSCAN + PyMuPDF
- **Frontend:** React + TypeScript + Tailwind CSS + Vite
- **ML Model:** Pre-trained `all-MiniLM-L6-v2` (384-dim embeddings)

## API Endpoints

- `POST /api/upload` - Upload a research paper PDF
- `POST /api/analyze` - Run complete AI analysis pipeline
- `GET /api/health` - Health check endpoint

