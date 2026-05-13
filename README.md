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


### When is this approach BEST?
- Exploring new cross-domain opportunities
- Initial literature review phase
- When you need explainable results
- When training data is unavailable



## Technical Stack

- **Backend:** FastAPI + Sentence-BERT + HDBSCAN + PyMuPDF
- **Frontend:** React + TypeScript + Tailwind CSS + Vite
- **ML Model:** Pre-trained `all-MiniLM-L6-v2` (384-dim embeddings)

## API Endpoints

- `POST /api/upload` - Upload a research paper PDF
- `POST /api/analyze` - Run complete AI analysis pipeline
- `GET /api/health` - Health check endpoint

