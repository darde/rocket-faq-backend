# Rocket Mortgage FAQ Bot — Backend

Python + FastAPI backend for the Rocket Mortgage FAQ chatbot. Implements a RAG (Retrieval-Augmented Generation) pipeline with Pinecone vector search, HuggingFace embeddings, and an OpenRouter-powered LLM. Includes a built-in evaluation framework with retrieval metrics and LLM-as-judge scoring.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                            │
│   /api/chat/   /api/eval/retrieval   /api/eval/judge           │
│                /api/eval/full        /health                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     RAG Pipeline                                │
│  1. Embed query    (sentence-transformers / all-MiniLM-L6-v2)  │
│  2. Vector search  (Pinecone — cosine similarity, top-k)       │
│  3. Build prompt   (system prompt + retrieved context)          │
│  4. Generate       (OpenRouter LLM — Gemini Flash)             │
└──────┬──────────────────┬──────────────────┬────────────────────┘
       │                  │                  │
┌──────▼──────┐  ┌───────▼───────┐  ┌───────▼──────────────┐
│  Embeddings │  │   Pinecone    │  │    OpenRouter LLM    │
│ HuggingFace │  │  Vector DB    │  │  (gemini-2.0-flash)  │
│ MiniLM-L6   │  │  384-dim,     │  │                      │
│ (384-dim)   │  │  cosine       │  │                      │
└─────────────┘  └───────────────┘  └──────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Module                           │
│  • Precision@k, Recall@k, MRR  (retrieval quality)             │
│  • LLM-as-Judge                (generation quality)             │
│    → relevance, correctness, completeness, faithfulness         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Observability                               │
│  structlog — structured, timestamped logs for every             │
│  search, LLM call, embedding, and evaluation event             │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
rocket-faq-backend/
├── source.md                    # FAQ knowledge base (Rocket Mortgage)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── scripts/
│   └── ingest.py                # Chunks source.md → embeddings → Pinecone
└── app/
    ├── main.py                  # FastAPI app, CORS, route registration
    ├── config.py                # Pydantic settings from .env
    ├── api/
    │   ├── chat.py              # POST /api/chat/
    │   └── evaluation.py        # POST /api/eval/{retrieval,judge,full}
    ├── core/
    │   ├── chunking.py          # FAQ-aware document chunker
    │   ├── embeddings.py        # Embedding provider abstraction (local / API)
    │   ├── vectorstore.py       # Pinecone index + search operations
    │   ├── llm.py               # OpenRouter client (OpenAI-compatible)
    │   └── rag.py               # Retrieve → augment → generate pipeline
    ├── evaluation/
    │   ├── metrics.py           # Precision@k, Recall@k, MRR
    │   └── judge.py             # LLM-as-judge evaluation
    └── observability/
        └── logger.py            # Structured logging (structlog)
```

---

## Prerequisites

- Python 3.11+
- [Pinecone](https://www.pinecone.io) account and API key
- [OpenRouter](https://openrouter.ai) account and API key

---

## Getting Started

**1. Clone the repo**

```bash
git clone git@github.com:darde/rocket-faq-backend.git
cd rocket-faq-backend
```

**2. Create a virtual environment and install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` with your API keys. See [Environment Variables](#environment-variables) for all options.

**4. Ingest FAQ data into Pinecone**

This reads `source.md`, chunks it into Q&A pairs, generates embeddings, creates the Pinecone index, and uploads all vectors. You only need to run this once.

```bash
python scripts/ingest.py
```

**5. Start the API server**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is available at **http://localhost:8000**. Interactive docs at http://localhost:8000/docs.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/chat/` | Ask a question — returns answer + source citations |
| `POST` | `/api/eval/retrieval` | Retrieval metrics (Precision, Recall, MRR) on the test dataset |
| `POST` | `/api/eval/judge` | LLM-as-judge evaluation on a single question |
| `POST` | `/api/eval/full` | Full evaluation (retrieval metrics + LLM-as-judge) |

### Example

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I set up autopay?"}'
```

---

## Evaluation

### Retrieval Metrics

Measured against a 10-query test dataset with known relevant chunks:

- **Precision@k** — Of the k retrieved documents, how many are relevant?
- **Recall@k** — Of all relevant documents, how many appear in the top k?
- **MRR** — Average of 1/rank of the first relevant document

### LLM-as-Judge

Uses the LLM to score generated answers on four dimensions (1–5 each):

- **Relevance** — Does the answer address the question?
- **Correctness** — Is it factually accurate based on the source context?
- **Completeness** — Does it fully cover the question?
- **Faithfulness** — Does it stay grounded in the retrieved context?

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PINECONE_API_KEY` | — | Pinecone API key (required) |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required) |
| `PINECONE_INDEX_NAME` | `rocket-mortgage-faq` | Pinecone index name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `api` (HF Inference API) |
| `LLM_MODEL` | `google/gemini-2.0-flash-001` | LLM model via OpenRouter |
| `FRONTEND_URL` | `http://localhost:5173` | Allowed CORS origin for the frontend |
| `TOP_K` | `5` | Number of documents to retrieve per query |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Related

- **Frontend**: [rocket-faq-frontend](https://github.com/darde/rocket-faq-frontend)
