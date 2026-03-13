# GraphRAG: Neo4j + RAG Chatbot

This repository hosts a small GraphRAG playground built on top of Neo4j and LangChain, with:

- A **History of India** chat experience (GraphRAG over a Neo4j graph and RAG context)
- A local **Streamlit** UI
- A deployed **Vercel** web UI backed by a Python serverless function

The actual graph/RAG logic lives under `codebase/knowledge-graph-rag`, while this top-level
project wires it into UIs and deployment.

## Project layout

```text
.
├── api/
│   └── chat.py                    # Vercel Python serverless function (GraphRAG backend)
├── index.html                     # Minimal web chat UI (History of India)
├── vercel.json                    # Vercel routing configuration
├── requirements.txt               # Backend dependencies for Vercel/serverless
└── codebase/
    └── knowledge-graph-rag/       # Upstream Knowledge-Graph-RAG project
        ├── kgraph_rag/
        │   ├── roman_emp_graph_rag.py
        │   └── roman_emp_graph_rag chat_ui.py  # Streamlit UI
        ├── healthcare/
        ├── prep_text_for_rag/
        ├── simple_kg/
        └── README.md              # Original project documentation
```

For details of the underlying graph/RAG examples (healthcare, simple KG, etc.), see
`codebase/knowledge-graph-rag/README.md`.

## Running locally (Streamlit UI)

1. Create and activate a virtual environment if you have not already:

```bash
cd "D:\AI Projects\Neo4j"
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
```

2. Install dependencies (either the upstream project or the minimal root deps):

```bash
pip install -r codebase/knowledge-graph-rag/requirements.txt
# or, for just the backend pieces:
pip install -r requirements.txt
```

3. Configure environment variables in a `.env` file at the repo root:

```text
NEO4J_URI=bolt+s://<your-neo4j-uri>
NEO4J_USERNAME=<username>
NEO4J_PASSWORD=<password>
AURA_INSTANCENAME=<optional-aura-instance-name>
OPENAI_API_KEY=<your-openai-api-key>
```

4. Run the Streamlit UI:

```bash
streamlit run "codebase/knowledge-graph-rag/kgraph_rag/roman_emp_graph_rag chat_ui.py"
```

This starts a local chat UI that talks to the same GraphRAG pipeline.

## Vercel deployment

The hosted UI at `https://graph-rag-beryl.vercel.app/` uses:

- `index.html` for the frontend chat experience
- `api/chat.py` as a Python serverless function that:
  - Connects to Neo4j
  - Runs the hybrid GraphRAG retrieval (structured + vector search)
  - Calls the LLM to answer the user’s question
  - Returns suggested follow‑up questions

### Environment variables on Vercel

In the Vercel project settings, configure the same variables used locally:

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `AURA_INSTANCENAME` (if needed)
- `OPENAI_API_KEY`

No build command is required; Vercel serves `index.html` as static content and
auto-detects the Python runtime for `api/chat.py`.

## How the GraphRAG pipeline works (high level)

- **Entity extraction**: an LLM extracts entities from the user question.
- **Structured retrieval**: Neo4j full‑text search finds matching entity nodes
  and explores their neighborhoods.
- **Unstructured retrieval**: a `Neo4jVector` index runs hybrid vector search
  over `Document` nodes.
- **Fusion**: structured and unstructured context are combined into a single
  context string.
- **Question condensation**: for follow‑up questions, chat history is condensed
  into a standalone question.
- **Answer generation**: an OpenAI chat model answers using only the fused context.

The same core logic is shared between the Streamlit UI and the Vercel API.

