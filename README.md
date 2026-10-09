# FAISS-vector-database-for-RAG-training
A FAISS-based vector database for RAG training, indexing 311 knowledge chunks from 13 JSONL knowledge bases using MiniLM embeddings and enabling semantic similarity search with metadata retrieval.
# FAISS Vector Database for RAG Training

A lightweight, local vector database implementation for Retrieval-Augmented Generation (RAG) using **FAISS**, **PyTorch**, and **Hugging Face Transformers**.

This project combines multiple JSONL knowledge bases into a single searchable vector index. It converts knowledge chunks into dense embeddings, stores them in FAISS, and retrieves relevant information through semantic similarity search.

## Overview

The project demonstrates the preparation of a knowledge retrieval layer for RAG-based applications and AI agents.

The current implementation loads 13 JSONL knowledge-base files, processes 311 knowledge chunks, generates 384-dimensional embeddings, and stores them in a FAISS index. A search script retrieves the most relevant chunks for a user's query while preserving their original metadata.

## Key Features

- **Multi-source ingestion:** Loads knowledge chunks from multiple JSONL files.
- **Text embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` through Hugging Face Transformers and PyTorch.
- **Vector indexing:** Stores embeddings in a FAISS `IndexFlatIP` index.
- **Semantic search:** Retrieves relevant knowledge chunks using query embeddings.
- **Metadata preservation:** Retains chunk IDs, sections, text, and source filenames.
- **Local execution:** Runs the embedding and retrieval pipeline locally on CPU.
- **Reusable foundation:** Can serve as a retrieval component for future RAG pipelines and AI agents.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| Vector database | FAISS CPU |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Machine learning framework | PyTorch |
| Model and tokenizer loading | Hugging Face Transformers |
| Input format | JSONL |
| Metadata storage | JSON |

## Dataset Summary

| Metric | Value |
|---|---:|
| Knowledge-base files | 13 |
| Knowledge chunks loaded | 311 |
| Embedding dimension | 384 |
| Indexed vectors | 311 |
| FAISS index type | `IndexFlatIP` |
| Similarity method | Cosine similarity using normalized vectors |
| Execution device | CPU |

The source knowledge bases cover technologies and frameworks such as Agno, Astro, Azure Data Factory, Databricks, Hermes, Ollama, OpenClaw, OpenFang, PySpark, pytest, swarms-RS, and Terraform.

## Architecture

```text
13 JSONL Knowledge Bases
          |
          v
   Load and Validate
   Knowledge Chunks
          |
          v
  Extract Text and Metadata
          |
          v
  MiniLM Embedding Model
          |
          v
  384-Dimensional Vectors
          |
          v
   FAISS IndexFlatIP
          |
          v
    knowledge.index
          |
          v
      User Query
          |
          v
   Query Embedding
          |
          v
   Top-K Similarity Search
          |
          v
 Retrieved Knowledge Chunks
```

## Repository Structure

```text
FAISS-vector-database-for-RAG-training/
├── build_index.py
├── search.py
├── faiss_index/
│   ├── knowledge.index
│   ├── metadata.json
│   └── config.json
├── requirements.txt
└── README.md
```

The original JSONL knowledge-base files can be maintained in the repository or supplied through a local dataset directory, depending on the intended data-sharing policy.

## Input Data Format

Each JSONL file contains one JSON object per line. Each valid knowledge chunk must include the following fields:

```json
{
  "id": "astro_architecture_001",
  "section": "Astro Architecture",
  "text": "Description of the relevant concept or implementation."
}
```

During ingestion, the pipeline adds a `source_file` field to each chunk so retrieved results can be traced back to their original knowledge base.

## Setup

### 1. Prerequisites

- Python 3.11
- pip
- Internet access for the initial embedding-model download
- Sufficient disk space for Python dependencies, model files, and the generated index

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe --version
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If the environment is not activated, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Building the FAISS Index

Configure `build_index.py` to point to the directory containing the JSONL knowledge-base files.

Run:

```powershell
python build_index.py
```

Or, when using the virtual environment directly:

```powershell
.\.venv\Scripts\python.exe build_index.py
```

The script performs the following operations:

1. Discovers JSONL files in the configured dataset directory.
2. Loads and validates knowledge chunks.
3. Preserves source metadata.
4. Loads the embedding model.
5. Generates normalized embeddings in batches.
6. Creates a FAISS index and adds the vectors.
7. Saves the index, metadata, and configuration files.

## Running Semantic Search

After building the index, run:

```powershell
python search.py
```

Or:

```powershell
.\.venv\Scripts\python.exe search.py
```

Enter a natural-language question when prompted. The search pipeline embeds the query, searches the FAISS index, and displays the top five results with similarity scores and metadata.

Example queries:

```text
What is the architecture of an Agno agent?
How does Ollama manage local AI models?
How does Astro handle client-side JavaScript?
How does Terraform manage infrastructure?
How does PySpark process large datasets?
```

The results include:

- Rank
- Similarity score
- Source filename
- Knowledge chunk ID
- Section
- Retrieved text

## Generated Artifacts

| File | Purpose |
|---|---|
| `knowledge.index` | Stores the FAISS vector index |
| `metadata.json` | Stores knowledge chunks and associated metadata |
| `config.json` | Records the embedding model, vector dimension, vector count, similarity metric, and source filenames |

Keep the metadata ordering consistent with the vector ordering. The search implementation uses the FAISS result index to retrieve the corresponding metadata record.

## Similarity Search

The implementation uses `IndexFlatIP`, which performs exact inner-product search.

Because both document embeddings and query embeddings are L2-normalized, their inner product corresponds to cosine similarity. The search returns the top-k vectors with the highest similarity scores.

Similarity scores indicate relative semantic alignment; they are not calibrated probabilities or guarantees that a result fully answers the query.

## Validation

The initial implementation was tested successfully with the following results:

- All 13 JSONL files were discovered and processed.
- 311 valid knowledge chunks were loaded.
- 311 embeddings of dimension 384 were generated.
- All 311 vectors were added to the FAISS index.
- The index and metadata were saved successfully.
- A semantic search query about Agno agent architecture returned relevant Agno knowledge chunks.

## Current Scope and Limitations

This repository currently implements **knowledge ingestion, embedding generation, vector indexing, and semantic retrieval**.

It does not yet implement a complete LLM-backed RAG response pipeline. The current search script returns relevant knowledge chunks rather than generating an answer from a language model.

Potential future improvements include:

- Integrating an LLM to generate context-grounded answers.
- Adding configurable top-k retrieval and similarity thresholds.
- Improving retrieval with reranking or hybrid keyword and vector search.
- Exposing retrieval through a FastAPI service.
- Integrating the retriever with an Agno agent or another RAG framework.
- Adding automated ingestion validation and retrieval tests.
- Supporting incremental index updates.

## Intended Use

This implementation can be used as a starting point for:

- RAG pipeline development and experimentation.
- Semantic search across technical documentation.
- Knowledge retrieval for AI agents.
- Local prototyping of vector search workflows.
- Exploring embedding models and vector indexing.

## Objective

The objective is to establish a reusable vector retrieval layer that transforms structured knowledge chunks into searchable embeddings and makes relevant context available to downstream RAG applications and AI agents.
