import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\Knowledge chunks for RAG Training\faiss_rag")
INDEX_DIR = BASE_DIR / "faiss_index"

INDEX_PATH = INDEX_DIR / "knowledge.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("=" * 70)
print("FAISS RAG SEARCH")
print("=" * 70)

print("\nLoading FAISS index...")

index = faiss.read_index(str(INDEX_PATH))

print(f"Index loaded successfully.")
print(f"Vectors: {index.ntotal}")
print(f"Dimension: {index.d}")


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print(f"Metadata records: {len(metadata)}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")
print(f"Model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

print(f"Device: {device}")


# ============================================================
# MEAN POOLING
# ============================================================

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    return torch.sum(
        token_embeddings * input_mask_expanded,
        dim=1
    ) / torch.clamp(
        input_mask_expanded.sum(dim=1),
        min=1e-9
    )


# ============================================================
# EMBED QUERY
# ============================================================

def embed_query(query):

    encoded = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        model_output = model(**encoded)

    embedding = mean_pooling(
        model_output,
        encoded["attention_mask"]
    )

    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding.cpu().numpy().astype("float32")


# ============================================================
# SEARCH
# ============================================================

def search(query, top_k=TOP_K):

    query_embedding = embed_query(query)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    print("\n" + "=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    print(f"\nQuery: {query}")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0:
            continue

        result = metadata[idx]

        print("\n" + "-" * 70)

        print(f"Rank       : {rank}")
        print(f"Similarity : {score:.4f}")
        print(f"Source     : {result.get('source_file', 'Unknown')}")
        print(f"ID         : {result.get('id', 'Unknown')}")
        print(f"Section    : {result.get('section', 'Unknown')}")

        print("\nText:")
        print(result.get("text", ""))


# ============================================================
# INTERACTIVE SEARCH
# ============================================================

print("\n" + "=" * 70)
print("READY FOR SEARCH")
print("=" * 70)

print("\nType a question.")
print("Type 'exit' to quit.")

while True:

    query = input("\nQuery > ").strip()

    if query.lower() in {"exit", "quit"}:
        print("\nExiting...")
        break

    if not query:
        continue

    search(query)