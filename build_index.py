import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(r"D:\Knowledge chunks for RAG Training")

OUTPUT_DIR = DATASET_DIR / "faiss_rag" / "faiss_index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND JSONL FILES
# ============================================================

jsonl_files = sorted(DATASET_DIR.glob("*.jsonl"))

print("=" * 70)
print("FAISS KNOWLEDGE BASE BUILDER")
print("=" * 70)

print(f"\nDataset directory:")
print(DATASET_DIR)

print(f"\nJSONL files found: {len(jsonl_files)}")

for file in jsonl_files:
    print(f"  - {file.name}")

if not jsonl_files:
    raise RuntimeError(
        "No JSONL files found in the dataset directory."
    )


# ============================================================
# LOAD ALL KNOWLEDGE CHUNKS
# ============================================================

all_chunks = []

print("\n" + "=" * 70)
print("LOADING KNOWLEDGE CHUNKS")
print("=" * 70)

for file_path in jsonl_files:

    file_count = 0

    print(f"\nLoading: {file_path.name}")

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                chunk = json.loads(line)

            except json.JSONDecodeError as error:

                print(
                    f"WARNING: Invalid JSON in "
                    f"{file_path.name}, line {line_number}"
                )

                print(error)
                continue

            # Required fields
            if not all(
                field in chunk
                for field in ["id", "section", "text"]
            ):
                print(
                    f"WARNING: Missing required fields in "
                    f"{file_path.name}, line {line_number}"
                )
                continue

            # Store source dataset
            chunk["source_file"] = file_path.name

            all_chunks.append(chunk)
            file_count += 1

    print(f"  Loaded chunks: {file_count}")


print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"\nTotal JSONL files : {len(jsonl_files)}")
print(f"Total chunks      : {len(all_chunks)}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

print("Embedding model loaded successfully.")


# ============================================================
# SELECT DEVICE
# ============================================================

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
# GENERATE EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING EMBEDDINGS")
print("=" * 70)

texts = [
    chunk["text"]
    for chunk in all_chunks
]

all_embeddings = []

total = len(texts)

for start in range(0, total, BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        total
    )

    batch_texts = texts[start:end]

    print(
        f"Embedding {start + 1}-{end} "
        f"of {total}"
    )

    encoded = tokenizer(
        batch_texts,
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

    embeddings = mean_pooling(
        model_output,
        encoded["attention_mask"]
    )

    # Normalize for cosine similarity
    embeddings = torch.nn.functional.normalize(
        embeddings,
        p=2,
        dim=1
    )

    all_embeddings.append(
        embeddings.cpu().numpy()
    )


# ============================================================
# COMBINE EMBEDDINGS
# ============================================================

embeddings = np.vstack(
    all_embeddings
).astype("float32")

print("\nEmbedding shape:")
print(embeddings.shape)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

print("\n" + "=" * 70)
print("CREATING FAISS INDEX")
print("=" * 70)

print(f"\nVector dimension: {dimension}")

# Inner product on normalized vectors
# = cosine similarity

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"Vectors stored: {index.ntotal}")


# ============================================================
# SAVE FAISS INDEX
# ============================================================

index_path = OUTPUT_DIR / "knowledge.index"

faiss.write_index(
    index,
    str(index_path)
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = OUTPUT_DIR / "metadata.json"

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "embedding_model": MODEL_NAME,
    "vector_dimension": dimension,
    "total_vectors": index.ntotal,
    "distance_metric": "cosine_similarity",
    "source_files": [
        file.name
        for file in jsonl_files
    ]
}

config_path = OUTPUT_DIR / "config.json"

with open(
    config_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config,
        file,
        indent=2
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FAISS INDEX BUILD COMPLETE")
print("=" * 70)

print(f"\nFAISS index:")
print(index_path)

print(f"\nMetadata:")
print(metadata_path)

print(f"\nConfiguration:")
print(config_path)

print(f"\nTotal vectors: {index.ntotal}")

print(f"Vector dimension: {dimension}")

print("\nDone!")