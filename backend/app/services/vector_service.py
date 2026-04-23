import os
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Absolute persistent path for Chroma
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../chroma_db"))

# Persistent Chroma client
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="clinical_records")


def create_embedding(text: str):
    return model.encode(text).tolist()


def store_record_embedding(record_id: int, clinical_note: str, extracted_entities: dict):
    embedding = create_embedding(clinical_note)

    diagnoses = ", ".join(extracted_entities.get("diagnoses", []))
    symptoms = ", ".join(extracted_entities.get("symptoms", []))
    medications = ", ".join(extracted_entities.get("medications", []))

    metadata = {
        "diagnoses": diagnoses,
        "symptoms": symptoms,
        "medications": medications
    }

    # Prevent duplicate-id errors after DB resets / reruns
    try:
        collection.delete(ids=[str(record_id)])
    except Exception:
        pass

    collection.add(
        documents=[clinical_note],
        embeddings=[embedding],
        ids=[str(record_id)],
        metadatas=[metadata]
    )


def search_similar_records(query: str, top_k=3):
    query_embedding = create_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    formatted_results = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

        formatted_results.append({
            "record_id": ids[i],
            "clinical_note": documents[i] if i < len(documents) else "",
            "diagnoses": metadata.get("diagnoses", ""),
            "symptoms": metadata.get("symptoms", ""),
            "medications": metadata.get("medications", ""),
            "distance": distances[i] if i < len(distances) else None
        })

    return {
        "query": query,
        "matches": formatted_results
    }