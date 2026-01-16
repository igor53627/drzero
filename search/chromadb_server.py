# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
from typing import List, Optional

import chromadb
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 3
    return_scores: bool = False


app = FastAPI()
client = None
collection = None
embedder = None  # For custom embedding model


def get_query_embeddings(queries: List[str]) -> List[List[float]]:
    """Generate embeddings for queries using e5 format."""
    global embedder
    if embedder is None:
        return None  # Fall back to ChromaDB's default
    
    # e5 models use "query:" prefix for queries
    formatted = [f"query: {q}" for q in queries]
    embeddings = embedder.encode(formatted, normalize_embeddings=True)
    return embeddings.tolist()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    ChromaDB retrieval endpoint - compatible with Dr. Zero's search API.
    Supports custom embeddings for better accuracy.
    """
    # Try custom embeddings first
    query_embeddings = get_query_embeddings(request.queries)
    
    if query_embeddings is not None:
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=request.topk,
            include=["documents", "metadatas", "distances"]
        )
    else:
        results = collection.query(
            query_texts=request.queries,
            n_results=request.topk,
            include=["documents", "metadatas", "distances"]
        )
    
    resp = []
    for i, query in enumerate(request.queries):
        docs = results["documents"][i] if results["documents"] else []
        metadatas = results["metadatas"][i] if results["metadatas"] else []
        distances = results["distances"][i] if results["distances"] else []
        
        query_results = []
        for j, doc in enumerate(docs):
            metadata = metadatas[j] if j < len(metadatas) else {}
            title = metadata.get("title", "")
            
            result = {
                "title": title,
                "text": doc,
                "contents": f"{title}\n{doc}" if title else doc
            }
            
            if request.return_scores:
                score = 1.0 - distances[j] if j < len(distances) else 0.0
                query_results.append({"document": result, "score": score})
            else:
                query_results.append(result)
        
        resp.append(query_results)
    
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB retrieval server for Dr. Zero")
    parser.add_argument("--chroma_path", type=str, required=True, help="Path to ChromaDB persistent storage")
    parser.add_argument("--collection_name", type=str, default="documents", help="ChromaDB collection name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--embedding_model", type=str, default=None, 
                        help="Custom embedding model (e.g., intfloat/e5-large-v2)")
    
    args = parser.parse_args()
    
    # Load custom embedding model if specified
    if args.embedding_model:
        print(f"Loading embedding model: {args.embedding_model}")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(args.embedding_model)
        print(f"Embedding dimension: {embedder.get_sentence_embedding_dimension()}")
    
    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_collection(name=args.collection_name)
    
    print(f"Loaded ChromaDB collection '{args.collection_name}' with {collection.count()} documents")
    
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
