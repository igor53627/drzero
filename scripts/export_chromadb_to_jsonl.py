#!/usr/bin/env python3
"""
Export ChromaDB collection to wiki-18.jsonl format for Dr. Zero training.

Usage:
    python scripts/export_chromadb_to_jsonl.py \
        --chroma_path /corpus/chromadb \
        --collection papers \
        --out /corpus/wiki-18.jsonl
"""
import argparse
import json
import chromadb


def main():
    parser = argparse.ArgumentParser(description="Export ChromaDB to JSONL")
    parser.add_argument("--chroma_path", required=True, help="Path to ChromaDB storage")
    parser.add_argument("--collection", default="papers", help="Collection name")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for fetching")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_collection(name=args.collection)
    total = collection.count()
    print(f"Exporting {total} documents from '{args.collection}'...")

    offset = 0
    exported = 0

    with open(args.out, "w") as f:
        while offset < total:
            results = collection.get(
                include=["documents", "metadatas"],
                limit=args.batch_size,
                offset=offset,
            )
            
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])

            for i, doc in enumerate(docs):
                metadata = metas[i] if i < len(metas) else {}
                title = metadata.get("title", "")
                
                if title:
                    contents = f"{title}\n{doc}"
                else:
                    contents = doc

                f.write(json.dumps({"contents": contents}) + "\n")
                exported += 1

            offset += args.batch_size
            print(f"  Exported {min(offset, total)}/{total}")

    print(f"Done. Wrote {exported} documents to {args.out}")


if __name__ == "__main__":
    main()
