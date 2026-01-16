# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs
from verl.prompts import *


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


HOPS = [1, 2, 3, 4]
HOP_RATIO = [4, 3, 2, 1]
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")


def load_corpus(corpus_path: str):
    """Load corpus from JSONL file."""
    import datasets
    return datasets.load_dataset(
        'json',
        data_files=corpus_path,
        split="train",
        num_proc=4
    )


def load_corpus_chromadb(chroma_path: str, collection_name: str = "papers"):
    """
    Load corpus directly from ChromaDB as an iterable.
    Returns a dataset-like object that yields {"contents": ...} dicts.
    """
    import chromadb
    
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)
    total = collection.count()
    logger.info(f"Loading {total} documents from ChromaDB collection '{collection_name}'")
    
    class ChromaCorpus:
        def __init__(self, collection, total):
            self.collection = collection
            self.total = total
            self._data = None
        
        def _load_all(self):
            if self._data is None:
                results = self.collection.get(
                    include=["documents", "metadatas"],
                    limit=self.total,
                )
                self._data = []
                for i, doc in enumerate(results.get("documents", [])):
                    meta = results["metadatas"][i] if i < len(results.get("metadatas", [])) else {}
                    title = meta.get("title", "")
                    contents = f"{title}\n{doc}" if title else doc
                    self._data.append({"contents": contents})
            return self._data
        
        def shuffle(self, seed=42):
            import random
            data = self._load_all()
            random.seed(seed)
            random.shuffle(data)
            return data
        
        def __iter__(self):
            return iter(self._load_all())
        
        def __len__(self):
            return self.total
    
    return ChromaCorpus(collection, total)


def process_single_row(row, corpus_iter, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    # question = row.get("question", "")  # not used

    # Build prompt structure
    doc = next(corpus_iter)['contents']
    doc_title = doc.split("\n")[0]
    doc_text = doc.split("\n")[1]

    raw_document = f"(Title: {doc_title})\n{doc_text}\n"
    encoded_document = TOKENIZER.encode(raw_document)[:256]
    decoded_document = TOKENIZER.decode(encoded_document)

    n_hop = np.random.choice(HOPS, size=1, p=np.array(HOP_RATIO)/sum(HOP_RATIO))[0]    
    user_content = user_content_prefix.format(hops=n_hop, document=decoded_document)
    prompt = [{"role": "user", "content": user_content}]

    # Process data source
    data_source_tagged = f"search_zero_{n_hop}"
    reward_model = row.get("reward_model")
    reward_model['ground_truth']['target'] = None

    # Build tools kwargs structure
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": "", "question": "", "data_source": data_source_tagged
            }
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []
    
    # Load corpus from ChromaDB or JSONL
    if args.chroma_path:
        corpus = load_corpus_chromadb(args.chroma_path, args.collection_name)
    else:
        corpus = load_corpus(args.corpus_dir)
    corpus_iter = iter(corpus.shuffle(seed=42))

    # Download and process files using temporary directory
    with tempfile.TemporaryDirectory() as tmp_download_dir:
        for split in ["train"]:
            parquet_filename = f"{split}.parquet"
            logger.info(f"Processing {split} split...")

            # Download Parquet file from HuggingFace
            logger.info(f"Downloading {parquet_filename} from {args.hf_repo_id}")
            local_parquet_filepath = hf_hub_download(
                repo_id=args.hf_repo_id,
                filename=parquet_filename,
                repo_type="dataset",
                local_dir=tmp_download_dir,
                local_dir_use_symlinks=False,
            )

            # Load and process Parquet file
            df_raw = pd.read_parquet(local_parquet_filepath)
            logger.info(f"Loaded {len(df_raw)} rows from {parquet_filename}")

            def apply_process_row(row, split_name=split):
                return process_single_row(row, corpus_iter, current_split_name=split_name, row_index=row.name)

            df_processed = df_raw.apply(apply_process_row, axis=1)

            # Save processed DataFrame
            ratio_postfix = "ratio" + "".join(str(x) for x in HOP_RATIO)
            output_file_path = os.path.join(local_save_dir, f"zero_{ratio_postfix}.parquet")
            df_processed.to_parquet(output_file_path, index=False)
            logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
            processed_files.append(output_file_path)

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    print("Example prompt: ", df_processed.prompt[0])
    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Search-R1 from HuggingFace, process, and save to Parquet."
    )
    parser.add_argument(
        "--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train", help="HuggingFace dataset repository ID."
    )
    parser.add_argument(
        "--local_dir",
        default="./data",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")
    parser.add_argument("--corpus_dir", default="./corpus/wiki-18.jsonl", help="Path to Wiki corpus JSONL (if not using ChromaDB).")
    parser.add_argument("--chroma_path", default=None, help="Path to ChromaDB persistent storage (preferred over corpus_dir).")
    parser.add_argument("--collection_name", default="papers", help="ChromaDB collection name.")
    args = parser.parse_args()

    user_content_prefix = DEFAULT_CHALLENGER_PREFIX

    main()