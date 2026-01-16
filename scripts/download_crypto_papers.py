#!/usr/bin/env python3
"""
Download cryptography papers from IACR ePrint for corpus expansion.

Categories to expand iO corpus:
- Functional Encryption
- Fully Homomorphic Encryption (FHE)
- Witness Encryption
- Multilinear Maps
- Lattice Cryptography / LWE
- Program Obfuscation

Usage:
    python scripts/download_crypto_papers.py --output-dir /path/to/pdfs --category fhe --limit 200
    python scripts/download_crypto_papers.py --output-dir /path/to/pdfs --all --limit 100

Requirements:
    pip install requests beautifulsoup4 tqdm
"""

import argparse
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# IACR ePrint search queries for each category
CATEGORIES = {
    "fhe": {
        "name": "Fully Homomorphic Encryption",
        "queries": [
            "fully homomorphic encryption",
            "FHE bootstrapping",
            "CKKS encryption",
            "BGV encryption",
            "TFHE",
        ],
        "priority": 1,
        "target": 300,
    },
    "functional_encryption": {
        "name": "Functional Encryption",
        "queries": [
            "functional encryption",
            "attribute-based encryption",
            "predicate encryption",
            "inner product encryption",
        ],
        "priority": 1,
        "target": 200,
    },
    "witness_encryption": {
        "name": "Witness Encryption",
        "queries": [
            "witness encryption",
            "extractable witness encryption",
        ],
        "priority": 1,
        "target": 50,
    },
    "multilinear_maps": {
        "name": "Multilinear Maps",
        "queries": [
            "multilinear maps",
            "graded encoding",
            "GGH multilinear",
            "CLT multilinear",
        ],
        "priority": 1,
        "target": 100,
    },
    "lattice": {
        "name": "Lattice Cryptography",
        "queries": [
            "learning with errors",
            "LWE cryptography",
            "lattice-based cryptography",
            "RLWE",
            "ring-LWE",
        ],
        "priority": 2,
        "target": 300,
    },
    "obfuscation": {
        "name": "Program Obfuscation",
        "queries": [
            "program obfuscation",
            "virtual black-box obfuscation",
            "differing inputs obfuscation",
            "best-possible obfuscation",
        ],
        "priority": 1,
        "target": 100,
    },
    "zk": {
        "name": "Zero Knowledge",
        "queries": [
            "zero knowledge proof",
            "SNARK",
            "STARK",
            "zkSNARK",
            "succinct argument",
        ],
        "priority": 3,
        "target": 200,
    },
    "mpc": {
        "name": "Secure Computation",
        "queries": [
            "secure multi-party computation",
            "garbled circuits",
            "secret sharing",
            "oblivious transfer",
        ],
        "priority": 3,
        "target": 200,
    },
}


class EPrintDownloader:
    """Download papers from IACR ePrint."""
    
    BASE_URL = "https://eprint.iacr.org"
    SEARCH_URL = "https://eprint.iacr.org/search"
    
    def __init__(self, output_dir: str, delay: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Research Bot - Corpus Building)"
        })
        self.downloaded = set()
        self._load_existing()
    
    def _load_existing(self):
        """Load already downloaded paper IDs."""
        for pdf in self.output_dir.glob("*.pdf"):
            # Extract paper ID from filename (e.g., "2024_123_Title.pdf" -> "2024/123")
            match = re.match(r"(\d{4})_(\d+)_", pdf.name)
            if match:
                self.downloaded.add(f"{match.group(1)}/{match.group(2)}")
    
    def search(self, query: str, limit: int = 100) -> list[dict]:
        """Search ePrint for papers matching query."""
        results = []
        
        try:
            # ePrint search API
            params = {
                "q": query,
                "s": "relevance",  # Sort by relevance
            }
            
            resp = self.session.get(self.SEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Find paper entries
            for entry in soup.select(".paperentry, .paper-entry, article"):
                try:
                    # Extract paper ID and title
                    link = entry.find("a", href=re.compile(r"/\d{4}/\d+"))
                    if not link:
                        continue
                    
                    href = link.get("href", "")
                    match = re.search(r"/(\d{4})/(\d+)", href)
                    if not match:
                        continue
                    
                    paper_id = f"{match.group(1)}/{match.group(2)}"
                    title = link.get_text(strip=True)
                    
                    if paper_id not in self.downloaded:
                        results.append({
                            "id": paper_id,
                            "title": title,
                            "year": match.group(1),
                            "number": match.group(2),
                        })
                    
                    if len(results) >= limit:
                        break
                        
                except Exception:
                    continue
            
        except Exception as e:
            print(f"Search error for '{query}': {e}")
        
        return results
    
    def download_paper(self, paper: dict) -> bool:
        """Download a single paper PDF."""
        paper_id = paper["id"]
        year, number = paper_id.split("/")
        
        # Clean title for filename
        title = re.sub(r'[^\w\s-]', '', paper["title"])
        title = re.sub(r'\s+', '_', title)[:80]
        
        filename = f"{year}_{number}_{title}.pdf"
        filepath = self.output_dir / filename
        
        if filepath.exists():
            return True
        
        try:
            pdf_url = f"{self.BASE_URL}/{paper_id}.pdf"
            resp = self.session.get(pdf_url, timeout=60)
            resp.raise_for_status()
            
            if resp.headers.get("content-type", "").startswith("application/pdf"):
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                self.downloaded.add(paper_id)
                return True
            else:
                print(f"Not a PDF: {paper_id}")
                return False
                
        except Exception as e:
            print(f"Download error for {paper_id}: {e}")
            return False
    
    def download_category(self, category: str, limit: int = None) -> int:
        """Download papers for a category."""
        if category not in CATEGORIES:
            print(f"Unknown category: {category}")
            return 0
        
        cat_info = CATEGORIES[category]
        limit = limit or cat_info["target"]
        
        print(f"\n=== {cat_info['name']} ===")
        print(f"Target: {limit} papers")
        
        all_papers = []
        for query in cat_info["queries"]:
            print(f"Searching: {query}")
            papers = self.search(query, limit=limit)
            all_papers.extend(papers)
            time.sleep(self.delay)
        
        # Deduplicate
        seen = set()
        unique_papers = []
        for p in all_papers:
            if p["id"] not in seen:
                seen.add(p["id"])
                unique_papers.append(p)
        
        print(f"Found {len(unique_papers)} new papers")
        
        # Download
        downloaded = 0
        for paper in tqdm(unique_papers[:limit], desc="Downloading"):
            if self.download_paper(paper):
                downloaded += 1
            time.sleep(self.delay)
        
        print(f"Downloaded: {downloaded}")
        return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download crypto papers from IACR ePrint")
    parser.add_argument("--output-dir", required=True, help="Output directory for PDFs")
    parser.add_argument("--category", choices=list(CATEGORIES.keys()), help="Category to download")
    parser.add_argument("--all", action="store_true", help="Download all categories")
    parser.add_argument("--priority", type=int, choices=[1, 2, 3], help="Only categories with this priority")
    parser.add_argument("--limit", type=int, help="Limit papers per category")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--list", action="store_true", help="List available categories")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n=== Available Categories ===\n")
        for key, info in sorted(CATEGORIES.items(), key=lambda x: x[1]["priority"]):
            print(f"[P{info['priority']}] {key}: {info['name']} (target: {info['target']})")
        print("\nPriority 1 = Core (directly related to iO)")
        print("Priority 2 = Foundational")
        print("Priority 3 = Extended")
        return
    
    downloader = EPrintDownloader(args.output_dir, delay=args.delay)
    
    if args.all:
        categories = list(CATEGORIES.keys())
        if args.priority:
            categories = [k for k, v in CATEGORIES.items() if v["priority"] <= args.priority]
    elif args.category:
        categories = [args.category]
    else:
        print("Specify --category, --all, or --list")
        return
    
    total = 0
    for category in categories:
        count = downloader.download_category(category, limit=args.limit)
        total += count
    
    print(f"\n=== Total Downloaded: {total} papers ===")


if __name__ == "__main__":
    main()
