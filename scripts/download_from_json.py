#!/usr/bin/env python3
"""
Download papers from extract-data JSON file and organize by topic.

Usage:
    python scripts/download_from_json.py \
        --json extract-data-2026-01-16.json \
        --output-dir /path/to/pdfs \
        --by-topic  # Optional: organize into subdirectories by topic
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm


def clean_filename(title: str, max_len: int = 80) -> str:
    """Clean title for use as filename."""
    # Remove special characters
    clean = re.sub(r'[^\w\s-]', '', title)
    # Replace spaces with underscores
    clean = re.sub(r'\s+', '_', clean)
    return clean[:max_len]


def extract_paper_id(url: str) -> str:
    """Extract paper ID from ePrint URL."""
    # Handle both formats:
    # https://eprint.iacr.org/2024/1477
    # https://eprint.iacr.org/2024/1477.pdf
    match = re.search(r'/(\d{4})/(\d+)', url)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None


def download_paper(url: str, output_path: Path, session: requests.Session) -> bool:
    """Download a single paper."""
    if output_path.exists():
        return True
    
    # Normalize URL to PDF
    if not url.endswith('.pdf'):
        url = url.rstrip('/') + '.pdf'
    
    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        
        if 'application/pdf' in resp.headers.get('content-type', ''):
            with open(output_path, 'wb') as f:
                f.write(resp.content)
            return True
        else:
            print(f"  Not a PDF: {url}")
            return False
            
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download papers from JSON metadata")
    parser.add_argument("--json", required=True, help="Path to JSON file with paper metadata")
    parser.add_argument("--output-dir", required=True, help="Output directory for PDFs")
    parser.add_argument("--by-topic", action="store_true", help="Organize by topic subdirectories")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between downloads (seconds)")
    parser.add_argument("--topics", nargs="+", help="Only download specific topics")
    
    args = parser.parse_args()
    
    # Load JSON
    with open(args.json) as f:
        data = json.load(f)
    
    papers = data.get("papers", [])
    print(f"Loaded {len(papers)} papers from JSON")
    
    # Filter by topics if specified
    if args.topics:
        papers = [p for p in papers if p.get("topic") in args.topics]
        print(f"Filtered to {len(papers)} papers for topics: {args.topics}")
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Research Bot - Academic Paper Download)"
    })
    
    # Track by topic
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    by_topic = {}
    
    for paper in tqdm(papers, desc="Downloading"):
        title = paper.get("title", "Unknown")
        url = paper.get("title_citation", "")
        topic = paper.get("topic", "Unknown")
        
        if not url:
            stats["skipped"] += 1
            continue
        
        # Determine output path
        paper_id = extract_paper_id(url)
        if not paper_id:
            stats["skipped"] += 1
            continue
        
        filename = f"{paper_id}_{clean_filename(title)}.pdf"
        
        if args.by_topic:
            topic_dir = output_dir / clean_filename(topic, max_len=50)
            topic_dir.mkdir(exist_ok=True)
            output_path = topic_dir / filename
        else:
            output_path = output_dir / filename
        
        # Download
        if download_paper(url, output_path, session):
            stats["downloaded"] += 1
            if topic not in by_topic:
                by_topic[topic] = 0
            by_topic[topic] += 1
        else:
            stats["failed"] += 1
        
        time.sleep(args.delay)
    
    # Summary
    print(f"\n=== Download Summary ===")
    print(f"Downloaded: {stats['downloaded']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    
    print(f"\n=== By Topic ===")
    for topic, count in sorted(by_topic.items(), key=lambda x: -x[1]):
        print(f"  [{count:3d}] {topic}")
    
    print(f"\nPapers saved to: {output_dir}")


if __name__ == "__main__":
    main()
