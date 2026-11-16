import argparse
import json
import lzma
from pathlib import Path
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/arlis_docs.jsonl.xz")
    args = parser.parse_args()
 
    dump_path = Path(args.input)
    if not dump_path.exists():
        print(f"Error: {dump_path} does not exist.")
        return
 
    print(f"File found: {dump_path}")
    print(f"File size: {dump_path.stat().st_size / (1024*1024):.2f} MB")
 
    total = 0
    with lzma.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                total += 1
                if total % 50000 == 0:
                    print(f"Read {total:,} records...")
 
    print(f"Validation successful: Total {total:,} documents found.")
 
 
if __name__ == "__main__":
    main()
