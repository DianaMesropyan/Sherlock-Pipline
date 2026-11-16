#!/usr/bin/env python3
"""
Step 2: Incorporation handling.
Group docs into acts by parent_id. Pick one canonical version per act.
Drop the rest.
 
Input:  normalized_documents.jsonl
Output: canonical_documents.jsonl  (one row per act)
        dropped_documents.jsonl    (the losers, for audit)
"""
 
from __future__ import annotations
 
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Dict, List, Tuple
from collections import defaultdict
 
 
# ==============================================================================
# HELPERS
# ==============================================================================
 
def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
 
 
def parse_date_key(d: str) -> str:
    """DD.MM.YYYY or YYYY-MM-DD → YYYYMMDD. Bad → '00000000'."""
    if not d:
        return '00000000'
    d = d.strip()
    # DD.MM.YYYY
    parts = d.split('.')
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        dd, mm, yyyy = parts
        return f'{yyyy}{mm.zfill(2)}{dd.zfill(2)}'
    # YYYY-MM-DD
    parts = d.split('-')
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        yyyy, mm, dd = parts
        return f'{yyyy}{mm.zfill(2)}{dd.zfill(2)}'
    return '00000000'
 
 
def version_score(doc: dict) -> Tuple[str, int, int]:
    """
    Higher = better. Order:
      1. newest enactment_date
      2. biggest text_length
      3. highest numeric uniqid
    """
    date_key = parse_date_key(doc.get('enactment_date', ''))
    text_len = int(doc.get('text_length') or 0)
    try:
        uid_num = int(doc.get('uniqid') or 0)
    except Exception:
        uid_num = 0
    return (date_key, text_len, uid_num)
 
 
def act_key(doc: dict) -> str:
    """
    If parent_id exists → group under it.
    Else → own uniqid (it's a base act).
    """
    pid = doc.get('parent_id')
    if pid:
        return str(pid)
    return str(doc.get('uniqid', ''))
 
 
# ==============================================================================
# MAIN
# ==============================================================================
 
def process(input_path: Path, output_path: Path, dropped_path: Optional[Path] = None):
    print("=" * 80)
    print("STEP 2: INCORPORATION HANDLING")
    print("=" * 80)
    print(f"📖 Input:   {input_path}")
    print(f"📝 Output:  {output_path}")
    if dropped_path:
        print(f"🗑️  Dropped: {dropped_path}")
    print("=" * 80)
 
    t0 = time.time()
 
    # ---------- PHASE 1: read + group by act_key ----------
    print("\n📖 PHASE 1: Grouping docs by act...")
    groups: Dict[str, List[dict]] = defaultdict(list)
    n_read = 0
 
    for doc in iter_jsonl(input_path):
        n_read += 1
        groups[act_key(doc)].append(doc)
        if n_read % 20000 == 0:
            print(f"   {n_read:,} read / {len(groups):,} acts ({time.time()-t0:.0f}s)")
 
    print(f"\n   ✅ Docs read:   {n_read:,}")
    print(f"   ✅ Acts:        {len(groups):,}")
 
    # ---------- PHASE 2: pick canonical ----------
    print("\n📊 PHASE 2: Picking canonical version per act...")
    kept: List[dict] = []
    dropped: List[dict] = []
    multi = 0
 
    for key, docs in groups.items():
        if len(docs) == 1:
            kept.append(docs[0])
            continue
        multi += 1
        winner = max(docs, key=version_score)
        kept.append(winner)
        for d in docs:
            if d is winner:
                continue
            d2 = dict(d)
            d2['_drop_reason'] = 'superseded_by'
            d2['_kept_uniqid'] = winner.get('uniqid')
            d2['_kept_date'] = winner.get('enactment_date')
            d2['_kept_len'] = winner.get('text_length')
            dropped.append(d2)
 
    print(f"   Acts with 2+ versions: {multi:,}")
    print(f"   Kept:                  {len(kept):,}")
    print(f"   Dropped:               {len(dropped):,}")
    reduction = (1 - len(kept) / max(n_read, 1)) * 100
    print(f"   Reduction:             {reduction:.1f}%")
 
    # ---------- PHASE 3: write ----------
    print("\n📄 PHASE 3: Writing outputs...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out:
        for d in kept:
            out.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f"   ✅ Kept:    {len(kept):,} → {output_path}")
 
    if dropped_path:
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dropped_path, 'w', encoding='utf-8') as out:
            for d in dropped:
                out.write(json.dumps(d, ensure_ascii=False) + '\n')
        print(f"   🗑️  Dropped: {len(dropped):,} → {dropped_path}")
 
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"⏱️  Time:      {elapsed:.1f}s")
    print(f"📖 Read:      {n_read:,}")
    print(f"✅ Kept:      {len(kept):,}")
    print(f"🗑️  Dropped:   {len(dropped):,}")
    print(f"📉 Reduction: {reduction:.1f}%")
    print("=" * 80)
 
 
# ==============================================================================
# CLI
# ==============================================================================
 
def main():
    ap = argparse.ArgumentParser(description='Step 2: incorporation handling')
    ap.add_argument('--input',  default='data/normalized/normalized_documents.jsonl')
    ap.add_argument('--output', default='data/normalized/canonical_documents.jsonl')
    ap.add_argument('--dropped', default='data/normalized/dropped_documents.jsonl')
    args = ap.parse_args()
 
    input_path = Path(args.input)
    output_path = Path(args.output)
    dropped_path = Path(args.dropped) if args.dropped else None
 
    if not input_path.exists():
        print(f"❌ Not found: {input_path}")
        sys.exit(1)
 
    process(input_path, output_path, dropped_path)
 
 
if __name__ == '__main__':
    main()
