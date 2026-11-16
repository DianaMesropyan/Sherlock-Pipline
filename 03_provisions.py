#!/usr/bin/env python3
"""
Step 3: Provisions.
Split each act's text_hy by 'Հոդված N'.
  - No Հոդված → whole act = 1 provision, article_label="0", part=0, n_parts=0
  - Article ≤ 1500 chars → 1 provision, part=0, n_parts=0
  - Article > 1500 chars → split at paragraph/sentence breaks, part=1..N
All text preserved. Metadata + title carried on every row.
 
Input:  data/normalized/hastat1_documents.jsonl
Output: data/normalized/hastat2_provisions.jsonl
"""
 
from __future__ import annotations
 
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, List, Dict
 
MAX = 1500   # split threshold
MIN = 250    # merge threshold (article smaller than this merges with previous)
 
# Armenian 'Հոդված' followed by a number, at the start of a line (allow leading spaces)
ARTICLE_RE = re.compile(r'(?m)^[ \t]*Հոդված\s+(\d+(?:[.\-]\d+)*)\b')
 
# Sentence split (Armenian '։' and ':' and '.' and '!' and '?')
SENT_SPLIT = re.compile(r'(?<=[։:.!?])\s+')
 
 
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
 
 
def split_article(text: str, max_len: int = MAX) -> List[str]:
    """Split an over-long article into <=max_len pieces at paragraph, then sentence, then hard cut."""
    # 1. paragraphs
    paras = text.split('\n\n')
    pieces: List[str] = []
    buf = ''
    for p in paras:
        # paragraph too long → split by sentence
        if len(p) > max_len:
            if buf.strip():
                pieces.append(buf.strip()); buf = ''
            sents = SENT_SPLIT.split(p)
            sbuf = ''
            for s in sents:
                if len(sbuf) + len(s) + 1 > max_len and sbuf:
                    pieces.append(sbuf.strip()); sbuf = ''
                # single sentence too long → hard cut
                while len(s) > max_len:
                    pieces.append(s[:max_len].strip())
                    s = s[max_len:]
                sbuf += s + ' '
            if sbuf.strip():
                pieces.append(sbuf.strip())
        else:
            if len(buf) + len(p) + 2 > max_len and buf:
                pieces.append(buf.strip()); buf = ''
            buf += p + '\n\n'
    if buf.strip():
        pieces.append(buf.strip())
    return pieces or [text]
 
 
def split_into_articles(text: str) -> List[Dict]:
    """
    Returns list of {'label': 'Հոդված N' or '0', 'text': '...'}.
    Preamble (before first Հոդված) is prepended to the first article's text.
    If no Հոդված found → single item with label '0'.
    """
    matches = list(ARTICLE_RE.finditer(text))
 
    if not matches:
        return [{'label': '0', 'text': text.strip()}]
 
    articles = []
    # preamble (text before first Հոդված)
    preamble = text[:matches[0].start()].strip()
 
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        label = f"Հոդված {m.group(1)}"
        articles.append({'label': label, 'text': chunk})
 
    # merge preamble into first article
    if preamble:
        articles[0]['text'] = preamble + '\n\n' + articles[0]['text']
 
    return articles
 
 
# ==============================================================================
# MAIN
# ==============================================================================
 
def process(input_path: Path, output_path: Path):
    print("=" * 80)
    print("STEP 3: PROVISIONS")
    print("=" * 80)
    print(f"📖 Input:  {input_path}")
    print(f"📝 Output: {output_path}")
    print(f"⚙️  MAX={MAX} chars, MIN={MIN} chars")
    print("=" * 80)
 
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    n_acts = 0
    n_provisions = 0
    n_whole_acts = 0   # no Հոդված
    n_split_articles = 0
 
    with open(output_path, 'w', encoding='utf-8') as out:
        for act in iter_jsonl(input_path):
            n_acts += 1
            act_id = str(act.get('act_id') or act.get('uniqid') or '')
            text = act.get('text_hy') or ''
            if not text.strip():
                continue
 
            articles = split_into_articles(text)
 
            if len(articles) == 1 and articles[0]['label'] == '0':
                n_whole_acts += 1
 
            for art in articles:
                body = art['text']
                label = art['label']
 
                # split if too long
                if len(body) > MAX:
                    parts = split_article(body, MAX)
                    n_split_articles += 1
                    for pi, p in enumerate(parts, 1):
                        row = {
                            'provision_id': f"{act_id}_{label.replace(' ', '_')}_{pi}",
                            'act_id': act_id,
                            'act_uniqid': act_id,
                            'act_title_hy': act.get('act_title_hy', ''),
                            'act_type': act.get('act_type', ''),
                            'organ': act.get('organ', ''),
                            'enactment_date': act.get('enactment_date', ''),
                            'status': act.get('status', ''),
                            'article_label': label,
                            'part': pi,
                            'n_parts': len(parts),
                            'text_hy': p,
                            'text_length': len(p),
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + '\n')
                        n_provisions += 1
                else:
                    row = {
                        'provision_id': f"{act_id}_{label.replace(' ', '_')}_0",
                        'act_id': act_id,
                        'act_uniqid': act_id,
                        'act_title_hy': act.get('act_title_hy', ''),
                        'act_type': act.get('act_type', ''),
                        'organ': act.get('organ', ''),
                        'enactment_date': act.get('enactment_date', ''),
                        'status': act.get('status', ''),
                        'article_label': label,
                        'part': 0,
                        'n_parts': 0,
                        'text_hy': body,
                        'text_length': len(body),
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    n_provisions += 1
 
            if n_acts % 5000 == 0:
                print(f"   {n_acts:,} acts → {n_provisions:,} provisions ({time.time()-t0:.0f}s)")
 
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"⏱️  Time:                  {elapsed:.1f}s")
    print(f"📖 Acts read:              {n_acts:,}")
    print(f"✅ Provisions written:      {n_provisions:,}")
    print(f"📦 Acts with no Հոդված:      {n_whole_acts:,}")
    print(f"✂️  Articles split (>1500):  {n_split_articles:,}")
    print(f"📁 Output:                 {output_path}")
    if output_path.exists():
        print(f"📦 Size:                   {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 80)
 
 
# ==============================================================================
# CLI
# ==============================================================================
 
def main():
    ap = argparse.ArgumentParser(description='Step 3: provisions')
    ap.add_argument('--input',  default='data/normalized/hastat1_documents.jsonl')
    ap.add_argument('--output', default='data/normalized/hastat2_provisions.jsonl')
    args = ap.parse_args()
 
    input_path = Path(args.input)
    output_path = Path(args.output)
 
    if not input_path.exists():
        print(f"❌ Not found: {input_path}")
        sys.exit(1)
 
    process(input_path, output_path)
 
 
if __name__ == '__main__':
    main()
