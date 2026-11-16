#!/usr/bin/env python3
"""
ARLIS Normalizer v8.1 - PURE NORMALIZATION ONLY
Read every record → normalize → write every record
No dedup. No grouping. No dropping.
"""
 
from __future__ import annotations
 
import argparse
import json
import lzma
import re
import sys
import time
from pathlib import Path
from typing import Iterator, Optional
 
# ==============================================================================
# CONSTANTS
# ==============================================================================
 
ARMENIAN_ABBREVIATIONS = {
    'ՀՀ', 'ԱՊՀ', 'ԵԱՏՄ', 'ԿԳՄՍ', 'ԱԺ', 'ՎՃ', 'ՀՕ',
    'ՄԱԿ', 'ԵՄ', 'ԱՄՆ', 'ՌԴ', 'ՎԿ', 'ԵՏ', 'ԱՏԳ',
    'ՀԾ', 'ՀՆ', 'ՎԱ', 'ՄԲ', 'ՍԴ', 'ԴԴ', 'ՔԿ',
    'ԿՎ', 'ՀԽ', 'ՄԽ', 'ՏԽ', 'ՎՏ', 'ԿԳ',
    'ԱՆ', 'ՊՆ', 'ԱՏՆ', 'ՔԱՆ', 'ՄՆ', 'ՖՆ',
    'ՀԱՊԿ', 'ԵԱՀԿ', 'ՆԱՏՕ', 'ԲՏՀ', 'ԳՄԿ',
    'ՏՀԶԿ', 'ՆԳՆ', 'ՎԻՆ', 'ՄՏՆ', 'ԷՆ', 'ԱՇՆ',
    'ՀԾԿՀ', 'ՖԻՄ', 'ՔԴ', 'ԶԻ', 'ՀՊ', 'ՄՀ',
    'ԼՂՀ', 'ՊՈԱԿ', 'ՍՊԸ', 'ՓԲԸ', 'ԲԲԸ', 'ՀՈԱԿ',
    'ՀԿ', 'ԿԲ', 'ԿԸՀ', 'ՊԵԿ', 'ԱԱԾ', 'ԱԻՆ',
}
 
ARM_UPPER = set('ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ')
ARM_LOWER = set('աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ')
ALL_ARM = ARM_UPPER | ARM_LOWER
 
SENTENCE_END = set('.:;!?…։՞՜')
 
# ==============================================================================
# HTML ENTITY MAP
# ==============================================================================
 
ENTITY_MAP = {}
for i, ch in enumerate('ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ'):
    ENTITY_MAP[1329 + i] = ch
for i, ch in enumerate('աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ'):
    ENTITY_MAP[1377 + i] = ch
ENTITY_MAP.update({
    1415: 'և', 171: '«', 187: '»', 160: ' ',
    13: '\n', 10: '\n', 9: ' ',
    8216: "'", 8217: "'", 8220: '"', 8221: '"',
    8211: '–', 8212: '—', 8226: '•', 8230: '…',
})
 
NAMED_ENTITIES = {
    'amp': '&', 'lt': '<', 'gt': '>', 'quot': '"', 'apos': "'",
    'nbsp': ' ', 'raquo': '»', 'laquo': '«',
    'mdash': '—', 'ndash': '–', 'hellip': '…',
    'bull': '•', 'copy': '©', 'reg': '®', 'trade': '™',
}
 
# ==============================================================================
# NORMALIZER
# ==============================================================================
 
class ArmenianNormalizer:
    def __init__(self, debug=False):
        self.debug = debug
        self.stats = {
            'entities_decoded': 0,
            'ligatures_restored': 0,
            'letter_spacing_fixed': 0,
            'words_lowercased': 0,
            'words_capitalized': 0,
        }
 
    def decode_entities(self, text: str) -> str:
        if not text:
            return text
 
        def dec_named(m):
            n = m.group(1).lower()
            if n in NAMED_ENTITIES:
                return NAMED_ENTITIES[n]
            return m.group(0)
        text = re.sub(r'&([a-zA-Z]+);', dec_named, text)
 
        def dec_dec(m):
            try:
                c = int(m.group(1))
                if c in ENTITY_MAP:
                    self.stats['entities_decoded'] += 1
                    return ENTITY_MAP[c]
                if 0x530 <= c <= 0x58F:
                    self.stats['entities_decoded'] += 1
                    return chr(c)
                return m.group(0)
            except Exception:
                return m.group(0)
        text = re.sub(r'&#([0-9]+);', dec_dec, text)
 
        def dec_hex(m):
            try:
                c = int(m.group(1), 16)
                if c in ENTITY_MAP:
                    self.stats['entities_decoded'] += 1
                    return ENTITY_MAP[c]
                if 0x530 <= c <= 0x58F:
                    self.stats['entities_decoded'] += 1
                    return chr(c)
                return m.group(0)
            except Exception:
                return m.group(0)
        text = re.sub(r'&#[xX]([0-9A-Fa-f]+);', dec_hex, text)
 
        return text.replace('\r\n', '\n').replace('\r', '\n')
 
    def strip_html(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r'<\s*br\s*/?\s*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</\s*(p|div|li|tr|h[1-6]|blockquote)\s*>', '\n',
                      text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        return text
 
    def fix_ligatures(self, text: str) -> str:
        before = text
        text = text.replace('ե վ', 'և').replace('Ե Վ', 'ԵՎ')
        text = text.replace('ե ւ', 'և').replace('Ե ւ', 'ԵՒ')
        text = text.replace('եւ', 'և').replace('ԵՒ', 'ԵՎ')
        text = text.replace('ո ւ', 'ու').replace('Ո ւ', 'Ու').replace('Ո Ւ', 'ՈՒ')
        if text != before:
            self.stats['ligatures_restored'] += 1
        return text
 
    def fix_letter_spacing(self, text: str) -> str:
        if not text:
            return text
        text = text.replace('ու', '§U§').replace('Ու', '§U2§').replace('ՈՒ', '§U3§')
        text = text.replace('և', '§E§').replace('ԵՎ', '§E2§')
        markers = ['§U§', '§U2§', '§U3§', '§E§', '§E2§']
 
        lines = text.split('\n')
        out_lines = []
        for line in lines:
            if not line.strip():
                out_lines.append(line)
                continue
            words = line.split()
            out = []
            i = 0
            while i < len(words):
                w = words[i]
                if w in markers:
                    out.append(w)
                    i += 1
                    continue
                if len(w) == 1 and w in ALL_ARM:
                    seq = [w]
                    j = i + 1
                    while j < len(words):
                        nw = words[j]
                        if len(nw) == 1 and nw in ALL_ARM:
                            seq.append(nw); j += 1
                        elif nw in markers:
                            seq.append(nw); j += 1
                        else:
                            break
                    if len(seq) >= 3:
                        out.append(''.join(seq))
                        self.stats['letter_spacing_fixed'] += 1
                        i = j
                        continue
                out.append(w)
                i += 1
            out_lines.append(' '.join(out))
        text = '\n'.join(out_lines)
        text = text.replace('§U§', 'ու').replace('§U2§', 'Ու').replace('§U3§', 'ՈՒ')
        text = text.replace('§E§', 'և').replace('§E2§', 'ԵՎ')
        return text
 
    def _lowercase_word(self, word: str) -> str:
        stripped = word.strip('«»()"\'.,:;!?-—–[]{}…։՞՜')
        if stripped in ARMENIAN_ABBREVIATIONS:
            return word
        if stripped.upper() in ARMENIAN_ABBREVIATIONS:
            return word
        out = []
        for ch in word:
            if 'Ա' <= ch <= 'Ֆ':
                out.append(chr(ord(ch) + 48))
            else:
                out.append(ch)
        return ''.join(out)
 
    def _capitalize_first_armenian(self, s: str) -> str:
        for i, ch in enumerate(s):
            if ch in ARM_LOWER:
                self.stats['words_capitalized'] += 1
                return s[:i] + ch.upper() + s[i + 1:]
            if ch in ARM_UPPER:
                return s
        return s
 
    def lowercase_armenian(self, text: str) -> str:
        if not text:
            return text
 
        lines = text.split('\n')
        result_lines = []
 
        for line in lines:
            if not line.strip():
                result_lines.append(line)
                continue
 
            tokens = line.split()
            result_tokens = []
            capitalize_next = True
 
            for token in tokens:
                lowered = self._lowercase_word(token)
                if capitalize_next:
                    lowered = self._capitalize_first_armenian(lowered)
                result_tokens.append(lowered)
                self.stats['words_lowercased'] += 1
 
                stripped = token.rstrip('»"\'”’)]}')
                if stripped and stripped[-1] in SENTENCE_END:
                    capitalize_next = True
                else:
                    capitalize_next = False
 
            result_lines.append(' '.join(result_tokens))
 
        return '\n'.join(result_lines)
 
    def clean_whitespace(self, text: str) -> str:
        lines = [re.sub(r'[ \t]+', ' ', ln).strip() for ln in text.split('\n')]
        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
 
    def normalize(self, text: str, context: str = 'body') -> str:
        if not text:
            return ""
        text = self.decode_entities(text)
        text = self.strip_html(text)
        text = self.fix_ligatures(text)
        text = self.fix_letter_spacing(text)
        text = self.lowercase_armenian(text)
        text = self.clean_whitespace(text)
        return text
 
 
# ==============================================================================
# FILE ITERATION
# ==============================================================================
 
def iter_jsonl(path: Path) -> Iterator[dict]:
    if path.suffix == '.xz':
        opener = lambda p: lzma.open(p, 'rt', encoding='utf-8', errors='replace')
    else:
        opener = lambda p: open(p, 'r', encoding='utf-8', errors='replace')
    with opener(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
 
 
# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
 
def process_full(input_path: Path, output_path: Path, debug: bool = False, limit: Optional[int] = None):
    print("=" * 80)
    print("ARLIS NORMALIZER v8.1 - PURE NORMALIZATION (no dedup, no grouping)")
    print("=" * 80)
    print(f"📖 Input:  {input_path}")
    print(f"📝 Output: {output_path}")
    if limit:
        print(f"🔢 Limit:  {limit}")
    print("=" * 80)
 
    normalizer = ArmenianNormalizer(debug=debug)
    t0 = time.time()
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    n_read = 0
    n_written = 0
    skipped_lang = 0
    skipped_empty = 0
 
    with open(output_path, 'w', encoding='utf-8') as out:
        for record in iter_jsonl(input_path):
            n_read += 1
 
            # Skip non-Armenian
            if record.get('language') and record.get('language') != 'AM':
                skipped_lang += 1
                continue
 
            body = record.get('body') or ''
            if not body:
                skipped_empty += 1
                continue
 
            normalized_text = normalizer.normalize(body)
            if not normalized_text:
                skipped_empty += 1
                continue
 
            title = normalizer.normalize(record.get('title') or '')[:500]
 
            parent_doc = record.get('parent_doc')
            parent_id = None
            if isinstance(parent_doc, dict):
                parent_id = str(parent_doc.get('id') or '') or None
 
            changed = record.get('changed_docs')
            n_amend = len(changed) if isinstance(changed, list) else 0
 
            row = {
                'uniqid': str(record.get('uniqid', '')),
                'parent_id': parent_id,
                'doc_type': record.get('DocType') or '',
                'n_amend': n_amend,
                'act_type': record.get('ActType') or '',
                'status': record.get('ActStatus') or '',
                'organ': record.get('EnactmentOrgan') or '',
                'enactment_date': record.get('EnactmentDate') or '',
                'act_title_hy': title,
                'text_hy': normalized_text,
                'text_length': len(normalized_text),
            }
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            n_written += 1
 
            if n_read % 5000 == 0:
                elapsed = time.time() - t0
                print(f"   {n_read:,} read / {n_written:,} written ({elapsed:.0f}s)")
 
            if limit and n_read >= limit:
                break
 
    elapsed = time.time() - t0
 
    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"⏱️  Time:              {elapsed:.1f}s")
    print(f"📖 Records read:       {n_read:,}")
    print(f"⏭️  Skipped (language): {skipped_lang:,}")
    print(f"⏭️  Skipped (empty):    {skipped_empty:,}")
    print(f"✅ Written:            {n_written:,}")
    print(f"📁 Output:             {output_path}")
    if output_path.exists():
        print(f"📦 Size:               {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 80)
 
    if debug:
        print("\n📊 Normalizer Stats:")
        for k, v in normalizer.stats.items():
            print(f"   {k}: {v:,}")
 
 
# ==============================================================================
# CLI
# ==============================================================================
 
def main():
    ap = argparse.ArgumentParser(
        description='ARLIS normalizer v8.1 - pure normalization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/01_normalize_arlis.py --limit 1000 --debug
  python scripts/01_normalize_arlis.py
        """
    )
    ap.add_argument('--input', default='data/raw/arlis_docs.jsonl.xz')
    ap.add_argument('--output', default='data/normalized/normalized_documents.jsonl')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--debug', action='store_true')
 
    args = ap.parse_args()
 
    input_path = Path(args.input)
    output_path = Path(args.output)
 
    if not input_path.exists():
        print(f"❌ Not found: {input_path}")
        sys.exit(1)
 
    process_full(input_path, output_path, args.debug, args.limit)
 
 
if __name__ == '__main__':
    main()
