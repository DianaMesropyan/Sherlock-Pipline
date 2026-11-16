#!/usr/bin/env python3
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sentence_transformers import SentenceTransformer

INDEX = Path('/kaggle/working/index')
PROVISIONS = Path('/kaggle/input/datasets/ineedyoutowork/armenian-legal-provisions/hastat2_provisions.jsonl')
MODEL = 'BAAI/bge-m3'
MAX_LEN = 512

STOPWORDS = {
    'և', 'ու', 'կամ', 'որ', 'ինչ', 'ինչպես', 'որտեղ', 'երբ', 'ով', 'ում',
    'այս', 'այն', 'այդ', 'սույն', 'մի', 'մը', 'ը', 'ն', 'ի', 'ից', 'ով',
    'ում', 'ին', 'ներ', 'ները', 'ըստ', 'համար', 'մասին', 'վրա', 'հետ',
    'դեպի', 'առանց', 'միջև', 'կողմից', 'այլ', 'նաև', 'միայն', 'եթե',
}

VERB_ENDINGS = [
    'ացնում', 'ացնել', 'ացվել', 'ացվում',
    'վում', 'վել',
    'ում', 'ել', 'ալ', 'իլ', 'ուլ',
    'եցի', 'եցիր', 'եցին', 'եց', 'աց',
    'ելու', 'ելիք',
]

LEGAL_PATTERNS = [
    (re.compile(r'ստեղծման կարգ', re.I), 4),
    (re.compile(r'վարման կարգ', re.I), 5),
    (re.compile(r'պահպանման կարգ', re.I), 4),
    (re.compile(r'գրանցման կարգ', re.I), 4),
    (re.compile(r'հաշվառման կարգ', re.I), 4),
    (re.compile(r'կառավարման կարգ', re.I), 4),
    (re.compile(r'շահագործման կարգ', re.I), 4),
    (re.compile(r'վարման և պահպանման', re.I), 5),
    (re.compile(r'ստեղծման և վարման', re.I), 6),
    (re.compile(r'սահմանվում է', re.I), 2),
    (re.compile(r'սահմանվում են', re.I), 2),
    (re.compile(r'սահմանում է', re.I), 2),
    (re.compile(r'սույն կարգով', re.I), 2),
]

REFERENCE_PATTERNS = [
    (re.compile(r'շտեմարանից ստացված', re.I), -4),
    (re.compile(r'ռեգիստրից ստացված', re.I), -4),
    (re.compile(r'ռեեստրից ստացված', re.I), -4),
    (re.compile(r'կադաստրից ստացված', re.I), -4),
    (re.compile(r'ռեգիստրի տվյալ', re.I), -2),
    (re.compile(r'ռեեստրի տվյալ', re.I), -2),
    (re.compile(r'շտեմարանի տվյալ', re.I), -2),
    (re.compile(r'կադաստրի տվյալ', re.I), -2),
    (re.compile(r'համակարգից ստացված', re.I), -2),
]

DATA_WORDS = [
    'տվյալ', 'տեղեկություն', 'գրառում', 'գրանցում',
    'անվանում', 'անուն', 'հասցե', 'համար', 'ծածկագիր', 'կոդ',
    'նույնականաց', 'նույնացուցիչ',
]


def parse_query(q):
    words = re.findall(r'[ա-ֆև]+', q.lower())
    targets, actions = [], []
    for w in words:
        if w in STOPWORDS or len(w) < 3:
            continue
        is_verb = False
        stem = w
        for end in VERB_ENDINGS:
            if w.endswith(end) and len(w) > len(end):
                is_verb = True
                stem = w[: -len(end)]
                break
        if is_verb and len(stem) >= 3:
            actions.append(stem)
        else:
            targets.append(w[:5] if len(w) >= 5 else w)
    return targets, actions


def score_provision(text, targets, actions):
    score = 0
    evidence = []
    matched_target = None
    matched_action = None
    target_pos = None
    action_pos = None

    for t in targets:
        idx = text.find(t)
        if idx >= 0:
            score += 10
            evidence.append(f'TARGET({t}) +10')
            if matched_target is None:
                matched_target = t
                target_pos = idx

    for a in actions:
        m = re.search(r'\b' + re.escape(a), text)
        if m:
            score += 5
            evidence.append(f'ACTION({a}) +5')
            if matched_action is None:
                matched_action = a
                action_pos = m.start()

    if target_pos is not None and action_pos is not None:
        dist = abs(target_pos - action_pos)
        if dist <= 80:
            score += 3; evidence.append('VERY_CLOSE +3')
        elif dist <= 150:
            score += 2; evidence.append('CLOSE +2')
        elif dist <= 250:
            score += 1; evidence.append('PROXIMITY +1')

    d = sum(1 for w in DATA_WORDS if w in text)
    if d:
        bonus = min(d, 4)
        score += bonus
        evidence.append(f'DATA +{bonus}')

    lg = sum(w for p, w in LEGAL_PATTERNS if p.search(text))
    if lg:
        bonus = min(lg, 6)
        score += bonus
        evidence.append(f'LEGAL +{bonus}')

    ref = sum(w for p, w in REFERENCE_PATTERNS if p.search(text))
    if ref:
        ref = max(ref, -6)
        score += ref
        evidence.append(f'REFERENCE {ref}')

    return score, evidence, matched_target, matched_action


class Search:
    def __init__(self):
        print('loading TF-IDF ...')
        with open(INDEX / 'tfidf_vectorizer.pkl', 'rb') as f:
            self.vec = pickle.load(f)
        self.matrix = sp.load_npz(INDEX / 'tfidf_matrix.npz')

        print('loading embeddings ...')
        self.emb = np.load(INDEX / 'embeddings.npy')

        print('loading provisions ...')
        self.provisions = []
        with open(PROVISIONS, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.provisions.append(json.loads(line))

        print('loading model ...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(
            MODEL, device=device,
            model_kwargs={'torch_dtype': torch.float16} if device == 'cuda' else {},
        )
        self.model.max_seq_length = MAX_LEN
        print('ready.')

    def search(self, query, top_k=20, lexical_candidates=1000, alpha=0.4):
        targets, actions = parse_query(query)
        print(f'query: {query!r}')
        print(f'  targets: {targets}')
        print(f'  actions: {actions}')

        q_vec = self.vec.transform([query.lower()])
        sims = (self.matrix @ q_vec.T).toarray().ravel()
        k = min(lexical_candidates, len(sims) - 1)
        top_idx = np.argpartition(-sims, k)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

        results = []
        for i in top_idx:
            d = self.provisions[i]
            text_full = ((d.get('act_title_hy') or '') + '\n' + (d.get('text_hy') or '')).lower()
            sc, ev, mt, ma = score_provision(text_full, targets, actions)
            if sc <= 0:
                continue
            cos = float(self.emb[i].astype(np.float32) @ q_emb)
            combined = alpha * (sc / 30.0) + (1 - alpha) * cos
            results.append({
                'act_id': d.get('act_id'),
                'act_title_hy': d.get('act_title_hy'),
                'article_label': d.get('article_label'),
                'text_hy': d.get('text_hy'),
                'lexical_score': sc,
                'cosine': cos,
                'combined': combined,
                'evidence': ev,
                'matched_target': mt,
                'matched_action': ma,
            })

        results.sort(key=lambda r: r['combined'], reverse=True)
        return results[:top_k]


def print_results(results):
    for i, r in enumerate(results, 1):
        print('=' * 100)
        print(f'[{i}] combined={r["combined"]:.3f}  lex={r["lexical_score"]}  cos={r["cosine"]:.3f}')
        print(f'    act_id: {r["act_id"]}  article: {r["article_label"]}')
        print(f'    title: {(r["act_title_hy"] or "")[:100]}')
        print(f'    evidence: {", ".join(r["evidence"])}')
        print(f'    text: {(r["text_hy"] or "")[:500]}')
        print()


if __name__ == '__main__':
    q = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'ինչպես ստանալ վկայական'
    s = Search()
    print_results(s.search(q, top_k=20))
