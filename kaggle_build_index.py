#!/usr/bin/env python3
import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


INPUT = '/kaggle/input/datasets/ineedyoutowork/armenian-legal-provisions/hastat2_provisions.jsonl'
OUT = Path('/kaggle/working/index')

MODEL = 'BAAI/bge-m3'
SHARD = 20_000
BATCH = 64
MAX_LEN = 512


def iter_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build_embed_text(d):
    title = (d.get('act_title_hy') or '').strip()
    label = (d.get('article_label') or '').strip()
    body = (d.get('text_hy') or '').strip()
    header = f'{title} | {label}' if title and label else (title or label)
    return f'{header}\n{body}'.strip() if header else body


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print('reading provisions ...')
    ids = []
    texts = []
    for i, d in enumerate(iter_jsonl(INPUT)):
        ids.append(d.get('provision_id') or d.get('act_id') or str(i))
        texts.append(build_embed_text(d))
        if (i + 1) % 100_000 == 0:
            print(f'  {i+1:,} loaded')
    n = len(texts)
    print(f'{n:,} provisions loaded')

    (OUT / 'ids.txt').write_text('\n'.join(ids), encoding='utf-8')

    if (OUT / 'tfidf_matrix.npz').exists() and (OUT / 'tfidf_vectorizer.pkl').exists():
        print('\nTF-IDF already built — skipping')
    else:
        print('\nbuilding TF-IDF ...')
        t0 = time.time()
        vec = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=3,
            max_features=500_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        X = vec.fit_transform(texts)
        sp.save_npz(OUT / 'tfidf_matrix.npz', X)
        with open(OUT / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vec, f)
        print(f'  TF-IDF: {X.shape} nnz={X.nnz:,} in {time.time()-t0:.0f}s')

    print(f'\nloading {MODEL} ...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
        MODEL,
        device=device,
        model_kwargs={'torch_dtype': torch.float16} if device == 'cuda' else {},
    )
    model.max_seq_length = MAX_LEN
    print(f'device={model.device}  max_seq_length={model.max_seq_length}')

    n_shards = (n + SHARD - 1) // SHARD
    t0 = time.time()
    done = 0
    for s in range(n_shards):
        dst = OUT / f'emb_{s:04d}.npy'
        lo, hi = s * SHARD, min((s + 1) * SHARD, n)
        if dst.exists():
            done += hi - lo
            continue
        ts = time.time()
        emb = model.encode(
            texts[lo:hi],
            batch_size=BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        np.save(dst, emb.astype(np.float16))
        done += hi - lo
        rate = (hi - lo) / max(0.01, time.time() - ts)
        eta = (n - done) / rate / 60 if rate else 0
        print(f'  shard {s+1}/{n_shards}  {hi:,}/{n:,}  {rate:.0f} rows/s  ETA {eta:.1f} min', flush=True)
        torch.cuda.empty_cache()

    print('\nmerging shards ...')
    parts = [np.load(OUT / f'emb_{s:04d}.npy') for s in range(n_shards)]
    all_emb = np.concatenate(parts, axis=0)
    assert all_emb.shape[0] == n
    np.save(OUT / 'embeddings.npy', all_emb)
    print(f'saved {all_emb.shape} fp16  ({all_emb.nbytes/1e9:.2f} GB)  total {(time.time()-t0)/60:.1f} min')

    for s in range(n_shards):
        (OUT / f'emb_{s:04d}.npy').unlink()
    print('done.')


if __name__ == '__main__':
    main()
