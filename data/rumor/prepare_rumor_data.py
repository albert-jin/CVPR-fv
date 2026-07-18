"""Convert raw rumor-detection dumps into the JSONL format Det2Ver expects.

Usage
-----

    python prepare_rumor_data.py --dataset liar  --raw_dir liar_raw  --out_dir .
    python prepare_rumor_data.py --dataset fnn   --raw_dir fnn_raw   --out_dir .
    python prepare_rumor_data.py --dataset covid --raw_dir covid_raw --out_dir .

Or generate a tiny placeholder file for smoke-testing:

    python prepare_rumor_data.py --dataset dummy --out_dir .

Every output line follows the schema

    {"id": "...", "claim": "...", "label": "REAL" | "FAKE"}
"""

import argparse
import csv
import json
import os
import sys
from glob import glob

# Some FakeNewsNet cells (article bodies) exceed csv's default 128 KB
# field cap; raise it to whatever the platform allows.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2 ** 31 - 1)


LIAR_LABEL_MAP = {
    'true': 'REAL',
    'mostly-true': 'REAL',
    'false': 'FAKE',
    'pants-fire': 'FAKE',
    'barely-true': 'FAKE',
    # 'half-true' is intentionally dropped (Section IV-B)
}


def prepare_liar(raw_dir: str, out_dir: str):
    """LIAR TSV → JSONL. Columns (see README of the LIAR release):

    id, label, statement, subject, speaker, job, state_info, party,
    barely_true_c, false_c, half_true_c, mostly_true_c, pants_fire_c,
    context
    """
    splits = {
        'train': os.path.join(raw_dir, 'train.tsv'),
        'validation': os.path.join(raw_dir, 'valid.tsv'),
    }
    for split, path in splits.items():
        if not os.path.exists(path):
            print(f'[liar] missing {path}, skip', file=sys.stderr)
            continue
        out_path = os.path.join(out_dir, f'liar_{split}.jsonl')
        rows = 0
        with open(path, encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as g:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for r in reader:
                if len(r) < 3:
                    continue
                raw_id, raw_label, statement = r[0], r[1].strip().lower(), r[2]
                mapped = LIAR_LABEL_MAP.get(raw_label)
                if mapped is None:
                    continue
                g.write(json.dumps({
                    'id': f'liar_{raw_id}',
                    'claim': statement,
                    'label': mapped,
                }, ensure_ascii=False) + '\n')
                rows += 1
        print(f'[liar] {out_path} — {rows} rows')


def prepare_fnn(raw_dir: str, out_dir: str):
    """FakeNewsNet CSVs → JSONL.

    Looks for the four canonical files:

        <raw_dir>/dataset/politifact_real.csv
        <raw_dir>/dataset/politifact_fake.csv
        <raw_dir>/dataset/gossipcop_real.csv
        <raw_dir>/dataset/gossipcop_fake.csv
    """
    dataset_dir = os.path.join(raw_dir, 'dataset')
    if not os.path.isdir(dataset_dir):
        # Some clones put CSV files at the root.
        dataset_dir = raw_dir
    files = {
        'politifact_real': ('politifact_real.csv', 'REAL'),
        'politifact_fake': ('politifact_fake.csv', 'FAKE'),
        'gossipcop_real':  ('gossipcop_real.csv',  'REAL'),
        'gossipcop_fake':  ('gossipcop_fake.csv',  'FAKE'),
    }
    out_path = os.path.join(out_dir, 'fnn_train.jsonl')
    val_out_path = os.path.join(out_dir, 'fnn_validation.jsonl')
    rows_train, rows_val = 0, 0
    with open(out_path, 'w', encoding='utf-8') as g_train, \
         open(val_out_path, 'w', encoding='utf-8') as g_val:
        for key, (fname, mapped) in files.items():
            path = os.path.join(dataset_dir, fname)
            if not os.path.exists(path):
                print(f'[fnn] missing {path}, skip', file=sys.stderr)
                continue
            with open(path, encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    title = (row.get('title') or row.get('text') or '').strip()
                    if not title:
                        continue
                    obj = {
                        'id': f'fnn_{key}_{row.get("id", "")}',
                        'claim': title,
                        'label': mapped,
                    }
                    # Reserve 10% of every stream for validation.
                    if hash(obj['id']) % 10 == 0:
                        g_val.write(json.dumps(obj, ensure_ascii=False) + '\n')
                        rows_val += 1
                    else:
                        g_train.write(json.dumps(obj, ensure_ascii=False) + '\n')
                        rows_train += 1
    print(f'[fnn] {out_path} — {rows_train} rows train, {rows_val} rows val')


def prepare_covid(raw_dir: str, out_dir: str):
    """COVID-19 Fake News CSV → JSONL."""
    candidates = [
        os.path.join(raw_dir, 'data', 'Constraint_Train.csv'),
        os.path.join(raw_dir, 'Constraint_Train.csv'),
        os.path.join(raw_dir, 'data', 'Constraint_English_Train.csv'),
    ]
    train_path = next((p for p in candidates if os.path.exists(p)), None)
    if train_path is None:
        raise FileNotFoundError('Could not find COVID-19 Fake News train CSV.')

    val_candidates = [
        os.path.join(raw_dir, 'data', 'Constraint_Val.csv'),
        os.path.join(raw_dir, 'Constraint_Val.csv'),
        os.path.join(raw_dir, 'data', 'Constraint_English_Val.csv'),
    ]
    val_path = next((p for p in val_candidates if os.path.exists(p)), None)

    for split, path in [('train', train_path), ('validation', val_path)]:
        if path is None:
            continue
        out_path = os.path.join(out_dir, f'covid_{split}.jsonl')
        rows = 0
        with open(path, encoding='utf-8', errors='ignore') as f, \
             open(out_path, 'w', encoding='utf-8') as g:
            reader = csv.DictReader(f)
            for row in reader:
                claim = (row.get('tweet') or row.get('text') or row.get('claim') or '').strip()
                if not claim:
                    continue
                raw_label = str(row.get('label', '')).strip().lower()
                if raw_label in ('real', 'true', '1'):
                    mapped = 'REAL'
                elif raw_label in ('fake', 'false', '0'):
                    mapped = 'FAKE'
                else:
                    continue
                g.write(json.dumps({
                    'id': f'covid_{row.get("id", "")}',
                    'claim': claim,
                    'label': mapped,
                }, ensure_ascii=False) + '\n')
                rows += 1
        print(f'[covid] {out_path} — {rows} rows')


def prepare_dummy(out_dir: str):
    """Synthesise tiny placeholder RD files so the pipeline can be tested
    without any download. Each of the three datasets receives 40 rows
    (20 REAL + 20 FAKE)."""
    real_templates = [
        'The World Health Organization confirmed the new vaccine efficacy in trials.',
        'The city council announced a public transport upgrade next month.',
        'Scientists observed a new exoplanet in the habitable zone of a nearby star.',
        'The national bank published its quarterly inflation report today.',
    ]
    fake_templates = [
        'Drinking bleach cures the flu virus overnight, doctors reveal.',
        '5G towers are transmitting a mind-control signal, whistleblowers claim.',
        'The moon landing was faked in a Hollywood studio, new footage shows.',
        'Eating carrots grants perfect night vision within a week.',
    ]
    for ds in ('liar', 'fnn', 'covid'):
        path = os.path.join(out_dir, f'{ds}_train.jsonl')
        rows = 0
        with open(path, 'w', encoding='utf-8') as g:
            for i in range(20):
                g.write(json.dumps({
                    'id': f'dummy_{ds}_real_{i}',
                    'claim': real_templates[i % len(real_templates)] + f' Report #{i}.',
                    'label': 'REAL',
                }, ensure_ascii=False) + '\n')
                g.write(json.dumps({
                    'id': f'dummy_{ds}_fake_{i}',
                    'claim': fake_templates[i % len(fake_templates)] + f' (post {i})',
                    'label': 'FAKE',
                }, ensure_ascii=False) + '\n')
                rows += 2
        print(f'[dummy] {path} — {rows} rows')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=['liar', 'fnn', 'covid', 'dummy'])
    p.add_argument('--raw_dir', default='.', help='directory containing the raw dump')
    p.add_argument('--out_dir', default='.', help='where to write the .jsonl files')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset == 'liar':
        prepare_liar(args.raw_dir, args.out_dir)
    elif args.dataset == 'fnn':
        prepare_fnn(args.raw_dir, args.out_dir)
    elif args.dataset == 'covid':
        prepare_covid(args.raw_dir, args.out_dir)
    elif args.dataset == 'dummy':
        prepare_dummy(args.out_dir)


if __name__ == '__main__':
    main()
