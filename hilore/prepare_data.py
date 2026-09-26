"""Reconstruct a frozen DeepMath split from local JSONL source snapshots."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata

from .splits import SplitManifest


def normalize(text):
    """Normalize Unicode and whitespace without changing mathematical case."""
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path, namespace, training=False, difficulty_width=1.0):
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get('question', item.get('problem'))
            if not isinstance(text, str) or not normalize(text):
                raise ValueError(f'{namespace}: invalid question at row {line_number}')
            digest = hashlib.sha256(normalize(text).encode()).hexdigest()
            identity = item.get('problem_id', f"{namespace}/{item.get('id', digest)}")
            row = {'problem_id': str(identity),
                   'variant_group': str(item.get('variant_group', f'text/{digest}')),
                   'normalized_text_sha256': digest, 'question': text}
            answer = item.get('final_answer', item.get('answer', item.get('solution')))
            if answer is None:
                raise ValueError(f'{namespace}: missing answer at row {line_number}')
            row['answer'] = str(answer)
            if training:
                topic, difficulty = item.get('topic'), float(item['difficulty'])
                if not isinstance(topic, str) or not topic.strip() or not math.isfinite(difficulty):
                    raise ValueError('Source topic and finite difficulty are required')
                row.update(topic=topic.strip(), difficulty=str(math.floor(difficulty / difficulty_width)),
                           source_difficulty=difficulty)
            rows.append(row)
    if not rows:
        raise ValueError(f'{namespace}: empty source snapshot')
    return rows


def clean_source(source, benchmarks):
    """Merge declared variants and exact normalized duplicates before exclusion."""
    all_rows = source + [r for rows in benchmarks.values() for r in rows]
    parent = list(range(len(all_rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(all_rows):
        for field in ('problem_id', 'variant_group', 'normalized_text_sha256'):
            key = (field, row[field])
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    excluded = {root(i) for i in range(len(source), len(all_rows))}
    groups = defaultdict(list)
    for i, row in enumerate(source):
        if root(i) not in excluded:
            groups[root(i)].append(row)
    # One representative per connected group gives exact split sizes and prevents
    # a large variant family from dominating a stratum.
    return [min(rows, key=lambda r: (r['normalized_text_sha256'], r['problem_id']))
            for rows in groups.values()]


def stratified_take(rows, count, rng):
    """Use proportional largest-remainder quotas in topic/difficulty strata."""
    if len(rows) < count:
        raise ValueError('Not enough eligible distinct problem groups')
    strata = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r['normalized_text_sha256'], r['problem_id'])):
        strata[(row['topic'], row['difficulty'])].append(row)
    keys = sorted(strata)
    quotas = {k: count * len(strata[k]) // len(rows) for k in keys}
    order = sorted(keys, key=lambda k: (-(count * len(strata[k]) % len(rows)), k))
    for key in order[:count - sum(quotas.values())]:
        quotas[key] += 1
    selected, remaining = [], []
    for key in keys:
        rng.shuffle(strata[key])
        selected.extend(strata[key][:quotas[key]])
        remaining.extend(strata[key][quotas[key]:])
    rng.shuffle(selected)
    return selected, remaining


def construct(source, benchmarks, seed=2026, train_count=10000, validation_count=2048):
    cleaned = clean_source(source, benchmarks)
    rng = random.Random(seed)
    selected, _ = stratified_take(cleaned, train_count + validation_count, rng)
    validation, train = stratified_take(selected, validation_count, rng)
    return train, validation, len(cleaned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--math500', type=Path, required=True)
    parser.add_argument('--gsm8k', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--difficulty-bin-width', type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory')
    if not math.isfinite(args.difficulty_bin_width) or args.difficulty_bin_width <= 0:
        parser.error('Difficulty bin width must be finite and positive')
    benchmarks = {'MATH500': read_rows(args.math500, 'MATH500'),
                  'GSM8K': read_rows(args.gsm8k, 'GSM8K')}
    if len(benchmarks['MATH500']) != 500 or len(benchmarks['GSM8K']) != 1319:
        parser.error('Supply the full 500/1319-problem evaluation reference snapshots')
    source = read_rows(args.source, 'DeepMath-103K', True, args.difficulty_bin_width)
    train, validation, eligible = construct(source, benchmarks)
    fields = ('problem_id', 'variant_group', 'normalized_text_sha256', 'topic', 'difficulty')
    def metadata(rows):
        return [{k: row[k] for k in fields if k in row} for row in rows]
    value = {'schema_version': 1, 'split_seed': 2026,
             'splits': {'train': metadata(train), 'validation': metadata(validation)},
             'test_benchmarks': {name: metadata(rows) for name, rows in benchmarks.items()},
             'construction': {'status': 'reconstructed_not_original',
                 'normalization': 'NFKC and collapsed whitespace, case preserved',
                 'stratification': 'topic and floor(difficulty/bin_width), proportional largest remainder',
                 'difficulty_bin_width': args.difficulty_bin_width,
                 'variant_policy': 'one representative per declared-variant/duplicate connected group',
                 'eligible_groups': eligible,
                 'source_sha256': {name: file_digest(path)
                     for name, path in [('DeepMath-103K', args.source), ('MATH500', args.math500), ('GSM8K', args.gsm8k)]}}}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    manifest = SplitManifest(manifest_path)
    for name, rows in [('train', train), ('validation', validation)]:
        with (args.output / f'{name}.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'manifest_sha256': manifest.sha256, 'eligible_groups': eligible,
                      'status': 'reconstructed_not_original'}))


if __name__ == '__main__':
    main()
