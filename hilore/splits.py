"""Validate supplied frozen problem manifests against the DeepMath protocol."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


COUNTS = {'train': 10000, 'validation': 2048}
IDENTITIES = ('problem_id', 'variant_group', 'normalized_text_sha256')


def _records(rows, name, stratified=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{name} must contain problem records')
    seen_ids, seen_text = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f'{name}: invalid problem record')
        fields = IDENTITIES + (('topic', 'difficulty') if stratified else ())
        for field in fields:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'{name}: missing {field}')
        if not re.fullmatch(r'[0-9a-f]{64}', row['normalized_text_sha256']):
            raise ValueError(f'{name}: invalid normalized-text SHA-256')
        if row['problem_id'] in seen_ids or row['normalized_text_sha256'] in seen_text:
            raise ValueError(f'{name}: duplicate problem ID or normalized text')
        seen_ids.add(row['problem_id'])
        seen_text.add(row['normalized_text_sha256'])
    return rows


def _disjoint(left, right, description):
    for field in IDENTITIES:
        if {r[field] for r in left} & {r[field] for r in right}:
            raise ValueError(f'{description}: overlap by {field}')


class SplitManifest:
    """Enforce membership in supplied lists, without fabricating original splits."""

    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('Manifest must be a JSON object')
        if value.get('schema_version') != 1 or value.get('split_seed') != 2026:
            raise ValueError('Expected manifest schema 1 and split seed 2026')
        splits = value.get('splits', {})
        if set(splits) != set(COUNTS):
            raise ValueError('Expected exactly train and validation problem lists')
        self.rows = {}
        for name, count in COUNTS.items():
            self.rows[name] = _records(splits[name], name, stratified=True)
            if len(self.rows[name]) != count:
                raise ValueError(f'{name}: expected {count} unique problems')
        tests = value.get('test_benchmarks', {})
        if set(tests) != {'MATH500', 'GSM8K'}:
            raise ValueError('Both MATH500 and GSM8K overlap-reference lists are required')
        for name, rows in tests.items():
            _records(rows, name)
            if len(rows) != {'MATH500': 500, 'GSM8K': 1319}[name]:
                raise ValueError(f'{name}: incomplete evaluation reference count')
        _disjoint(self.rows['train'], self.rows['validation'], 'Train/validation')
        for split, rows in self.rows.items():
            for benchmark, test_rows in tests.items():
                _disjoint(rows, test_rows, f'{split}/{benchmark}')
        self.ids = {name: {r['problem_id'] for r in rows} for name, rows in self.rows.items()}
        self.benchmark_counts = {name: len(rows) for name, rows in tests.items()}

    def require_usage(self, problem_ids, purpose, digest):
        if digest != self.sha256:
            raise ValueError('Replay/calibration manifest digest does not match the frozen lists')
        split = {'train': 'train', 'calibration': 'train', 'validation': 'validation'}.get(purpose)
        if split is None:
            raise ValueError('Unsupported problem usage')
        if not isinstance(problem_ids, list) or not problem_ids or any(
            not isinstance(value, str) or value not in self.ids[split] for value in problem_ids
        ):
            raise ValueError(f'{purpose} contains a problem outside its permitted split')

    def summary(self):
        return {'manifest_sha256': self.sha256, 'split_seed': 2026,
                'counts': {name: len(rows) for name, rows in self.rows.items()},
                'supplied_benchmark_counts': self.benchmark_counts,
                'topic_difficulty_counts': {
                    name: [{'topic': topic, 'difficulty': difficulty, 'count': count}
                           for (topic, difficulty), count in sorted(Counter(
                               (r['topic'], r['difficulty']) for r in rows).items())]
                    for name, rows in self.rows.items()},
                'status': 'supplied_manifest_checks_passed',
                'unverified': ['original problem-list identity', 'source preprocessing and variant discovery',
                               'stratification algorithm and quotas', 'benchmark reference completeness']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = SplitManifest(args.manifest).summary()
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


def validate_replay_provenance(data, prompt_ids, manifest):
    """Bind replay groups to globally stable source IDs and one frozen manifest."""
    problem_ids = data.get('problem_ids')
    if not isinstance(problem_ids, list) or len(problem_ids) != len(prompt_ids):
        raise ValueError('Replay requires one stable problem ID per response')
    manifest.require_usage(problem_ids, 'train', data.get('split_manifest_sha256'))
    group_to_problem, problem_to_group = {}, {}
    for group, problem in zip(prompt_ids, problem_ids):
        if group in group_to_problem and group_to_problem[group] != problem:
            raise ValueError('A response group mixes different source problems')
        if problem in problem_to_group and problem_to_group[problem] != group:
            raise ValueError('A source problem appears in multiple prompt groups')
        group_to_problem[group], problem_to_group[problem] = problem, group


if __name__ == '__main__':
    main()
