import argparse
import ast
import copy
import math
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import statistics
import unittest
from unittest.mock import Mock

from hilore.profile_schema import choose_capture, development_acceptable, shortlist, validate_profile


ROOT = Path(__file__).resolve().parents[1]


def definitions(filename, names, namespace):
    tree = ast.parse((ROOT / 'hilore' / filename).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(selected) != len(names):
        raise AssertionError('Requested production definitions are missing')
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


def fixture(initializing=False):
    plan = {'shape': [1, 3072], 'representations': ['H', 'R'],
            'units': [{'actions': ['R', 'H', 'L'], 'recomputation_verified': True, 'selected_index': 0},
                      {'actions': ['R'], 'recomputation_verified': False, 'selected_index': 0}],
            'utility_seconds': [[0, 0.2, 0.1], [0, 0, 0]],
            'marginal_peak_bytes': [[0, 1, 1], [0, 0, 0]],
            'measurement_scope': 'microbatch', 'estimated_same_plan_all_r_peak_bytes': 105,
            'microbatch_all_r_peak_bytes': 80, 'microbatch_all_r_seconds': 3,
            'estimated_microbatch_seconds': 2.8, 'interaction_screening_passed': True,
            'validation_scope': 'complete_update', 'validated_peak_bytes': 108,
            'validated_actor_update_seconds': 90}
    return {'schema_version': 3, 'configuration_fingerprint': 'fingerprint',
            'measured_gc_peak_bytes': 100, 'memory_multiplier': 1.1, 'risk_budget': 0.01,
            'memory_metric': 'nvml_device_used_max', 'measurement_world_size': 4,
            'status': 'initializing' if initializing else 'development_only',
            'joint_validation_passed': not initializing, 'terminal_quality_passed': False,
            'initialization': {'protocol': 'bounded_microbatch_v1', 'candidate_limit': 8,
                'calibration_microbatches': 16, 'microbatch_repeats': 2, 'microbatch_warmups': 1,
                'interaction_round_limit': 2, 'complete_update_pairs': 1, 'complete_update_warmups': 0},
            'plans': [plan], 'calibration_shape': [1, 3072],
            'calibration': {'0': {'group': 'mlp/0', 'selected_index': 0, 'kappa': 0.5, 'mean_error': 0.01}},
            'development_measurements': [{'measurement_scope': 'complete_update', 'seconds': 90,
                                         'gc_seconds': 100, 'peak_bytes': 108, 'gradient_error': 0.01,
                                         'recovery_action_counts': {'R': 100, 'H': 8, 'L': 4}}]}


class ProfileTests(unittest.TestCase):
    def check(self, profile, **kwargs):
        return validate_profile(profile, 'fingerprint', 1.1, 0.01, **kwargs)

    def test_development_profile_requires_explicit_permission(self):
        with self.assertRaises(ValueError):
            self.check(fixture())
        self.assertAlmostEqual(self.check(fixture(), allow_development=True), 110)

    def test_unvalidated_profile_is_not_a_training_profile(self):
        profile = fixture(True)
        with self.assertRaises(ValueError):
            self.check(profile, allow_development=True)
        self.check(profile, initializing=True)
        profile['joint_validation_passed'] = True
        with self.assertRaises(ValueError):
            self.check(profile, initializing=True)

    def test_rejected_and_old_profiles_cannot_load(self):
        for change in ({'status': 'rejected'}, {'status': 'failed'}, {'schema_version': 2},
                       {'configuration_fingerprint': 'another-model'}, {'risk_budget': 0.02}):
            with self.subTest(change=change):
                profile = fixture()
                profile.update(change)
                with self.assertRaises(ValueError):
                    self.check(profile, allow_development=True)

    def test_microbatch_errors_cannot_certify_complete_update(self):
        profile = fixture()
        profile['development_measurements'][0]['measurement_scope'] = 'microbatch'
        with self.assertRaises(ValueError):
            self.check(profile, allow_development=True)

    def test_failed_fidelity_memory_and_speed_are_rejected(self):
        for field, value in [('gradient_error', 0.016), ('gradient_error', float('nan')),
                             ('peak_bytes', 111), ('seconds', 101), ('seconds', 0)]:
            with self.subTest(field=field, value=value):
                profile = fixture()
                profile['development_measurements'][0][field] = value
                with self.assertRaises(ValueError):
                    self.check(profile, allow_development=True)

    def test_l_requires_calibration_for_the_exact_component(self):
        for change in ({}, {'0': {'group': 'mlp/0', 'selected_index': 1, 'kappa': 1, 'mean_error': 0.01}}):
            profile = fixture()
            profile['calibration'] = change
            with self.assertRaises(ValueError):
                self.check(profile, allow_development=True)

    def test_complete_all_r_cannot_be_published_as_hilore(self):
        profile = fixture()
        profile['development_measurements'][0]['recovery_action_counts'] = {'R': 112, 'H': 0, 'L': 0}
        with self.assertRaises(ValueError):
            self.check(profile, allow_development=True)

    def test_plan_must_match_the_actual_validation_measurement(self):
        profile = fixture()
        profile['plans'][0]['validated_peak_bytes'] = 107
        with self.assertRaises(ValueError):
            self.check(profile, allow_development=True)

    def test_capture_limit_and_action_compatibility(self):
        profile = fixture()
        profile['plans'][0]['representations'][1] = 'H'
        profile['initialization']['candidate_limit'] = 1
        with self.assertRaises(ValueError):
            self.check(profile, allow_development=True)
        profile = fixture()
        profile['plans'][0]['units'][1]['actions'] = ['R', 'H']
        with self.assertRaises(ValueError):
            self.check(profile, allow_development=True)

    def test_no_plan_and_unverified_plan_are_rejected(self):
        for field, value in [('plans', []), ('calibration_shape', [2, 2048])]:
            profile = fixture()
            profile[field] = value
            with self.assertRaises(ValueError):
                self.check(profile, allow_development=True)
        profile = fixture()
        profile['plans'][0]['interaction_screening_passed'] = False
        with self.assertRaises(ValueError):
            self.check(profile, allow_development=True)

    def test_fidelity_is_not_accepted_from_empty_or_nonfinite_rows(self):
        self.assertFalse(development_acceptable([], 110))
        self.assertFalse(development_acceptable(fixture()['development_measurements'], float('inf')))
        self.assertTrue(development_acceptable(fixture()['development_measurements'], 110))


class SelectionTests(unittest.TestCase):
    def test_shortlist_respects_headroom_ranking_and_limit(self):
        units = [{'capture_bytes_estimate': {'H': 10, 'L': 8}, 'recomputation_seconds_estimate': value}
                 for value in (1, 4, 3, 4)]
        units.append({'capture_bytes_estimate': {'H': 100, 'L': 90}, 'recomputation_seconds_estimate': 100})
        self.assertEqual(shortlist(units, 2, 20), [1, 3])
        self.assertEqual(shortlist(units, 8, 7), [])

    def test_capture_fractions_preserve_separate_budgets(self):
        ranked = [{'unit': i, 'H': {'capture_cost': 4}, 'L': {'capture_cost': 3}} for i in range(4)]
        self.assertEqual(choose_capture(ranked, 0.5, 12, 4), ['H', 'L', 'L', 'R'])
        self.assertEqual(choose_capture(ranked, 1, 12, 4), ['H', 'H', 'H', 'R'])
        self.assertEqual(choose_capture(ranked, 0, 12, 4), ['L'] * 4)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.cuda = SimpleNamespace(empty_cache=Mock(), synchronize=Mock(), reset_peak_memory_stats=Mock(),
                                    max_memory_allocated=lambda: 10, max_memory_reserved=lambda: 12)
        self.dist = SimpleNamespace(is_initialized=lambda: True, barrier=Mock())
        @contextmanager
        def sampler():
            yield SimpleNamespace(peak=15)
        counter = iter(range(1000))
        self.ns = definitions('profiling.py', {'measure_execution', 'measure_microbatch', 'measure_update_once', 'paired_measure'},
            {'torch': SimpleNamespace(cuda=self.cuda), 'dist': self.dist, 'gc': SimpleNamespace(collect=Mock()),
             'time': SimpleNamespace(perf_counter=lambda: next(counter)), 'MemorySampler': sampler,
             'rank_max': lambda values: values, 'statistics': statistics})

    def test_complete_update_is_called_once_without_full_warmup(self):
        run = Mock(return_value={'complete_update': True, 'after_reduction_before_clipping': True, 'gradients': {'g': 1}})
        restore = Mock()
        result = self.ns['measure_update_once'](run, restore, [], [])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(restore.call_count, 1)
        self.assertEqual(result['measurement_scope'], 'complete_update')
        self.assertEqual(result['warmups'], 0)
        self.assertEqual(result['runs'][0]['gradients'], {'g': 1})

    def test_microbatch_has_one_warmup_and_two_repetitions(self):
        run = Mock(return_value={'complete_update': False, 'after_reduction_before_clipping': True})
        result = self.ns['measure_microbatch'](run, Mock(), ['H'], ['R'])
        self.assertEqual(run.call_count, 3)
        self.assertEqual(len(result['runs']), 2)
        self.assertEqual(result['warmups'], 1)
        self.assertNotIn('gradients', result['runs'][0])

    def test_measurement_scope_cannot_be_mislabeled(self):
        run = Mock(return_value={'complete_update': False, 'after_reduction_before_clipping': True})
        with self.assertRaises(ValueError):
            self.ns['measure_update_once'](run, Mock(), [], [])
        run = Mock(return_value={'complete_update': True, 'after_reduction_before_clipping': True})
        with self.assertRaises(ValueError):
            self.ns['measure_microbatch'](run, Mock(), [], [])

    def test_formal_repetitions_remain_five(self):
        run = Mock(return_value={'complete_update': True, 'after_reduction_before_clipping': True, 'gradients': {}})
        result = self.ns['paired_measure'](run, Mock(), [], [])
        self.assertEqual(run.call_count, 6)
        self.assertEqual(result['repetitions'], 5)


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.ns = definitions('initialize.py', {'InitializationRejected', 'Initializer'},
            {'contextmanager': contextmanager, 'copy': copy, 'choose_capture': choose_capture})
        self.init = self.ns['Initializer'].__new__(self.ns['Initializer'])
        self.init.interaction_rounds = 2

    def test_stable_plan_passes_without_additional_joint_measurement(self):
        self.init.dynamic_batch = Mock(return_value={'seconds': 11})
        profile = fixture(True)
        self.init.validate_interactions(None, profile, 'identity', {'seconds': 10, 'active_units': [0]})
        self.assertEqual(self.init.dynamic_batch.call_count, 1)
        self.assertTrue(profile['plans'][0]['interaction_screening_passed'])
        self.assertEqual(profile['plans'][0]['interaction_rounds'], 1)

    def test_pruning_does_not_change_capture(self):
        profile = fixture(True)
        plan = profile['plans'][0]
        plan['representations'] = ['H', 'H']
        plan['units'][1]['actions'] = ['R', 'H']
        plan['utility_seconds'][1] = [0, 0.1, 0]
        self.init.dynamic_batch = Mock(side_effect=[{'seconds': 9}, {'seconds': 11},
                                                   {'seconds': 9, 'active_units': [1]}, {'seconds': 10}])
        self.init.validate_interactions(None, profile, 'identity', {'seconds': 10, 'active_units': [0, 1]})
        self.assertEqual(plan['representations'], ['H', 'H'])
        self.assertEqual(plan['units'][0]['actions'], ['R'])
        self.assertEqual(plan['utility_seconds'][0], [0, 0, 0])
        self.assertTrue(plan['interaction_screening_passed'])

    def test_unstable_second_round_rejects_instead_of_falling_back(self):
        profile = fixture(True)
        profile['plans'][0]['units'][1]['actions'] = ['R', 'H']
        self.init.dynamic_batch = Mock(side_effect=[{'seconds': 9}, {'seconds': 11},
                                                   {'seconds': 9, 'active_units': [1]}, {'seconds': 8}])
        with self.assertRaises(self.ns['InitializationRejected']):
            self.init.validate_interactions(None, profile, 'identity', {'seconds': 10, 'active_units': [0, 1]})
        self.assertEqual(self.init.dynamic_batch.call_count, 4)

    def test_all_r_is_not_a_success(self):
        with self.assertRaises(self.ns['InitializationRejected']):
            self.init.validate_interactions(None, fixture(True), 'identity', {'seconds': 10, 'active_units': []})


class WorkloadTests(unittest.TestCase):
    def test_graph_verifies_only_the_shortlisted_units(self):
        class Profiler:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def events(self):
                return []
        units = [{'saved_state_signature': [], 'selected_index': 0, 'eligible': True,
                  'capture_bytes_estimate': {'H': 2, 'L': 1}, 'actions': ['R'],
                  'recomputation_verified': False} for _ in range(4)]
        torch = SimpleNamespace(profiler=SimpleNamespace(profile=lambda **kwargs: Profiler(),
                                ProfilerActivity=SimpleNamespace(CPU=1, CUDA=2)))
        ns = definitions('initialize.py', {'Initializer', 'InitializationRejected'},
            {'contextmanager': contextmanager, 'torch': torch, 'shortlist': shortlist,
             'capture_metadata': lambda *args: copy.deepcopy(units),
             'across_ranks': lambda value: [copy.deepcopy(value) for _ in range(4)],
             'recomputation_seconds': lambda events, name: [1, 4, 3, 2][int(name.split('/')[-1])],
             'verify_elimination': lambda *args: {'kernel': 1}})
        init = ns['Initializer'].__new__(ns['Initializer'])
        init.count, init.candidate_limit, init.all_r = 4, 2, ['R'] * 4
        init.restore = Mock()
        def batch(value, q, actions, observer=None):
            if observer:
                observer(None, None, None)
        init.batch = Mock(side_effect=batch)
        metadata, candidates = init.graph(None, 100)
        self.assertEqual(candidates, [1, 2])
        self.assertEqual(init.batch.call_count, 5)
        self.assertEqual(metadata[0]['actions'], ['R'])
        self.assertEqual(metadata[1]['actions'], ['R', 'H', 'L'])
        self.assertEqual(metadata[2]['actions'], ['R', 'H', 'L'])
        self.assertEqual(metadata[3]['actions'], ['R'])

    def test_eight_units_keep_sixteen_isolated_calibration_samples(self):
        fit = definitions('calibration.py', {'fit'}, {'math': math, 'defaultdict': defaultdict})['fit']
        ns = definitions('initialize.py', {'Initializer'},
            {'contextmanager': contextmanager, 'math': math, 'fit': fit,
             'across_ranks': lambda values: [values], 'summed': lambda values: values,
             'exposure': lambda *args: 1, 'gradient_error': lambda *args, **kwargs: 0.01})
        init = ns['Initializer'].__new__(ns['Initializer'])
        ids = SimpleNamespace(tolist=lambda: [0])
        init.data = {'prompt_ids': ids, 'problem_ids': ['problem']}
        init.batches = [{'prompt_ids': ids}] * 16
        init.all_r, init.restore = ['R'] * 8, Mock()
        init.manifest = SimpleNamespace(sha256='manifest', require_usage=Mock())
        init.recovery = SimpleNamespace(measure_error=False, bundles={i: SimpleNamespace(
            error_numerator=1, state_squared_norm=100, selected=0) for i in range(8)})
        def batch(value, q, actions, observer=None, collect_gradients=False):
            if observer:
                observer(init.recovery, None, {'response_mask': SimpleNamespace(bool=lambda: None)})
            return {'gradients': {}}
        init.batch = Mock(side_effect=batch)
        result = init.calibrate(list(range(16)), list(range(8)))
        self.assertEqual(init.batch.call_count, 144)
        self.assertEqual(len(result), 8)
        self.assertEqual(init.manifest.require_usage.call_count, 128)
        self.assertTrue(all(row['selected_index'] == 0 for row in result.values()))


class RuntimeTests(unittest.TestCase):
    def test_unprofiled_shape_uses_exact_recovery(self):
        namespace = definitions('runtime.py', {'Scheduler'},
            {'validate_profile': validate_profile, 'warmup': lambda: None})
        scheduler = namespace['Scheduler'](fixture(), 'fingerprint', 1.1, 0.01, allow_development=True)
        self.assertEqual(scheduler.before_forward([1, 3072], 2), ['H', 'R'])
        self.assertEqual(scheduler.before_forward([2, 2048], 2), ['R', 'R'])
        self.assertIsNone(scheduler.plan)

    def test_cli_defaults_match_the_protocol(self):
        namespace = definitions('initialize.py', {'make_parser'}, {'argparse': argparse, 'Path': Path, 'CONFIG_PATH': 'config.json'})
        args = namespace['make_parser']().parse_args(['--split-manifest', 'manifest.json', '--updates', 'updates',
                                                     '--output', 'output', '--risk-budget', '0.01'])
        self.assertEqual((args.candidate_limit, args.microbatch_repeats, args.interaction_rounds), (8, 2, 2))


if __name__ == '__main__':
    unittest.main()
