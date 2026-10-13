import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'experiments/e0_shared_space.py'
spec = importlib.util.spec_from_file_location('e0', path)
e0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e0)


class IntegrityTests(unittest.TestCase):
    def test_smoke_never_claims_scientific_pass(self):
        self.assertEqual(e0.evaluate(e0.smoke_data())['status'], 'smoke_only')

    def test_split_group_leakage_rejected(self):
        data = e0.smoke_data()
        data['validation'][0]['group'] = data['train'][0]['group']
        with self.assertRaisesRegex(ValueError, 'family overlap'):
            e0.evaluate(data)

    def test_duplicate_recipe_rejected_even_with_different_group(self):
        data = e0.smoke_data()
        data['validation'][0]['formula_ppm'] = data['train'][0]['formula_ppm'].copy()
        with self.assertRaisesRegex(ValueError, 'formula overlaps'):
            e0.evaluate(data)

    def test_invalid_formulas_rejected(self):
        for bad in ({'MAT_99999999': 1000000}, {'MAT_00000001': 999999},
                    {'MAT_00000001': True}, {'MAT_00000001': 1000000.0}):
            with self.subTest(bad=bad):
                data = e0.smoke_data()
                data['validation'][0]['formula_ppm'] = bad
                with self.assertRaises(ValueError):
                    e0.evaluate(data)

    def test_nonfinite_label_rejected(self):
        data = e0.smoke_data()
        data['validation'][0]['ratings'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            e0.evaluate(data)

    def test_validation_labels_do_not_affect_predictions(self):
        data = e0.smoke_data()
        before = e0.evaluate(data)['example']
        for row in data['validation']:
            row['ratings'] = [0.0, 0.0]
        self.assertEqual(before, e0.evaluate(data)['example'])

    def test_small_human_dataset_cannot_pass_gate(self):
        data = e0.smoke_data()
        data['evidence'] = 'human_observed'
        self.assertEqual(e0.evaluate(data)['status'], 'insufficient_evidence')

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            json.loads('{"a":1,"a":2}', object_pairs_hook=e0.unique_object)


if __name__ == '__main__':
    unittest.main()
