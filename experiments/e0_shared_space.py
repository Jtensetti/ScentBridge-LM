"""CPU reference experiment, not a perfume generator or safety validator."""
import argparse
import hashlib
import json
import math
import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 1729


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(data):
    attrs, materials = data['attributes'], data['material_ids']
    for names in (attrs, materials):
        require(isinstance(names, list) and names, 'Nonempty ordered names required')
        require(all(isinstance(s, str) and s.strip() for s in names), 'Invalid name')
        require(len(names) == len(set(names)), 'Duplicate dimension names')
    require(all(re.fullmatch(r'MAT_[0-9]{8}', s) for s in materials), 'Invalid material ID')
    require(data['rights_status'] == 'cleared', 'Input rights must be cleared externally')
    require(data['evidence'] in ('human_observed', 'synthetic'), 'Unknown evidence class')
    require(5 <= len(data['train']) <= 5000, 'Need 5–5000 training rows')
    require(2 <= len(data['validation']) <= 5000, 'Need 2–5000 validation rows')
    ids, train_groups, train_formulas = set(), set(), set()
    for split in ('train', 'validation'):
        for row in data[split]:
            require(isinstance(row['id'], str) and row['id'], 'Invalid record ID')
            require(row['id'] not in ids, 'Duplicate record ID')
            ids.add(row['id'])
            require(isinstance(row['group'], str) and row['group'], 'Missing group')
            require(isinstance(row['text'], str) and row['text'].strip(), 'Missing text')
            formula = row['formula_ppm']
            require(isinstance(formula, dict) and formula, 'Missing formula')
            require(set(formula) <= set(materials), 'Unknown material')
            require(all(type(v) is int and 0 < v <= 1000000 for v in formula.values()),
                    'Amounts must be positive integer ppm')
            require(sum(formula.values()) == 1000000, 'Mass balance must be 1000000 ppm')
            ratings = row['ratings']
            require(len(ratings) == len(attrs), 'Rating dimension mismatch')
            require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                        for v in ratings), 'Ratings must be finite values in [0,1]')
            signature = tuple(sorted(formula.items()))
            if split == 'train':
                train_groups.add(row['group'])
                train_formulas.add(signature)
            else:
                require(row['group'] not in train_groups, 'Split family overlap')
                require(signature not in train_formulas, 'Exact formula overlaps splits')


def tokens(text):
    return re.findall(r'[^\W\d_]+', text.lower(), flags=re.UNICODE)


def text_matrix(rows, vocabulary):
    index = {word: i for i, word in enumerate(vocabulary)}
    result = np.zeros((len(rows), len(vocabulary)))
    for i, row in enumerate(rows):
        for word, count in Counter(tokens(row['text'])).items():
            if word in index:
                result[i, index[word]] = count
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1)


def formula_matrix(rows, materials):
    return np.sqrt([[row['formula_ppm'].get(m, 0) / 1000000 for m in materials]
                    for row in rows])


def ridge_predict(x, y, query, alpha=1.0):
    """Centered dual ridge: cost depends on pilot sample count, not vocabulary size."""
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    centered = x - xm
    dual = np.linalg.solve(centered @ centered.T + alpha * np.eye(len(x)), y - ym)
    return np.clip((query - xm) @ centered.T @ dual + ym, 0, 1)


def improvement_interval(baseline_error, model_error, groups):
    # Resample entire groups; retain all observations in each sampled group.
    unique = sorted(set(groups))
    delta = baseline_error - model_error
    sums = np.array([delta[np.array(groups) == g].sum() for g in unique])
    counts = np.array([groups.count(g) for g in unique])
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(unique), size=(1000, len(unique)))
    values = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return np.quantile(values, [0.025, 0.975]).tolist()


def evaluate(data):
    start = time.perf_counter()
    validate(data)
    train, val = data['train'], data['validation']
    attrs, materials = data['attributes'], data['material_ids']
    vocabulary = sorted({t for row in train for t in tokens(row['text'])})
    require(bool(vocabulary), 'Training vocabulary is empty')
    xt, xv = text_matrix(train, vocabulary), text_matrix(val, vocabulary)
    ft, fv = formula_matrix(train, materials), formula_matrix(val, materials)
    yt, yv = np.array([r['ratings'] for r in train]), np.array([r['ratings'] for r in val])
    pred_text = ridge_predict(xt, yt, xv)
    pred_formula = ridge_predict(ft, yt, fv)
    catalog_z = ridge_predict(ft, yt, ft)
    baseline_error = np.abs(yv - yt.mean(axis=0)).mean(axis=1)
    baseline_mae = float(baseline_error.mean())
    groups = [r['group'] for r in val]
    metrics = {}
    for name, prediction in [('text_to_attributes', pred_text), ('formula_to_attributes', pred_formula)]:
        errors = np.abs(prediction - yv).mean(axis=1)
        mae = float(errors.mean())
        metrics[name] = {
            'mae': mae,
            'relative_improvement': (baseline_mae - mae) / baseline_mae if baseline_mae > 0 else None,
            'paired_improvement_95ci': improvement_interval(baseline_error, errors, groups),
        }
    nearest = [int(np.argmin(((catalog_z - z) ** 2).mean(axis=1))) for z in pred_text]
    retrieved_attributes = yt[nearest]
    numeric_pass = all(m['relative_improvement'] is not None and
                       m['relative_improvement'] >= 0.10 and m['paired_improvement_95ci'][0] > 0
                       for m in metrics.values())
    if data['evidence'] == 'synthetic':
        status = 'smoke_only'
    elif len(val) < 100 or len(set(groups)) < 30:
        status = 'insufficient_evidence'
    else:
        status = 'numeric_gate_pass_needs_external_audit' if numeric_pass else 'numeric_gate_fail'
    encoded = json.dumps(data, sort_keys=True, allow_nan=False).encode()
    return {
        'status': status,
        'data_sha256': hashlib.sha256(encoded).hexdigest(),
        'seed': SEED, 'python': platform.python_version(), 'numpy': np.__version__,
        'train_rows': len(train), 'validation_rows': len(val),
        'validation_groups': len(set(groups)), 'attribute_dimensions': len(attrs),
        'text_dimensions': len(vocabulary), 'material_dimensions': len(materials),
        'train_mean_mae': baseline_mae, 'metrics': metrics,
        'catalog_retrieval_attribute_mae': float(np.abs(retrieved_attributes - yv).mean()),
        'example': {
            'description_to_catalog_formula_ppm': train[nearest[0]]['formula_ppm'],
            'formula_to_description': '; '.join(f'{a}: {v:.2f}/1' for a, v in zip(attrs, pred_formula[0])),
            'constraint_status': 'not_evaluated',
        },
        'wall_seconds': time.perf_counter() - start,
        'limitations': 'Catalog retrieval and attribute rendering only; no sensory or safety claim.',
    }


def smoke_data():
    data = {'attributes': ['floral', 'woody'],
            'material_ids': ['MAT_00000001', 'MAT_00000002'],
            'rights_status': 'cleared', 'evidence': 'synthetic', 'train': [], 'validation': []}
    # Deliberately fictional linear world for checking execution, never scientific evidence.
    for i in range(1, 37):
        share = i / 37
        floral_words = max(1, round(share * 10))
        amount = round(share * 1000000)
        row = {'id': f'fiction-{i}', 'group': f'fiction-family-{i}',
               'text': ' '.join(['floral'] * floral_words + ['woody'] * (11 - floral_words)),
               'formula_ppm': {'MAT_00000001': amount, 'MAT_00000002': 1000000 - amount},
               'ratings': [share, 1 - share]}
        data['validation' if i % 3 == 0 else 'train'].append(row)
    return data


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--data', type=Path)
    args = parser.parse_args()
    try:
        data = smoke_data() if args.smoke else json.loads(
            args.data.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
        print(json.dumps(evaluate(data), indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'Input/experiment error: {exc}\n')


if __name__ == '__main__':
    main()
