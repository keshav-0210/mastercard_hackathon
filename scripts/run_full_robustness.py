import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'src'))

os.chdir(ROOT)

from mastercard_defence.loop import ClosedLoop, load_config


def to_jsonable(value):
    if hasattr(value, 'model_dump'):
        return to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_family_analysis(suite: dict) -> list[dict]:
    rows = []
    for run in suite['by_seed']:
        for result in run['results']:
            for family, metrics in result['detection'].get('by_attack_family', {}).items():
                rows.append({
                    'seed': run['seed'],
                    'round': result['round'],
                    'attack_family': family,
                    **metrics,
                })
    return rows


def main() -> None:
    config = load_config('config/default.yaml')
    run_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    artifacts_path = Path(config['paths']['artifacts'])
    config['paths']['memory_db'] = str(artifacts_path / f'attack_memory_full_{run_stamp}.sqlite')
    loop = ClosedLoop(config)
    try:
        suite = loop.run_robustness_suite(seeds=3, rounds=5)
        artifact = {
            'run_timestamp_utc': run_stamp,
            'seed_count': suite['seed_count'],
            'rounds': suite['rounds'],
            'families_per_run': suite['families_per_run'],
            'summary': suite['summary'],
            'family_analysis': build_family_analysis(suite),
            'by_seed': suite['by_seed'],
        }
        artifact_path = artifacts_path / f'robustness_results_{run_stamp}.json'
        artifact_path.write_text(json.dumps(to_jsonable(artifact), indent=2), encoding='utf-8')
        print(json.dumps({key: artifact[key] for key in ('run_timestamp_utc', 'seed_count', 'rounds', 'families_per_run', 'summary')}, indent=2))
        print(f'RESULTS_SAVED {artifact_path}')
    finally:
        loop.close()


if __name__ == '__main__':
    main()
