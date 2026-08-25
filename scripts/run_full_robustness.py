import json
import os
import sys
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'src'))

os.chdir(ROOT)

from mastercard_defence.loop import ClosedLoop, load_config
from adaptive.generate_family_charts import build_charts


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


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main(seeds: int = 1) -> None:
    config = load_config('config/default.yaml')
    run_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    artifacts_path = Path(config['paths']['artifacts'])
    config['paths']['memory_db'] = str(artifacts_path / f'attack_memory_full_{run_stamp}.sqlite')
    log_path = artifacts_path / f'adaptive_run_{run_stamp}.log'
    artifacts_path.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log_file, redirect_stdout(Tee(sys.__stdout__, log_file)), redirect_stderr(Tee(sys.__stderr__, log_file)):
        print(f'LOG_SAVED {log_path}')
        print(f'RUN_CONFIGURATION seeds={seeds} rounds={max(50, config["pipeline"]["rounds"])}')
        loop = ClosedLoop(config)
        try:
            suite = loop.run_robustness_suite(seeds=seeds, rounds=max(50, config['pipeline']['rounds']))
            artifact = {
                'run_timestamp_utc': run_stamp,
                'seed_count': suite['seed_count'],
                'rounds': suite['rounds'],
                'families_per_run': suite['families_per_run'],
                'summary': suite['summary'],
                'family_analysis': build_family_analysis(suite),
                'round_metrics': [
                    {
                        'seed': run['seed'],
                        'round': result['round'],
                        'attack_family': result['specification'].attack_family,
                        'family_decision': result['family_decision'],
                        'all_family_metrics': result['detection'].get('all_family_metrics', {}),
                    }
                    for run in suite['by_seed']
                    for result in run['results']
                ],
                'by_seed': suite['by_seed'],
            }
            artifact_path = artifacts_path / f'robustness_results_{run_stamp}.json'
            artifact_path.write_text(json.dumps(to_jsonable(artifact), indent=2), encoding='utf-8')
            build_charts(artifact_path, Path(ROOT) / 'adaptive' / 'charts')
            print(json.dumps({key: artifact[key] for key in ('run_timestamp_utc', 'seed_count', 'rounds', 'families_per_run', 'summary')}, indent=2))
            print(f'RESULTS_SAVED {artifact_path}')
            print(f'LOG_SAVED {log_path}')
        finally:
            loop.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the adaptive robustness experiment.')
    parser.add_argument('--seeds', type=int, default=1)
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError('--seeds must be at least 1')
    main(seeds=args.seeds)
