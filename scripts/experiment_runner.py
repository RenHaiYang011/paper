"""
Small experiment runner:
- Generates configs for a grid of coverage_weight / distance_weight / n_agents
- For each combination, creates a copy of params_test.yaml with overrides and a run directory
- Launches training sequentially and writes a `runs.csv` with metadata (seed, params, logdir)

This script is intentionally simple and synchronous to avoid background-process management complexity.
"""
import os
import yaml
import shutil
import csv
import subprocess
import uuid
from datetime import datetime
import argparse

ROOT = os.path.dirname(os.path.dirname(__file__))
PARAMS_TEMPLATE = os.path.join(ROOT, 'marl_framework', 'params_test.yaml')
RUNS_DIR = os.path.join(ROOT, 'runs')
LOG_BASE = os.path.join(ROOT, 'marl_framework', 'log')

os.makedirs(RUNS_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--coverage', nargs='+', type=float, default=[0.0, 0.05])
parser.add_argument('--distance', nargs='+', type=float, default=[0.0, 0.01])
parser.add_argument('--agents', nargs='+', type=int, default=[2,4])
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--python', type=str, default='python')
parser.add_argument('--ldpreload', type=str, default='')
args = parser.parse_args()

combinations = []
for c in args.coverage:
    for d in args.distance:
        for a in args.agents:
            combinations.append((c,d,a))

runs_csv = os.path.join(RUNS_DIR, 'runs.csv')
if not os.path.exists(runs_csv):
    with open(runs_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id','timestamp','seed','coverage_weight','distance_weight','n_agents','config_path','logdir'])

for (c,d,a) in combinations:
    for r in range(args.repeats):
        seed = r + 1
        run_id = uuid.uuid4().hex[:8]
        timestamp = datetime.utcnow().isoformat()
        run_dir = os.path.join(RUNS_DIR, f'run_{run_id}')
        os.makedirs(run_dir, exist_ok=True)
        # load template
        with open(PARAMS_TEMPLATE, 'r') as f:
            params = yaml.safe_load(f)
        # override
        params['experiment']['missions']['n_agents'] = a
        params['experiment']['coverage_weight'] = float(c)
        params['experiment']['distance_weight'] = float(d)
        params['environment']['seed'] = int(seed)
        # write config
        cfg_path = os.path.join(run_dir, 'params.yaml')
        with open(cfg_path, 'w') as f:
            yaml.dump(params, f)
        # define logdir under marl_framework/log/run_<id>
        logdir = os.path.join(LOG_BASE, f'run_{run_id}')
        os.makedirs(logdir, exist_ok=True)
        # record run
        with open(runs_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([run_id, timestamp, seed, c, d, a, cfg_path, logdir])

        # build command
        cmd = []
        if args.ldpreload:
            cmd.append(args.ldpreload)
        cmd.extend([args.python, '-m', 'marl_framework.main', '--config', cfg_path])
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, cwd=ROOT)
        print('Finished run', run_id)

print('All runs finished. Runs metadata saved to', runs_csv)
