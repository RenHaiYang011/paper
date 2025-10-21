"""
Plot scalar results grouped by run_id (expects output from collect_tb_scalars.py)
Produces a PNG per tag under runs/plots/
"""
import os
import csv
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--scalars_csv', type=str, default='runs/scalars.csv')
parser.add_argument('--out_dir', type=str, default='runs/plots')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

data = defaultdict(list)
with open(args.scalars_csv, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        tag = r['tag']
        step = int(r['step'])
        value = float(r['value'])
        run_id = r['run_id']
        data[tag].append((run_id, step, value))

for tag, rows in data.items():
    # group by run_id
    by_run = defaultdict(list)
    for run_id, step, value in rows:
        by_run[run_id].append((step, value))
    plt.figure()
    for run_id, vals in by_run.items():
        vals_sorted = sorted(vals, key=lambda x: x[0])
        steps = [v[0] for v in vals_sorted]
        values = [v[1] for v in vals_sorted]
        plt.plot(steps, values, label=run_id)
    plt.title(tag)
    plt.xlabel('step')
    plt.ylabel('value')
    plt.legend(fontsize='small')
    outpath = os.path.join(args.out_dir, tag.replace('/', '_') + '.png')
    plt.savefig(outpath, dpi=200)
    plt.close()
    print('Wrote', outpath)
