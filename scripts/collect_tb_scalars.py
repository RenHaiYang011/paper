"""
Collect TensorBoard scalars from runs (uses TensorBoard EventAccumulator).
Produces a CSV with columns: run_id, scalar_tag, step, value

Usage: python scripts/collect_tb_scalars.py --runs_csv runs/runs.csv --out runs/scalars.csv
"""
import os
import csv
import argparse
try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception as e:
    print('tensorboard not available in this environment. Install tensorboard to use this script. Error:', e)
    event_accumulator = None

parser = argparse.ArgumentParser()
parser.add_argument('--runs_csv', type=str, default='runs/runs.csv')
parser.add_argument('--out', type=str, default='runs/scalars.csv')
args = parser.parse_args()

rows = []
with open(args.runs_csv, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        run_id = r['run_id']
        logdir = r['logdir']
        if not os.path.exists(logdir):
            print('Logdir not found for', run_id, logdir)
            continue
        if event_accumulator is None:
            print('Skipping', run_id, 'because tensorboard.event_accumulator is unavailable')
            continue
        ea = event_accumulator.EventAccumulator(logdir)
        try:
            ea.Reload()
        except Exception as e:
            print('Failed to read event files in', logdir, e)
            continue
        tags = ea.Tags().get('scalars', [])
        for tag in tags:
            events = ea.Scalars(tag)
            for ev in events:
                rows.append([run_id, tag, ev.step, ev.value])

with open(args.out, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['run_id','tag','step','value'])
    for r in rows:
        writer.writerow(r)

print('Wrote', args.out)
