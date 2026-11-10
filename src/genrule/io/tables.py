"""Read cached paper inputs without experiment-store or plotting dependencies."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def read_image_clock_data(directory):
    directory = Path(directory)
    with (directory/'image_onsets.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    with (directory/'image_trajectories.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            grouped[(row['run'],row['metric'])].append((float(row['step']),float(row['value'])))
    curves = {key:(np.array([v[0] for v in pairs]), np.array([v[1] for v in pairs]))
              for key,pairs in grouped.items()}
    return rows, curves
