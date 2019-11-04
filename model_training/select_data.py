import pandas as pd
import sys
import os
import glob


def filter_data(ch, y_rw, angle):
    return ch[(ch['angle'] == angle) & (ch['y_rw'] < y_rw + 0.1) & (ch['y_rw'] > y_rw - 0.1)]


in_path = sys.argv[1]
f_paths = glob.glob(os.path.join(in_path, '*.csv'))
new_dir = os.path.join(in_path, 'sel_csv')
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

angle = int(sys.argv[2])
y_rw = float(sys.argv[3])

for path in f_paths:
    iter_csv = pd.read_csv(path, iterator=True, chunksize=100000)
    df = pd.concat([filter_data(chunk, y_rw, angle) for chunk in iter_csv])
    f_name = os.path.split(path)[1].split('.')[0]
    df.to_csv('{}_{}_{}.csv'.format(os.path.join(new_dir, f_name), angle, y_rw))
