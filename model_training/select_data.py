import pandas as pd
import sys


def filter_data(ch, y_rw, angle):
    return ch[(ch['angle'] == y_rw) & (ch['y_rw'] < angle + 0.1) & (ch['y_rw'] > angle - 0.1)]


nm = sys.argv[1]
iter_csv = pd.read_csv(nm, iterator=True, chunksize=100000)

angle = -10
y_rw = -3.2

df = pd.concat([filter_data(chunk, y_rw, angle) for chunk in iter_csv])

df.to_csv('{}_{}_{}.csv'.format(angle, y_rw, nm))
