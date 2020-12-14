import argparse
import pickle

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--od_path', type=str, default='data/ods.pickle', help='')
parser.add_argument('--cluster_path', type=str, default='data/cluster.parquet', help='')
parser.add_argument('--save_path', type=str, default='data/demands.parquet', help='')
parser.add_argument('--freq', type=str, default='15min', help='')

args = parser.parse_args()

if __name__ == '__main__':
    ods, clusters = pickle.load(open(args.od_path, 'rb')), pd.read_parquet(args.cluster_path)

    src, dst = ods[['stime', 'index_grid_s']], ods[['etime', 'index_grid_d']]
    index2cluster = {index: cluster for index, cluster in zip(clusters.index, clusters.belong)}
    src['sunit'] = src.index_grid_s.transform(lambda index: index2cluster[index])
    dst['eunit'] = dst.index_grid_d.transform(lambda index: index2cluster[index])

    src = src.groupby([pd.Grouper(key='stime', freq=args.freq), 'sunit']).count().index_grid_s
    dst = dst.groupby([pd.Grouper(key='etime', freq=args.freq), 'eunit']).count().index_grid_d

    src.index.rename(['time', 'unit'], inplace=True)
    dst.index.rename(['time', 'unit'], inplace=True)

    pd.DataFrame({'src': src, 'dst': dst}).to_parquet(args.save_path, compression='gzip')
