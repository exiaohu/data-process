import argparse
import pickle
from typing import Callable, Tuple

import pandas as pd
from tqdm import tqdm

from utils import create_kdtree

parser = argparse.ArgumentParser()
parser.add_argument('--od_path', type=str, default='data/ods.pickle', help='')
parser.add_argument('--cells_path', type=str, default='data/cells.pickle', help='')
parser.add_argument('--delta_threshold', type=float, default=0.00001, help='')
parser.add_argument('--rho_threshold', type=float, default=500, help='')
parser.add_argument('--save_path', type=str, default='data/cluster.parquet', help='')

args = parser.parse_args()


class DensityPeakCluster:

    def __init__(
            self,
            rho: pd.Series,
            index_to_coord: Callable[[int], Tuple[float, float]],
            coord_to_index: Callable[[Tuple[float, float]], int]
    ):
        """
        load data from custom grid data
        :param rho: input data,
                a pd.Series, with indexes as unit_nos and values as local density
        :param index_to_coord: function, get unit_no and return a (float, float) tuple as its coordinate.
        :param coord_to_index: function, get a (float, float) tuple as its coordinate and return its unit_no.
        """
        self.data = pd.DataFrame({'rho': rho}).sort_values(by='rho', ascending=False)
        self.data['index'] = self.data.index
        self.__calculate_delta_nneigh(index_to_coord, coord_to_index)

    def __calculate_delta_nneigh(self, index_to_coord, coord_to_index):
        """
        calculate delta from rho, which is the minimal distance from a point to another point that has a larger rho
        :param index_to_coord: function, get unit_no and return a (float, float) tuple as its coordinate.
        :param coord_to_index: function, get a (float, float) tuple as its coordinate and return its unit_no.
        :return: nothing
        """
        print("Calculating min distance BEGIN")
        print('node num is {}'.format(len(self.data['rho'])))
        coords = list(map(lambda unit_no: index_to_coord(unit_no), self.data.index.values))

        tree, idx, dist = create_kdtree(dimensions=2), [coord_to_index(coords[0])], [float('inf')]

        tree.add(coords[0])
        for item in tqdm(coords[1:]):
            i, d = tree.search_nn(item)
            i = i.data
            idx.append(coord_to_index(i))
            dist.append(d)
            tree.add(item)

        self.data['nneigh'] = idx
        self.data['delta'] = dist
        print("Calculating min distance END")

    def clustering(self, rho_threshold, delta_threshold):
        """
        choosing cluster centers and calculating the cluster every points belong
        :param rho_threshold: choosing points whose rho larger than or equal to the threshold
        :param delta_threshold: choosing points whose delta larger than or equal to the threshold
        :return: nothing
        """
        print("Clustering BEGIN")
        self.data['belong'] = self.data.apply(
            lambda row: row['index'] if row['rho'] >= rho_threshold and row['delta'] >= delta_threshold else -1,
            axis=1
        ).astype(int)

        total_count = len(self.data)
        assigned_count = len(self.data[self.data['belong'] >= 0])

        if assigned_count == 0:
            raise ValueError('rho threshold or delta threshold is too big, cannot find any cluster center.')

        print("Calculating %d cluster centers." % assigned_count)

        def cal_belong(row):
            if row['belong'] >= 0:
                return row['belong']
            return self.data.at[int(row['nneigh']), 'belong']

        pre_assigned_count = 0
        while assigned_count < total_count:
            print("calculating times %d of %d" % (assigned_count, total_count))
            if assigned_count == pre_assigned_count:
                raise ValueError("Cannot assign more points to any cluster.")
            pre_assigned_count = assigned_count
            self.data['belong'] = self.data.apply(cal_belong, axis=1).astype(int)
            assigned_count = len(self.data[self.data['belong'] >= 0])
        print("Clustering END")


if __name__ == '__main__':
    ods, cells = pickle.load(open(args.od_path, 'rb')), pickle.load(open(args.cells_path, 'rb'))
    data = ods.index_grid_d.value_counts().add(ods.index_grid_s.value_counts(), fill_value=0.0)
    cells_lookup = cells.geometry.transform(lambda i: (i.centroid.x, i.centroid.y))

    i2c = cells_lookup.to_dict()
    c2i = {v: k for k, v in i2c.items()}

    dpc = DensityPeakCluster(data, lambda i: i2c[i], lambda c: c2i[c])
    dpc.clustering(args.rho_threshold, args.delta_threshold)
    dpc.data.to_parquet(args.save_path, compression='gzip')
