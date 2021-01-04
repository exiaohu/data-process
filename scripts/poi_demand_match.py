import argparse
import pickle

import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from utils.coord_transform import bd2wgs

parser = argparse.ArgumentParser()
parser.add_argument('--poi_path', type=str, default='/home/buaa/data/process-data/poi-data/baidu-poi.csv',
                    help='Baidu crs')
parser.add_argument('--od_path', type=str, default='data/ods.pickle', help='as WGS84座')
parser.add_argument('--threshold', type=float, default=1e-3, help='')
parser.add_argument('--save_path', type=str, default='data/poi_demand_match.pickle', help='')

args = parser.parse_args()

tqdm.pandas()
poi_search_trees = pd.read_csv(args.poi_path, usecols=['lng', 'lat', 'type']).groupby('type').progress_apply(
    lambda data: KDTree([bd2wgs(lat, lng)[::-1] for lng, lat in zip(data.lng, data.lat)]))

ods = pickle.load(open(args.od_path, 'rb'))
src, dst = ods[['geometry_s', 'stime']], ods[['geometry_d', 'etime']]
src_search_trees = src.groupby(src.stime.dt.hour).progress_apply(
    lambda d: KDTree([(p.x, p.y) for p in d.geometry_s]))
dst_search_trees = dst.groupby(dst.etime.dt.hour).progress_apply(
    lambda d: KDTree([(p.x, p.y) for p in d.geometry_d]))

data = dict()
for poi_type, search_tree in poi_search_trees.items():
    for hour, src_demand in tqdm(src_search_trees.items(), f'src {poi_type}'):
        values = list(map(len, search_tree.query_ball_tree(src_demand, args.threshold)))
        data[('src', poi_type, hour)] = sum(values) / len(values)
    for hour, dst_demand in tqdm(dst_search_trees.items(), f'dst {poi_type}'):
        values = list(map(len, search_tree.query_ball_tree(dst_demand, args.threshold)))
        data[('dst', poi_type, hour)] = sum(values) / len(values)

pickle.dump(data, open(args.save_path, 'wb+'))
