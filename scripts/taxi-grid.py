import argparse
import json
import os
import pickle
import re
from datetime import datetime

import numpy as np
import geopandas as gpd
import pandas as pd
from pyproj import Transformer, CRS
from shapely.geometry import shape, Point, box
from shapely.ops import transform

parser = argparse.ArgumentParser()
parser.add_argument('--od_data_dir', type=str, default='/home/buaa/data/taxi_OD/03', help='')
parser.add_argument('--shenzhen_geojson', type=str, default='data/shenzhen.geojson', help='')
parser.add_argument('--od_path', type=str, default='data/ods.pickle', help='')
parser.add_argument('--cells_path', type=str, default='data/cells.pickle', help='')
parser.add_argument('--grid_size', type=int, default=50, help='square grid size in meters.')

args = parser.parse_args()

start_date, end_date = datetime(2019, 3, 1), datetime(2019, 4, 1)


def transform_coordinate_system(geoms, source_cs='EPSG:4326', target_cs='EPSG:3857'):
    project = Transformer.from_crs(CRS(source_cs), CRS(target_cs), always_xy=True).transform

    if isinstance(geoms, (list, tuple)):
        return [transform(project, geom) for geom in geoms]
    else:
        return transform(project, geoms)


def get_bins(shenzhen, _size, crs='EPSG:4326'):
    x_min, y_min, x_max, y_max = shenzhen.bounds

    lt, rb = Point(x_min, y_min), Point(x_max, y_max)
    lt, rb = transform_coordinate_system(lt), transform_coordinate_system(rb)

    x_min_m, y_min_m, x_max_m, y_max_m = lt.x, lt.y, rb.x, rb.y

    nx, ny = (x_max_m - x_min_m + _size - 1e-5) // _size, (y_max_m - y_min_m + _size - 1e-5) // _size
    span_x, span_y = (x_max - x_min) / nx, (y_max - y_min) / ny

    bins = list()
    for x0 in np.arange(x_min, x_max + span_x, span_x):
        for y0 in np.arange(y_min, y_max + span_y, span_y):
            bins.append(box(x0, y0, x0 + span_x, y0 + span_y))
    return gpd.GeoDataFrame(bins, columns=['geometry'], crs=crs)


def transform_data(_data: gpd.GeoDataFrame, bins: gpd.GeoDataFrame):
    return gpd.sjoin(_data, bins, how='left', op='within', lsuffix='point', rsuffix='grid')


def get_data(_od_data_dir, _shenzhen_geojson, grid_size, crs='EPSG:4326'):
    print('Load Shenzhen geometry.')
    shenzhen = shape(json.load(open(_shenzhen_geojson)).get('features')[0].get('geometry'))

    print('Load OD data.')
    ods = list()
    for name in sorted(filter(lambda n: re.fullmatch(r'[0-9][0-9].parquet', n), os.listdir(_od_data_dir))):
        ods.append(pd.read_parquet(os.path.join(_od_data_dir, name)))
    ods = pd.concat(ods, axis=0, ignore_index=True)

    print('Transform OD data to GeoDataFrame')
    src, dst = ods[['stime', 'slng', 'slat']], ods[['etime', 'elng', 'elat']]
    src = gpd.GeoDataFrame(src[['stime']], geometry=gpd.points_from_xy(src.slng, src.slat), crs=crs)
    dst = gpd.GeoDataFrame(dst[['etime']], geometry=gpd.points_from_xy(dst.elng, dst.elat), crs=crs)

    print('Filter OD data by locations.')
    src, dst = gpd.clip(src, shenzhen), gpd.clip(dst, shenzhen)

    print('Filter OD data by timestamps.')
    src = src[(src.stime >= start_date) & (src.stime < end_date)]
    dst = dst[(dst.etime >= start_date) & (dst.etime < end_date)]

    print('Get Griding cells')
    bins = get_bins(shenzhen, grid_size)

    print('Gridding of OD points.')
    src, dst = transform_data(src, bins), transform_data(dst, bins)

    print('Final mergence of OD data.')
    ods = pd.merge(src, dst, left_index=True, right_index=True, suffixes=('_s', '_d'))

    print('Done')
    return ods, bins


if __name__ == '__main__':
    data, cells = get_data(args.od_data_dir, args.shenzhen_geojson, args.grid_size)

    pickle.dump(data, open(args.od_path, 'wb+'))
    pickle.dump(cells, open(args.cells_path, 'wb+'))
