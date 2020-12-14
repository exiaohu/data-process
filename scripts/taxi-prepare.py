import argparse
import os
import zipfile
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/buaa/data/taxi_GPS/03', help='')
parser.add_argument('--save_dir', type=str, default='/home/buaa/data/taxi_OD/03', help='')

args = parser.parse_args()

columns = ['fdate', 'ftime', 'busline_name', 'vehicle_id', 'lng', 'lat', 'speed', 'angle', 'operation_status',
           'company_code']
use_cols = ['fdate', 'ftime', 'vehicle_id', 'lng', 'lat', 'operation_status']
dtypes = {
    'vehicle_id': str,
    'lng': np.float32,
    'lat': np.float32,
    'speed': np.float32,
    'angle': np.float32,
    'operation_status': np.int8
}


def get_data(ddir: zipfile.ZipFile):
    _data = list()
    for info in tqdm(ddir.filelist, desc=f'collecting... {fullname}'):
        if info.is_dir():
            continue

        with ddir.open(info) as file:
            datum = pd.read_csv(
                file,
                header=None,
                names=columns,
                usecols=use_cols,
                parse_dates={'gen_dt': ['fdate', 'ftime']},
                dtype=dtypes
            )
            if len(datum) > 0:
                datum.dropna(axis=0, how='any', inplace=True)

                datum['gen_dt'] = pd.to_datetime(datum.gen_dt, format='%Y%m%d %H%M%S', errors='coerce')
                datum = datum[~pd.isnull(datum.gen_dt)]

                _data.append(datum)

    _data = pd.concat(_data, axis=0, ignore_index=True)
    _data.sort_values('gen_dt', inplace=True)
    return _data


def parse_traj(traj: pd.DataFrame):
    records = list()

    cur_s, cur_e, prev_status = None, None, None
    for record in traj.itertuples():
        gen_dt, lng, lat, status = record.gen_dt, record.lng, record.lat, record.operation_status
        if prev_status == 0 and status == 1:
            cur_s = (gen_dt, lng, lat)
        elif prev_status == 1 and status == 0 and cur_s is not None:
            cur_e = (gen_dt, lng, lat)
            if timedelta(days=1) > (cur_e[0] - cur_s[0]) > timedelta(minutes=5):
                records.append(cur_s + cur_e)

            cur_s, cur_e = None, None
        prev_status = status

    return pd.DataFrame(records, columns=['stime', 'slng', 'slat', 'etime', 'elng', 'elat'])


if __name__ == '__main__':
    tqdm.pandas()
    for name in sorted(filter(lambda n: zipfile.is_zipfile(os.path.join(args.data_dir, n)), os.listdir(args.data_dir))):
        try:
            fullname = os.path.join(args.data_dir, name)

            data = get_data(zipfile.ZipFile(fullname))

            print('data parsing...', fullname)

            data = data.groupby(['vehicle_id']).progress_apply(lambda traj: parse_traj(traj))

            data.to_parquet(os.path.join(args.save_dir, name.replace('zip', 'parquet')), compression='gzip')
            data.info()
        except ValueError:
            print('error occured while parsing', name)
