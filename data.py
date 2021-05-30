from datetime import timedelta
import pandas as pd
import numpy as np

def hour_rounder(t):
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               + timedelta(hours=t.minute//30))

def get_data(path, target=True):
    traffic_cols = [
        'datetime',
        'road_id',
        'road_km',
        'data_id',
        'lane_count',
        'volume',
        'occupancy',
        'speed'
    ]

    traffic = pd.read_csv('data/raw/traffic.csv', usecols=traffic_cols, parse_dates=['datetime'])
    traffic['datetime'] = traffic['datetime'].map(lambda x: hour_rounder(x))
    
    repair = pd.read_csv('data/raw/repair.csv', parse_dates=['datetime'])
    repair['year'] = repair['datetime'].map(lambda x: x.year)

    road_ints = pd.read_csv('data/custom/road_intervals.csv')
    road_5km_ints = pd.read_csv('data/custom/road_intervals_5km.csv')

    if target:
        df = pd.read_csv(path, usecols=[0, 1, 2, 9, 10], parse_dates=['datetime'])
        df = df.sort_values('datetime', ascending=True)
        df.reset_index(drop=True, inplace=True)

        dfc = df[['datetime', 'road_km', 'target', 'road_id']]
    else:
        dfc = pd.read_csv(path, parse_dates=['datetime'])

    data = pd.merge(dfc, traffic, how='outer', on=['datetime', 'road_km', 'road_id'])

    if target:
        condition1 = (data['target'].isnull()) & (data['data_id'].isnull())
        condition2 = ~(data['target'].isnull()) & (data['data_id'].isnull())
        data = data.drop(data[condition1].index)
        data = data.drop(data[condition2].index)
    else:
        condition1 = (data['target'].isnull()) & (data['data_id'].isnull())
        data = data.drop(data[condition1].index)

    data['year'] = data['datetime'].map(lambda x: x.year)
    index_repair = data[data.set_index(['road_km','year', 'road_id']).index.isin(repair.set_index(['road_km', 'year', 'road_id']).index)].index
    data.loc[index_repair, 'repair'] = 1
    data = data.fillna(0)
    data['repair'] = data['repair'].astype(int)

    data['road_km_int'] = data['road_km'] // 20 * 20
    data = pd.merge(data, road_ints, how='left', on=['road_id', 'road_km_int'])

    data['road_5km_int'] = data['road_km'] // 5 * 5

    for i in range(10):
        data[f'top{i+1}_interval'] = 0

    for road_id in data['road_id'].unique():
        for i in range(10):
            interval = road_5km_ints[road_5km_ints['road_id'] == road_id]['road_5km_int'].iloc[i]
            data.loc[(data['road_id'] == road_id) & (data['road_5km_int'] == interval), f'top{i+1}_interval'] = 1

    hours = data['datetime'].map(lambda x: x.hour)

    data['night'] = ((hours > 21) | (hours < 6)).astype(int)
    data['month'] = data['datetime'].map(lambda x: x.month)
    data['weekday'] = data['datetime'].map(lambda x: x.weekday())


    drop_cols = [
        'data_id',
        'year',
        'road_km_int'
    ]

    data.drop(drop_cols, axis=1, inplace=True)

    if target:
        data.reset_index(drop=True, inplace=True)

    return data