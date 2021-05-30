from datetime import timedelta
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

def hour_rounder(t):
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               + timedelta(hours=t.minute//30))

def get_data(path, target=True, features=[]):

    traffic = pd.read_csv('data/raw/traffic.csv', usecols=[0, 1, 2, 4, 5, 9, 10, 11], parse_dates=['datetime'])
    traffic['datetime'] = traffic['datetime'].map(lambda x: hour_rounder(x))
    
    repair = pd.read_csv('data/raw/repair.csv', parse_dates=['datetime'])
    repair['year'] = repair['datetime'].map(lambda x: x.year)

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
        data = data.drop(data[condition2].index)
    else:
        condition1 = (data['target'].isnull()) & (data['data_id'].isnull())
    
    data = data.drop(data[condition1].index)

    data['year'] = data['datetime'].map(lambda x: x.year)
    index_repair = data[data.set_index(['road_km','year', 'road_id']).index.isin(repair.set_index(['road_km', 'year', 'road_id']).index)].index
    data.loc[index_repair, 'repair'] = 1
    data = data.fillna(0)

    hours = data['datetime'].map(lambda x: x.hour)

    data['night'] = ((hours > 21) | (hours < 6)).astype(int)
    data['month'] = data['datetime'].map(lambda x: x.month)
    data['weekday'] = data['datetime'].map(lambda x: x.weekday())

    if target:
        columns = features + ['datetime', 'road_km', 'target']
        data = data[columns]
        data.reset_index(drop=True, inplace=True)

    return data