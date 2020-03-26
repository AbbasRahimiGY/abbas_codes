from datetime import datetime
import pandas as pd
import os

os.chdir(r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast')
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype as is_num

df = pd.read_excel(r'Light Vehicle Production Forecast 03-20-2020 05_35 PM.xlsx')
df.columns = df.columns.str.strip().str.upper().str.replace('VP: ', '').str.replace(' ', '_'). \
    str.replace('-', '_').str.replace('___', '_')
months = [name for name in df.columns if is_num(df[name])]
months = [name for name in months if 'CY_' not in name]
months = [name for name in months if '20' in name]
index = [datetime.strptime(date, '%b_%Y') for date in months]
df['MODEL-VERSION'] = df.VEHICLE.apply(lambda x: "_".join([item.upper() for item in x.split(':')[1:3]]) + str('_') +
                                                 x.split(':')[3].split(' ')[0])
df1 = df.copy()
other_col = ['SALES_GROUP', 'PRODUCTION_NAMEPLATE',
             'GLOBAL_PRODUCTION_PRICE_CLASS', 'REGIONAL_SALES_SUB_SEGMENT', 'MODEL-VERSION']
columns = months + other_col
df1 = df1.groupby(other_col)[months].sum().T
df1.index = index
df1.index.rename('PR_DATE', inplace=True)
if 'Fiat' in df1.columns:
    df1['Chrysler'] = df1[['Chrysler', 'Fiat']].sum(axis=1)
df1.rename(columns={'General Motors': 'GM', 'Subaru': 'SIA', 'Volkswagen': 'VW', 'Hyundai': 'HYUNDAI'}, inplace=True)
df1 = df1.stack(level=[i for i in range(len(other_col))]).reset_index().rename(columns={0: 'Volume'}).fillna(0)
df1.rename(columns={'SALES_GROUP': 'MAKE', 'PRODUCTION_NAMEPLATE': 'MODEL',
                    'GLOBAL_PRODUCTION_PRICE_CLASS': 'TIER',
                    'REGIONAL_SALES_SUB_SEGMENT': 'CATEGORY'}, inplace=True)
df1['MODEL'] = df1[['MAKE', 'MODEL']].agg('-'.join, axis=1)
df1.to_excel(r'\\AKRTABLEAUPNA01\Americas_Market_Analytics$\IHS_vehicle_production.xlsx')
