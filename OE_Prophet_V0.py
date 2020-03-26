import pandas as pd
from datetime import datetime, timedelta
import pyodbc
import os, csv
import numpy as np
from dateutil import relativedelta
from tqdm import tqdm
from fbprophet import Prophet
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed, parallel_backend
from random import sample
from OE_production import prophet_configs, grid_search
import logging
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


logging.getLogger('fbprophet').setLevel(logging.WARNING)

plt.style.use('fivethirtyeight')
os.chdir(r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast')
link = ('DSN=EDWTDPRD;UID=AA68383;PWD=baradarkhobvaghashang1364')
pyodbc.pooling = False


def direct_shipment_query():
    query = '''SELECT NA_BI_VWS.DELIVERY.ACTL_GOODS_ISS_DT AS SHIP_DT,
                                NA_BI_VWS.CUSTOMER.CUST_GRP_NAME AS CUSTOMER,
                           --  NA_BI_VWS.DELIVERY.DISTR_CHAN_CD AS DC,
                            --NA_BI_VWS.CUSTOMER.distr_chan_name AS DC,
                           --NA_BI_VWS.MATERIAL.MKT_CTGY_MKT_GRP_NAME AS CTGY,
                          --  NA_BI_VWS.MATERIAL.MKT_CTGY_PROD_GRP_NAME AS PRD_TIER,
                           -- NA_BI_VWS.MATERIAL.MKT_CTGY_PROD_LINE_NAME AS CTGY_LINE,
                            SUM(NA_BI_VWS.DELIVERY.RPT_DELIV_QTY) AS SHIPPED_QTY
                            FROM NA_BI_VWS.DELIVERY INNER JOIN NA_BI_VWS.CUSTOMER ON 
                            NA_BI_VWS.DELIVERY.SHIP_TO_CUST_ID = NA_BI_VWS.CUSTOMER.SHIP_TO_CUST_ID 
                            INNER JOIN NA_BI_VWS.MATERIAL ON NA_BI_VWS.DELIVERY.MATL_ID = NA_BI_VWS.MATERIAL.MATL_ID
                            WHERE NA_BI_VWS.DELIVERY.ACTL_GOODS_ISS_DT >= CAST('2014-01-01' AS DATE)
                                AND  NA_BI_VWS.MATERIAL.PBU_NBR = '01'
                                AND  NA_BI_VWS.MATERIAL.SUPER_BRAND_ID in ('01','02','03')
                                AND  NA_BI_VWS.CUSTOMER.CUST_HIER_GRP_2_DESC = 'OE'
                                AND NA_BI_VWS.CUSTOMER.SALES_ORG_CD IN('N302', 'N312')
                            GROUP BY 1,2
                            ORDER BY 1,2'''
    with pyodbc.connect(link, autocommit=True) as connect:
        df = pd.read_sql(query, connect)
    df = df.pivot_table('SHIPPED_QTY', index='SHIP_DT', columns='CUSTOMER',
                        aggfunc='sum', dropna=False, fill_value=0)

    df.index = pd.to_datetime(df.index)
    df = df.resample('D').sum()
    df1 = df.resample('y').sum()
    active_customers = [col for col in df1.columns if df1['2019':][col].sum() > 100] # check for non existing cust
    print('active customer are={}'.format(active_customers))
    df = df[active_customers]
    df = df['2014':]
    df[df < 0] = 0
    return df



def events_holidays_update(link):
    query = '''Select CAL_DT,HLDY_DESC 
                    From GDYR_BI_VWS.GDYR_HLDY_CURR                    
                    order By CAL_DT
                    where CAL_DT >= Cast('2014-01-01' As Date)'''
    with pyodbc.connect(link, autocommit=True) as connect:
        GY_Holiday = pd.read_sql(query, connect)
    GY_Holiday = GY_Holiday[GY_Holiday['HLDY_DESC'] != 'Sunday']
    GY_Holiday = pd.DataFrame({
        'holiday': GY_Holiday['HLDY_DESC'],
        'ds': pd.to_datetime(GY_Holiday['CAL_DT'])
    })
    GY_Holiday['holiday'] = GY_Holiday['holiday'].str.split(' - ').str.get(0)
    GY_Holiday['holiday'] = np.where(GY_Holiday['holiday'] == 'New Years Day', 'New Years', GY_Holiday['holiday'])
    GY_Holiday['holiday'] = np.where(GY_Holiday['holiday'] == 'Independence Day - N',
                                     'Independence Day', GY_Holiday['holiday'])
    GY_Holiday['holiday'] = np.where(GY_Holiday['holiday'] == 'Memorial Day - Natio',
                                     'Memorial Day', GY_Holiday['holiday'])

    APRIL = pd.DataFrame({
        'holiday': 'APRIL',
        'ds': pd.date_range("2014-4-15", "2024-4-15", freq='12M') + timedelta(days=-15)})

    STRIKE = pd.DataFrame({
        'holiday': 'STRIKE',
        'ds': "2019-10-1"}, index=[0])  # GM STRIKE

    GY_Holiday = pd.concat((GY_Holiday, STRIKE))
    GY_Holiday.reset_index(inplace=True)

    GY_Holiday = GY_Holiday[(GY_Holiday.holiday != 'Day After Thanksgivi') &
                            (GY_Holiday.holiday != 'Good Friday - Local') &
                            (GY_Holiday.holiday != 'Good Friday') &
                            (GY_Holiday.holiday != 'New Year Eve') &
                            (GY_Holiday.holiday != 'Christmas Eve')]

    GY_Holiday['upper_window'] = 0
    GY_Holiday['lower_window'] = 0
    GY_Holiday['upper_window'] = np.where(GY_Holiday['holiday'] == 'Christmas', 4, GY_Holiday['upper_window'])
    GY_Holiday['lower_window'] = np.where(GY_Holiday['holiday'] == 'Christmas', -2, GY_Holiday['lower_window'])
    GY_Holiday['upper_window'] = np.where(GY_Holiday['holiday'] == 'APRIL', 5, GY_Holiday['upper_window'])
    GY_Holiday['lower_window'] = np.where(GY_Holiday['holiday'] == 'APRIL', -5, GY_Holiday['lower_window'])
    GY_Holiday['upper_window'] = np.where(GY_Holiday['holiday'] == 'Memorial Day', 3, GY_Holiday['upper_window'])
    GY_Holiday['lower_window'] = np.where(GY_Holiday['holiday'] == 'Memorial Day', 0, GY_Holiday['lower_window'])
    GY_Holiday['upper_window'] = np.where(GY_Holiday['holiday'] == 'Labor Day', 1, GY_Holiday['upper_window'])
    GY_Holiday['lower_window'] = np.where(GY_Holiday['holiday'] == 'Labor Day', -1, GY_Holiday['lower_window'])
    GY_Holiday['upper_window'] = np.where(GY_Holiday['holiday'] == 'STRIKE', 25, GY_Holiday['upper_window'])
    GY_Holiday['lower_window'] = np.where(GY_Holiday['holiday'] == 'STRIKE', -15, GY_Holiday['lower_window'])
    GY_Holiday['upper_window'] = np.where(GY_Holiday['holiday'] == 'New Years', 1, GY_Holiday['upper_window'])
    GY_Holiday['lower_window'] = np.where(GY_Holiday['holiday'] == 'New Years', 0, GY_Holiday['lower_window'])

    return GY_Holiday


def median_filter(df, varname=None, window=30, std=3):
    dfc = df.loc[:, [varname]]
    dfc['median'] = dfc[varname].rolling(window, center=True).median()
    dfc['std'] = dfc[varname].rolling(window, center=True).std()
    dfc.loc[dfc.loc[:, varname] >= dfc['median'] + std * dfc['std'], varname] = dfc['median'].median()
    dfc.loc[dfc.loc[:, varname] <= dfc['median'] - std * dfc['std'], varname] = dfc['median'].median()
    return pd.DataFrame(dfc.loc[:, varname])


if __name__ == '__main__':
    # df_SOP=SOP_query()
    delivery = direct_shipment_query()

    holidays = events_holidays_update(link)
    cfg_list = prophet_configs() # configure the combo
    cfg_list = sample(sample(cfg_list, 10000) ,1000) # set the number of sub sample
    subtype = None
    # remain_search=[col for col in delivery.columns if col not in search]
    if subtype == 'CD':
        for CD in delivery.columns:  # delivery.columns:

            data = delivery[CD]

            # grid search
            scores = grid_search(data, holidays, cfg_list)
            print('done')

            # list top 3 configs
            for cfg, error in scores[:3]:
                print(cfg, error)

            with open('{}_prophet_param.csv'.format(CD), 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                for tup in scores:
                    writer.writerow(tup)
    else:
        print(len(cfg_list))
        data = pd.DataFrame(delivery.sum(axis=1), columns=['SHIPPED_QTY'])
       # data = median_filter(data, varname='SHIPPED_QTY', window=91, std=2)
        data= data['SHIPPED_QTY']
        scores = grid_search(data, holidays, cfg_list)
        print('done')

        # list top 3 configs
        for cfg, error in scores[:3]:
            print(cfg, error)

        with open('total_daily_prophet_param.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for tup in scores:
                writer.writerow(tup)

