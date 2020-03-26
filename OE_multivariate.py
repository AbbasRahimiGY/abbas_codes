import pandas as pd
from datetime import datetime, timedelta
import pyodbc
import numpy as np
import os
import matplotlib.pyplot as plt
from dateutil import relativedelta
from tqdm import tqdm
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot  # locations of the signification changepoints
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from pmdarima import auto_arima
from calendar import monthrange
import logging, warnings
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
warnings.filterwarnings('ignore')

logging.getLogger('fbprophet').setLevel(logging.WARNING)

plt.style.use('fivethirtyeight')

os.chdir(r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast')
# Make a connection
link = ('DSN=EDWTDPRD;UID=AA68383;PWD=baradarkhobvaghashang1364')
pyodbc.pooling = False

def direct_shipment_query():
    query = '''SELECT  cast(CAST(NA_BI_VWS.DELIVERY.ACTL_GOODS_ISS_DT AS DATE FORMAT 'YYYY-MM-DD') AS CHAR(10))  AS SHIP_DT,
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
            WHERE CAST(NA_BI_VWS.DELIVERY.ACTL_GOODS_ISS_DT AS DATE FORMAT 'YYYY-MM-DD') >= CAST('2014-01-01' AS DATE)
            AND  NA_BI_VWS.MATERIAL.PBU_NBR = '01'
            AND  NA_BI_VWS.MATERIAL.SUPER_BRAND_ID in ('01','02','03')
            AND  NA_BI_VWS.CUSTOMER.CUST_HIER_GRP_2_DESC = 'OE'
            AND 
            (NA_BI_VWS.CUSTOMER.SALES_ORG_CD IN('N302', 'N312')
            OR  NA_BI_VWS.CUSTOMER.DISTR_CHAN_CD = '32' )
            GROUP BY 1,2
            ORDER BY 1,2;'''

    with pyodbc.connect(link, autocommit=True) as connect:
        df = pd.read_sql(query, connect)
    df['SHIP_DT']= pd.to_datetime(df['SHIP_DT'])

    _ = df.pivot_table('SHIPPED_QTY', index='SHIP_DT', columns='CUSTOMER',
                        aggfunc='sum', dropna=False, fill_value=0).resample('Y').sum()

    active_customers=[col for col in _.columns if _['2019':][col].sum()>100]
    df=df.loc[df.CUSTOMER.isin (active_customers),:]
    return df


def order_query(DATE):
    cnxn = pyodbc.connect(link)
    query = '''SELECT ODS.SNAP_DT AS SNAP_DT,
    M.PROD_LINE_NBR,
    M.MKT_CTGY_PROD_LINE_NBR,
    ODS.DELIV_BLOCK_CD,
    ODS.PLN_SHIP_DT AS DEL_DT,
    M.MKT_CTGY_MKT_AREA_NBR,
    M.MKT_CTGY_MKT_AREA_NAME,
    SUM(ODS.CNFRM_QTY) AS CNFRM_QTY,
    SUM(ODS.INPROC_QTY) AS INPROC_QTY,
    SUM(ODS.CANCEL_QTY) AS CANCEL_QTY,
    SUM(ODS.ORDER_QTY) AS ORDER_QTY,
    SUM(ODS.UNCNFRM_QTY) AS UNCNFRM_QTY,
    SUM(ODS.PAST_DUE_CNFRM_QTY) AS PAST_DUE_CNFRM_QTY,
    SUM(ODS.BACKORDER_QTY) AS BACKORDER_QTY,
    SUM(ODS.DEFER_QTY) AS DEFER_QTY,
    SUM(ODS.WAITLIST_QTY) AS WAITLIST_QTY,
    SUM(ODS.OTHR_OPEN_ORDER_QTY) AS OTHR_OPEN_ORDER_QTY
FROM NA_BI_VWS.ORDER_DELIVERY_DETAIL_SNAP ODS
    INNER JOIN NA_BI_VWS.MATERIAL M
    ON ODS.MATL_ID = M.MATL_ID
    INNER JOIN NA_BI_VWS.CUSTOMER C
    ON ODS.SHIP_TO_CUST_ID = C.SHIP_TO_CUST_ID
    AND ODS.CO_CD = C.CO_CD
    WHERE ODS.SNAP_DT = '{DATE}'
        AND  ODS.OPEN_ORDER_IND = 'Y'
        AND  M.PBU_NBR = '01'
        AND  C.CUST_HIER_GRP_2_DESC = 'OE'
        AND M.SUPER_BRAND_ID IN('01', '02', '03')
      	AND(C.SALES_ORG_CD IN('N302', 'N312')
        OR C.DISTR_CHAN_CD = '32')
    GROUP BY ODS.SNAP_DT,
        M.PROD_LINE_NBR,
        M.MKT_CTGY_PROD_LINE_NBR,
        ODS.DELIV_BLOCK_CD,
        ODS.PLN_SHIP_DT,
        M.MKT_CTGY_MKT_AREA_NBR,
        M.MKT_CTGY_MKT_AREA_NAME;
                 '''

    print(datetime.strftime(DATE, '%Y-%m-%d'))
    query = query.format(DATE=datetime.strftime(DATE, '%Y-%m-%d'))
    df = pd.read_sql(query, cnxn)
    cnxn.close()
    return df

def order_query_rachael(DATE):
    cnxn = pyodbc.connect(link)
    query = '''SELECT
                 
                  CAST(COALESCE(O1.SNAP_DT, D1.SNAP_DT) AS DATE) AS SNAP_DT,
                  C.CUST_GRP_NAME AS CUSTOMER,
                  CAST(COALESCE(O1.PLN_DELIV_DT, D1.PLN_DELIV_DT) AS DATE) AS PLN_DEL_DT,
                 
                  SUM(ZEROIFNULL(O1.BPI_COMMIT)) AS PRI_AND_CURR_MTH_COMMIT_QTY,
                 
                 SUM(ZEROIFNULL(D1.PRI_AND_CURR_MTH_INPROC_QTY)        ) AS PRI_AND_CURR_MTH_IN_PROC_QTY,
                  Zeroifnull(PRI_AND_CURR_MTH_COMMIT_QTY)+ zeroifnull(PRI_AND_CURR_MTH_IN_PROC_QTY) AS WORKING,
                  SUM(zeroifnull(D1.DELIV_QTY)) AS SHIP_QTY,
                  SHIP_QTY +WORKING AS shipped_plus_working

               
            FROM
                          (
                                         SELECT 
                                                       O.SHIP_TO_CUST_ID,
                                                       O.SNAP_DT,
                                                       O.PLN_DELIV_DT,
                                                       O.MATL_ID,
                                                      -- O.DELIV_BLK_CD,
                                                       SUM(O.RPT_ORDER_QTY) AS ORD_QTY,
                                                       SUM(O.RPT_CNFRM_QTY) AS CONFIRM_QTY,
                                                       SUM(
                                                       CASE
                                                     
                                                          WHEN O.PLN_GOODS_ISS_DT < ADD_MONTHS(TRUNC(O.SNAP_DT, 'MM') , 1) 
                                                                        THEN O.OPEN_CNFRM_QTY
                                                                      ELSE 0
                                                       END) AS BPI_COMMIT,
                                                       SUM(O.OPEN_CNFRM_QTY) AS OPEN_CONFIRM_QTY
                                         FROM
                                                       NA_BI_VWS.ORDER_SCHD_AGR_DETAIL_SNAP O
                                         WHERE
                                                       O.OPEN_ORDER_IND = 'Y'
                                                       AND O.PO_TYPE_ID <>'RO'
                                                       AND O.SNAP_DT = '{DATE}'
                                                  --     AND  TRUNC(O.SNAP_DT,'MM') =TRUNC(O.PLN_DELIV_DT,'mm')
                                         GROUP BY 
                                                       O.SHIP_TO_CUST_ID,
                                                       O.SNAP_DT,
                                                       O.MATL_ID,
                                                       O.PLN_DELIV_DT
                                                       --   O.DELIV_BLK_CD
                          )
                          O1
                          FULL OUTER JOIN
                                         (
                                                       SELECT
                                                                      D.SHIP_TO_CUST_ID,
                                                                      D.MATL_ID,
                                                                      D.SNAP_DT,
                                                                      --   ''AS DELIV_BLK_CD,
                                                                      D.PLN_GOODS_MVT_DT AS PLN_DELIV_DT,
                                                                      SUM(CASE WHEN D.ACTL_GOODS_ISS_DT IS NOT NULL THEN D.RPT_DELIV_QTY ELSE 0 END) AS DELIV_QTY,
                                                                        SUM(
                                                                            CASE
                                                                                WHEN TRUNC(D.PLN_GOODS_MVT_DT, 'MM') <= TRUNC(D.SNAP_DT, 'MM')
                                                                                        AND D.ACTL_GOODS_ISS_DT IS NULL
                                                                                    THEN ZEROIFNULL(D.DELIV_QTY)
                                                                                 ELSE 0 
                                                                        END) AS PRI_AND_CURR_MTH_INPROC_QTY
                                                       FROM
                                                                      NA_BI_VWS.DELIVERY_SNAP D
                                                       WHERE
                                                                      D.SNAP_DT = '{DATE}'
                                                                      AND TRUNC(COALESCE(D.ACTL_GOODS_ISS_DT, CAST('{DATE}' AS DATE)) ,'MM') = TRUNC(CAST('{DATE}' AS DATE),'MM')
                                                       GROUP BY
                                                                      D.SHIP_TO_CUST_ID,
                                                                      D.MATL_ID,
                                                                      D.SNAP_DT,
                                                                      D.PLN_GOODS_MVT_DT
                                                                     
                                         )
                                         D1
                          ON
                                         O1.SNAP_DT = D1.SNAP_DT
                                         AND O1.SHIP_TO_CUST_ID = D1.SHIP_TO_CUST_ID
                                         AND O1.MATL_ID = D1.MATL_ID
                                         AND O1.PLN_DELIV_DT = D1.PLN_DELIV_DT
                  
                          INNER JOIN NA_BI_VWS.CUSTOMER C
                          ON
                                         C.SHIP_TO_CUST_ID = COALESCE(O1.SHIP_TO_CUST_ID, D1.SHIP_TO_CUST_ID)
                          INNER JOIN NA_BI_VWS.MATERIAL M
                          ON
                                         M.MATL_ID = COALESCE(O1.MATL_ID, D1.MATL_ID)
            WHERE
                          -- M.PBU_NBR = '03'
                          --AND M.MKT_AREA_NBR ='05'
                          M.PBU_NBR = '01'
                          AND M.SUPER_BRAND_ID IN('01', '02', '03')
                          AND(C.SALES_ORG_CD IN('N302', 'N312'))
            GROUP BY
                          1,
                          2,
                          3
            ORDER BY
                          1,
                          2,3;
                 '''

    print(datetime.strftime(DATE, '%Y-%m-%d'))
    query = query.format(DATE=datetime.strftime(DATE, '%Y-%m-%d'))
    df = pd.read_sql(query, cnxn)
    cnxn.close()
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

    GY_Holiday = pd.concat((GY_Holiday, APRIL, STRIKE))
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

    return GY_Holiday


def acquire_historical_order():
    orders = pd.DataFrame()
    dates = pd.date_range(start='2013-12-31',
                          end= datetime.strftime(datetime.today()-timedelta(days=1), "%Y-%m-%d"),
                          freq='d')
    for i in range(len(dates) - 1):
        order = order_query_rachael(dates[i])
#        order['DEL_DT'] = pd.to_datetime(order['DEL_DT'], errors='coerce')
        order['SNAP_DT'] = pd.to_datetime(order['SNAP_DT'], errors='coerce')
        # order = order.dropna(subset=['SNAP_DT','DEL_DT'])
        orders = orders.append(order, ignore_index=True)
    orders.to_pickle('historical_orders\historical_order_Rachael.pickle')


def prep_confirm_order():
    df = pd.read_pickle('historical_orders\historical_order_Rachael.pickle')
    df['DEL_DT'] = pd.to_datetime(df['DEL_DT'])
    df.index = df.DEL_DT
    df = df.groupby('CUSTOMER').resample('d').sum().reset_index().fillna(value=0)

    return df


def LV_Product(active_cust):
    '''
    The following code will read IHS data and group them based on active customers
    Then upsampling on daily basis
    :param active_cust:
    :return:
    '''
    from pandas.api.types import is_string_dtype
    from pandas.api.types import is_numeric_dtype as is_num

    df=pd.read_excel(r'Light Vehicle Production Forecast 02-14-2020 04_20 PM.xlsx')
    df.columns=df.columns.str.strip().str.upper().str.replace('VP: ','').str.replace(' ','_').\
                str.replace('-','_').str.replace('___','_')
    months=[name for name in df.columns if is_num(df[name])]
    months=[name for name in months if 'CY_' not in name]
    DATES=[datetime.strptime(month,'%b_%Y') for month in months]
    df1=df.copy()
    columns=months+['SALES_GROUP']
    df1=df1.groupby('SALES_GROUP')[months].sum().T
    df1.index= DATES
    df1.index.rename('PR_DATE',inplace=True)
    df1['Chrysler']=df1[['Chrysler','Fiat']].sum(axis=1)
    df1.rename(columns={'General Motors':'GM','Subaru':'SIA','Volkswagen':'VW','Hyundai':'HYUNDAI'},inplace=True)
    cols=[col for col in df1.columns if col in active_cust]
    df1 = df1[cols]
    df1 = df1.stack().reset_index(name='PRD_QTY').rename(columns={'SALES_GROUP': 'CUSTOMER'})

    # upsampling on a daily basis
    d={}
    for cust in df1.CUSTOMER.unique():
        d[cust] = df1[df1.CUSTOMER==cust].resample('D').interpolate(method='spline',order=2).\
        replace({'CUSTOMER':{np.nan:cust}})

    df1 = pd.concat({k:pd.DataFrame(v) for k,v in d.items()},axis=0).reset_index().drop(columns='level_0')
    df1 = df1.groupby(['PR_DATE','CUSTOMER'],as_index=False)['PRD_QTY'].sum()

    return df1


def data_prep_total(df1, df2, link):
    # df1>> order;df2>>shipment
    df = pd.merge(df2, df1, left_on=['SHIP_DT', 'CUSTOMER'],
                  right_on=['DEL_DT', 'CUSTOMER'], how='right'). \
        drop(columns=['SHIP_DT']).rename(columns={'DEL_DT': 'DATE'})
    # put small bussiness in one basket
    df = df.groupby(['DATE'], as_index=False).agg({'PRI_AND_CURR_MTH_COMMIT_QTY': 'sum',
                                                   'PRI_AND_CURR_MTH_IN_PROC_QTY': 'sum',
                                                   'SHIPPED_QTY': 'sum'})
    df = df.loc[df.DATE >= '2014', :]
    # df['SHIPPED_QTY'] = median_filter(df, varname='SHIPPED_QTY', window=30, std=3)
    df['year'] = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    df['week'] = df['DATE'].dt.week
    df['day'] = df['DATE'].dt.day
    df['weekday'] = df['DATE'].dt.dayofweek
    df['total_shipment'] = df.groupby(['year', 'month'])['SHIPPED_QTY'].cumsum()
    df['working'] = df[['PRI_AND_CURR_MTH_COMMIT_QTY',
                        'PRI_AND_CURR_MTH_IN_PROC_QTY']].sum(axis=1)
    df['working_agg'] = df.groupby(['year', 'month'])['working'].cumsum()
    df['max_monthly_shipment'] = df.groupby(['year', 'month'])['total_shipment'].transform('last')
    df['num_day_in_month'] = df.groupby(['year', 'month'])['day'].transform('last')
    df['day_left'] = df['num_day_in_month'] - df['day']

    max_daily = df[df.DATE >= '2014'].groupby(['month'], as_index=False)['SHIPPED_QTY'].max()
    min_daily = df[df.DATE >= '2014'].groupby(['month'], as_index=False)['SHIPPED_QTY'].min()
    for i in range(1, 13):
        df.loc[df.month == i, 'max_daily'] = max_daily.loc[max_daily.month == i, 'SHIPPED_QTY'].values * 1.1
        df.loc[df.month == i, 'min_daily'] = min_daily.loc[min_daily.month == i, 'SHIPPED_QTY'].values * 0.9
    # now add holidays
    holidays = events_holidays_update(link)
    df['is_holiday'] = np.where(df.DATE.isin(holidays.ds), 1, 0)
    df['next_is_holiday'] = df.is_holiday.shift(-1).fillna(value=0)
    df['pre_is_holiday'] = df.is_holiday.shift(1).fillna(value=0)
    log_cols = ['SHIPPED_QTY', 'PRI_AND_CURR_MTH_COMMIT_QTY', 'max_daily', 'min_daily',
                'PRI_AND_CURR_MTH_IN_PROC_QTY', 'working_agg', 'working']
    for name in log_cols:
        df.loc[(df[name] < 0) | (df[name].isnull()), name] = 0
        df[name] = np.log(df[name] + 1)
    df['cap'] = np.max(df['SHIPPED_QTY'])
    return df

def mape(ac,fc):
    return((ac-fc)/ac)

def model_prophet():
    config = [10.0, 3.1668421052631577, 30.0, 'additive', 8.25, 'logistic', 15, 15, 5, 20]
    cp, pscale, season, season_m, holiday_sc, growth, monthFreq, weekFreq, quarterFreq, yearFreq = config
    model_fbp = Prophet(seasonality_mode=season_m,
                        changepoint_range=0.78,
                        n_changepoints=cp,
                        changepoint_prior_scale=pscale,
                        seasonality_prior_scale=season,
                        holidays_prior_scale=holiday_sc,
                        yearly_seasonality=False,
                        daily_seasonality=False,
                        weekly_seasonality=False,
                        holidays=holidays,
                        growth=growth
                        ).add_seasonality(name='monthly',
                                          period=30.5,
                                          fourier_order=monthFreq
                                          ).add_seasonality(name='weekly',
                                                            period=7,
                                                            fourier_order=weekFreq
                                                            ).add_seasonality(name='yearly',
                                                                              period=365.25,
                                                                              fourier_order=yearFreq
                                                                              ).add_seasonality(name='quarterly',
                                                                                                period=365.25 / 4,
                                                                                                fourier_order=quarterFreq
                                                                                                # its 15, I just changed it to see the potential impact of quaretr end
                                                                                                )

if __name__ == '__main__':
    acquire_historical_order()
    order = prep_confirm_order()
    delivery = direct_shipment_query()
    holidays = events_holidays_update(link)
    df_customer = data_prep_total(order, delivery, link)
    df_customer.index = pd.to_datetime(df_customer.DATE)
    exogenous_features = ['working']
    test_start = pd.date_range(start='2019-01-01', end='2019-12-31', freq='MS')
    test_end = pd.date_range(start='2019-01-01', end='2019-12-31', freq='M')
    cutoff = [pd.date_range(start=st, end=en, freq='B')[4] for st, en in zip(test_start, test_end)]
    test_start = [i + timedelta(days=1) for i in cutoff]  # lag cutoff date by 1 day
    error = []

    for i in range(len(cutoff)):
        cut_date = datetime.strftime(cutoff[i], '%Y-%m-%d')
        ts = datetime.strftime(test_start[i], '%Y-%m-%d')
        te = datetime.strftime(test_end[i], '%Y-%m-%d')

        df_train = df_customer.loc[df_customer.DATE < cut_date,:]
        df_valid = df_customer.loc[(df_customer.DATE >= ts) & (df_customer.DATE <= te),:]

        exogenous_features = ['PRI_AND_CURR_MTH_COMMIT_QTY','working','day_left', 'is_holiday']

        model_fbp=Prophet(seasonality_mode='multiplicative',holidays=holidays)
        model_fbp.add_country_holidays(country_name='US')
        for feature in exogenous_features:
            model_fbp.add_regressor(feature, prior_scale=0.1)

        with suppress_stdout_stderr():
            model_fbp.fit(
                df_train[["DATE", 'SHIPPED_QTY','cap'] + exogenous_features].
                    rename(columns={"DATE": "ds", 'SHIPPED_QTY': "y"}))
        forecast = model_fbp.predict(
            df_valid[["DATE", 'SHIPPED_QTY','cap'] + exogenous_features].rename(columns={"DATE": "ds"}))
        # model_fbp.plot_components(forecast)
        df_valid.loc[:, "Forecast_Prophet"] = np.exp(forecast.yhat).values
        # model_fbp.plot_components(forecast)
        actual, forecast = sum(np.exp(df_valid['SHIPPED_QTY']).values), sum(df_valid["Forecast_Prophet"].values)
        sub_error = mape(actual, forecast)
        print(datetime.strftime(test_start[i], '%Y-%B'), np.round(sub_error, 3))
        error.append([datetime.strftime(test_start[i], '%Y-%B'), actual, forecast, sub_error])
