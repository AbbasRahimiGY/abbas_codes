import calendar
import pandas as pd
from datetime import datetime, timedelta
import pyodbc
import numpy as np
import os, warnings
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import concurrent.futures
from models import prophetCV, poly_linearCV, mars, model_stack, cbCV, xgbCV,xgb,poly_linear,cb

os.environ['NUMEXPR_NUM_THREADS'] = '8'
warnings.simplefilter('ignore')
# ---------------------------------------------------#
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
os.chdir(r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast')
# Make a connection
link = ('DSN=EDWTDPRD;UID=AA68383;PWD=baradarkhobvaghashang1364')
pyodbc.pooling = False


def direct_shipment_query(link):
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
            AND   NA_BI_VWS.CUSTOMER.SALES_ORG_CD IN('N302', 'N312')
            GROUP BY 1,2
            ORDER BY 1,2;'''

    with pyodbc.connect(link, autocommit=True) as connect:
        df = pd.read_sql(query, connect)
    df['SHIP_DT'] = pd.to_datetime(df['SHIP_DT'])

    _ = df.pivot_table('SHIPPED_QTY', index='SHIP_DT', columns='CUSTOMER',
                       aggfunc='sum', dropna=False, fill_value=0).resample('Y').sum()

    active_CUSTOMERs = [col for col in _.columns if _['2019':][col].sum() > 100]
    df = df.loc[df.CUSTOMER.isin(active_CUSTOMERs), :]
    df = df.rename(columns={'SHIP_DT': 'SNAP_DT', 'SHIPPED_QTY': 'SHIP_QTY'})
    # fill in for CUSTOMER which are missing on certain days
    df = pd.pivot_table(df, index=['SNAP_DT'], columns='CUSTOMER',
                        values=['SHIP_QTY']).fillna(0)
    df = df.stack(level=['CUSTOMER']).reset_index()
    df['SHIP_QTY'] = df['SHIP_QTY'].astype('int32')

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

    GY_Holiday.reset_index(inplace=True)

    GY_Holiday = GY_Holiday[(GY_Holiday.holiday != 'Day After Thanksgivi') &
                            (GY_Holiday.holiday != 'Good Friday - Local') &
                            (GY_Holiday.holiday != 'Good Friday')]
    """    GY_Holiday['day'] = GY_Holiday.ds.dt.day
        GY_Holiday['month'] = GY_Holiday.ds.dt.strftime("%b")
        GY_Holiday['year'] = GY_Holiday.ds.dt.year
        df = pd.concat([GY_Holiday, pd.get_dummies(GY_Holiday['holiday'])], axis=1)
        df = df.drop(columns=['index', 'holiday', 'ds'])"""
    return GY_Holiday


def plan_query(link):
    query = '''SELECT
               SP.PERD_BEGIN_MTH_DT AS "DATE",
               --SP.LAG_DESC,
               SP.FORC_TYP_DESC,
               --SP.SALES_ORG_CD,
               --SP.DISTR_CHAN_CD,
               --SP.CUST_GRP_ID,
               CG.CUST_GRP_NAME AS "CUSTOMER",
               --SP.MATL_ID,
               SUM(SP.OFFCL_AOP_QTY) AS AOP_QTY,
               SUM(SP.OFFCL_SOP_SLS_PLN_QTY) AS SOP_QTY
                FROM
                       NA_BI_VWS.CUST_SLS_PLN_SNAP SP
                       LEFT OUTER JOIN NA_BI_VWS.CUSTOMERGROUP CG
                       ON
                             SP.CO_CD = CG.CO_CD
                             AND SP.SALES_ORG_CD = CG.SALES_ORG_CD
                             AND SP.DISTR_CHAN_CD = CG.DISTR_CHAN_CD
                             AND SP.CUST_GRP_ID = CG.CUST_GRP_ID
                       INNER JOIN NA_BI_VWS.MATERIAL M
                       ON
                             M.MATL_ID = SP.MATL_ID
                WHERE
                       --SP.LAG_DESC = '0'
                       SP.EST_TYP_IND ='C'
                       AND SP.PERD_BEGIN_MTH_DT BETWEEN '2020-01-01' AND '2020-12-01'
                       AND M.PBU_NBR = '01'
                      -- AND SP.SALES_ORG_CD IN('N302', 'N312')
                       AND SP.DATA_SRC_CD = 'CUST_SLS_PLN_MTH'
                       AND CG.CUST_HIER_GRP_2_DESC ='oe'
                GROUP BY
                       SP.PERD_BEGIN_MTH_DT,
                       --SP.LAG_DESC,
                       SP.FORC_TYP_DESC,
                       --SP.SALES_ORG_CD,
                       --SP.DISTR_CHAN_CD,
                       --SP.CUST_GRP_ID,
                       CG.CUST_GRP_NAME
                       --SP.MATL_ID
                HAVING
                       AOP_QTY + SOP_QTY <> 0
                ORDER BY 1;'''
    with pyodbc.connect(link, autocommit=True) as connect:
        df = pd.read_sql(query, connect)
    df['DATE'] = pd.to_datetime(df['DATE'])
    DATE = datetime.now() - timedelta(days=1)
    df = df[(df.DATE.dt.month == DATE.month) & (df.DATE.dt.year == DATE.year)]
    df['SOP_QTY'] = df['SOP_QTY'].astype(int)
    return df


def prep_order_data(active_CUSTOMER):
    def add_missing_dates(grp):
        _ = grp.set_index('PLN_DEL_DT')
        _ = _.reindex(idx)
        return _

    # read orders
    df = pd.read_pickle('historical_orders\historical_order_Rachael.pickle')
    df[['PRI_AND_CURR_MTH_COMMIT_QTY',
        'PRI_AND_CURR_MTH_IN_PROC_QTY']] = df[['PRI_AND_CURR_MTH_COMMIT_QTY',
                                               'PRI_AND_CURR_MTH_IN_PROC_QTY']].astype('int32')
    df = df[df.CUSTOMER.isin(active_CUSTOMER)]
    df['PLN_DEL_DT'] = df.PLN_DEL_DT.astype('Datetime64')
    df = df.drop_duplicates()
    # fill in for days some CUSTOMERs are missing
    df = df[['SNAP_DT', 'CUSTOMER', 'PLN_DEL_DT',
             'PRI_AND_CURR_MTH_COMMIT_QTY', 'PRI_AND_CURR_MTH_IN_PROC_QTY']]
    df = pd.pivot_table(df, index=['SNAP_DT', 'PLN_DEL_DT'], columns='CUSTOMER',
                        values=['PRI_AND_CURR_MTH_IN_PROC_QTY', 'PRI_AND_CURR_MTH_COMMIT_QTY']).fillna(0)
    df = df.stack(level=['CUSTOMER']).reset_index()

    df.PLN_DEL_DT = pd.to_datetime(df.PLN_DEL_DT)
    df = df[(df.PLN_DEL_DT >= '2014') & (df.PLN_DEL_DT < '2021') & (df.SNAP_DT >= '2014')]

    # fill in all missing PLN_DEL_DT
    idx = pd.date_range(df.PLN_DEL_DT.min(), df.PLN_DEL_DT.max(), freq='d')
    # Group by country name and extend
    df = df.groupby(['SNAP_DT', 'CUSTOMER']).apply(add_missing_dates).drop(columns=['SNAP_DT', 'CUSTOMER']). \
        reset_index().fillna(0).rename(columns={'level_2': 'PLN_DEL_DT'})
    # fill in for missing CUSTOMERs
    df[['PRI_AND_CURR_MTH_COMMIT_QTY',
        'PRI_AND_CURR_MTH_IN_PROC_QTY']] = df[['PRI_AND_CURR_MTH_COMMIT_QTY',
                                               'PRI_AND_CURR_MTH_IN_PROC_QTY']].astype('int32')
    df.to_pickle('historical_orders\historical_order_Rachael_filled_missing.pickle')
    return df


def prep_data(data, holiday, scale_list=None, scale_type='no_scale', pca_dim=0):
    '''
    current columns
    ['index', 'SNAP_DT', 'CUSTOMER', 'CON', 'IP', 'SHIP_QTY', 'day', 'month',
       'num_day_in_month', 'day_left', 'tot_ship', 'stright_line',
       'tot_ship_time_day', 'working', 'final_week', 'final_week_ip',
       'final_week_con', 'Christmas', 'Christmas Eve', 'Easter',
       'Independence Day', 'Labor Day', 'Mday', 'New Year Eve', 'NY',
       'Thanksgiving', 'Apr', 'Aug', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May',
       'Nov', 'Oct', 'Sep', 'Fri', 'Mon', 'Sat', 'Thu', 'Tue', 'Wed']
    :param pca_dim: number of decompositions
    :param scale_type: "MinMax" or "log"
    :param scale_list: exogenous features
    :param data:
    :param holiday:
    :return:
    '''
    df = data.copy()
    if df.index.dtype != 'Datetime64':
        df = df.set_index('PLN_DEL_DT')
    # some shipped qty are negative so we set them to zero
    df.loc[df.SHIP_QTY < 0, 'SHIP_QTY'] = 0
    df = df. \
        rename(columns={'PRI_AND_CURR_MTH_COMMIT_QTY': 'CON', 'PRI_AND_CURR_MTH_IN_PROC_QTY': 'IP'})
    df['day'] = df.index.day
    df['weekday'] = df.index.strftime("%a")
    df['month'] = df.index.strftime("%b")
    df['year'] = df.index.year
    df['num_day_in_month'] = df['num_day_in_month'] = df.index.to_series(). \
        apply(lambda x: calendar.monthrange(x.year, x.month)[1])
    df['day_left'] = df['num_day_in_month'] - df['day']
    df['tot_ship'] = df.groupby(['CUSTOMER', 'year', 'month'])['SHIP_QTY'].cumsum().astype('int32')
    df['working'] = df['IP'] + df['CON']
    df['working_shipped'] = df['working'] + df['tot_ship']
    df['straight_line'] = ((df['tot_ship'] / df['day']) * (df['day'] + df['day_left'])).astype('int32')
    df['tot_ship_time_day'] = (df['tot_ship'] * df['day_left']).astype('int32')
    df['final_week'] = np.where(df['day_left'] < 4, 1, 0)
    df['final_week_ip'] = df['IP'] * df['final_week']
    df['final_week_con'] = df['CON'] * df['final_week']
    df['yday_gain'] = df.groupby('CUSTOMER')['SHIP_QTY'].transform(lambda x: x.diff())
    df['past_two'] = df.groupby(['CUSTOMER'])['SHIP_QTY'].rolling(window=2).mean().values
    df['past_seven'] = df.groupby(['CUSTOMER'])['SHIP_QTY'].rolling(window=7).mean().values
    df['past_twentyeight'] = df.groupby(['CUSTOMER'])['SHIP_QTY'].rolling(window=28).mean().values
    df.dropna(how='any', inplace=True)
    # add holidays
    holiday_nd = holiday.drop_duplicates()
    df.reset_index(inplace=True)
    df = pd.merge(df, holiday_nd, left_on=['PLN_DEL_DT'], right_on=['ds'], how='left').drop(columns=['index', 'ds'])
    df['holiday'] = df['holiday'].fillna(value='regular')
    df = pd.concat([df, pd.get_dummies(df['holiday'])], axis=1). \
        drop(columns=['regular', 'holiday']).rename(
        columns={'Memorial Day': 'Mday', 'New Years': 'NY'}).drop_duplicates()
    # get months dummy
    df = pd.concat([df, pd.get_dummies(df['month'])], axis=1)
    # get weekdays dummy
    df = pd.concat([df, pd.get_dummies(df['weekday'])], axis=1)
    df.drop(columns=['day', 'month', 'Sun', 'year', 'Dec', 'weekday'], inplace=True)
    df = df[['SNAP_DT', 'PLN_DEL_DT', 'CUSTOMER', 'SHIP_QTY'] + scale_list]
    if scale_list is None:
        scale_list = []
    if len(scale_list) > 0:
        if scale_type == 'MinMax':
            for name in df.CUSTOMER.unique():
                # scale between 0 and 1
                st_scaler = preprocessing.StandardScaler()  # MinMaxScaler()
                df.loc[df.CUSTOMER == name, scale_list] = \
                    st_scaler.fit_transform(df.loc[df.CUSTOMER == name, scale_list])
            if pca_dim > 0:
                df_pca = []
                for customer in df.CUSTOMER.unique():
                    df_cust = df[df.CUSTOMER == customer].groupby(['SNAP_DT', 'PLN_DEL_DT', 'SHIP_QTY'])[
                        scale_list].sum()
                    pca = PCA(pca_dim)
                    '''ex_variance = np.var(pca.fit_transform(df_cust), axis=0)
                    ex_variance_ratio = ex_variance / np.sum(ex_variance)
                    print  (ex_variance_ratio)'''
                    pca = pd.DataFrame(pca.fit_transform(df_cust),
                                       columns=['pca{}'.format(i) for i in range(pca_dim)])
                    pca['CUSTOMER'] = customer
                    for col in pca.columns:
                        df_cust[col] = pca.loc[:, col].values
                    df_cust.reset_index(inplace=True)
                    # I added IP and CON to later in each model set forecast to zero if there is no orders
                    df_cust.drop(columns=[col for col in scale_list if col not in ['IP', 'CON']], inplace=True)
                    df_pca.append(df_cust)
                df = pd.concat(df_pca)
                scale_list = ['pca{}'.format(i) for i in range(pca_dim)]

        elif scale_type == 'log':
            for col in scale_list:
                df.loc[df[col] < 0, col] = 0
                df.loc[col] = np.log(df[col] + 1)

    return scale_list, df


def data_resampling(df, df_shipment, current, link):
    '''
    Perform resampling on result of several models fit into a normal distrubution
    :param df_shipment:
    :param current:
    :param df:
    :return:
    '''
    def prep_for_tableau(df_sample,df_shipment):
        ship,dif = df_shipment.rename(columns={'SNAP_DT':'Date'}).copy(),df_sample.copy()
        ship[['Date']] = ship[['Date']].astype('Datetime64')
        dif[['Date']] = dif[['Date']].astype('Datetime64')
        df_combo = pd.concat([ship, dif]).fillna(0)[
            ['Date', 'CUSTOMER', 'SHIP_QTY', 'forecast_high', 'forecast_low', 'forecast_mean']]
        for col in ['forecast_high', 'forecast_low', 'forecast_mean']:
            df_combo.loc[:, col] = df_combo.loc[:, [col, 'SHIP_QTY']].sum(axis=1)
        df_combo.drop(columns='SHIP_QTY', inplace=True)
        df_combo.to_excel(r'\\AKRTABLEAUPNA01\Americas_Market_Analytics$\COVID\daily__order_forecast.xlsx')

    #current_month = datetime.strftime(datetime.strptime(current, '%Y-%m-%d'), '%Y-%m')
    df.reset_index(inplace=True)
    df.to_excel('daily_forecast_{}.xlsx'.format(current))
    model_names = [col for col in df.columns if col not in ['Date', 'CUSTOMER']]
    df.Date = df.Date.ffill()
    df['mean'] = df.loc[:, model_names].mean(axis=1).astype('int')
    df['std'] = df.loc[:, model_names].std(axis=1).astype('int')
    for index, row in df.iterrows():
        samples = np.random.normal(loc=row['mean'], scale=row['std'], size=1000)
        df.loc[index, 'forecast_mean'] = round(samples.mean(), 0)
        df.loc[index, 'forecast_low'] = np.where(round(np.percentile(samples, 20), 0) > 0,
                                                 round(np.percentile(samples, 20), 0), 0)
        df.loc[index, 'forecast_high'] = np.where(round(np.percentile(samples, 80), 0) > 0,
                                                  round(np.percentile(samples, 80), 0), 0)
    df.drop(columns=['mean', 'std'], inplace=True)
    prep_for_tableau(df, df_shipment)
    # now merge with current month shipment
    df = df.groupby('CUSTOMER')['forecast_low', 'forecast_mean', 'forecast_high'].sum()
    mtd_ship = df_shipment.copy()
    mtd_ship.set_index('SNAP_DT', inplace=True)
    start_date = datetime.strftime(datetime.strptime(current, '%Y-%m-%d').replace(day=1), '%Y-%m-%d')
    mtd_ship = mtd_ship[start_date:current].groupby('CUSTOMER').resample('M').sum()
    df = pd.merge(df, mtd_ship, on=['CUSTOMER'])
    df['forecast_mean'] = df.loc[:, ['forecast_mean', 'SHIP_QTY']].sum(axis=1).astype('int')
    df['forecast_low'] = df.loc[:, ['forecast_low', 'SHIP_QTY']].sum(axis=1).astype('int')
    df['forecast_high'] = df.loc[:, ['forecast_high', 'SHIP_QTY']].sum(axis=1).astype('int')
    df.drop(columns='SHIP_QTY', inplace=True)
    # now lets add S&OP to our forecast
    plan = plan_query(link)
    pd.merge(df, plan, on=['CUSTOMER']).drop(columns=['DATE', 'FORC_TYP_DESC']).to_excel(
        'monthly_forecast_asof_{}.xlsx'.format(current))


if __name__ == '__main__':
    # -------call input here
    holidays = events_holidays_update(link)
    shipment = direct_shipment_query(link)
    active_CUSTOMER = shipment.CUSTOMER.unique().tolist()
    orders = pd.read_pickle('historical_orders\historical_order_Rachael_filled_missing.pickle')
    type = 'parallel'
    pca_dim = 0
    scale_type = 'None'#'MinMax'
    # -----------------------
    exogenous_features = ['CON', 'IP', 'past_seven', 'past_twentyeight','yday_gain', 'past_two',
                          'final_week', 'final_week_ip', 'final_week_con',
                          'Christmas', 'Christmas Eve', 'Easter',
                          'Independence Day', 'Labor Day', 'Mday', 'New Year Eve', 'NY',
                          'Thanksgiving', 'Apr', 'Aug', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May','Nov', 'Oct', 'Sep',
                           #'Fri', 'Mon', 'Sat', 'Thu', 'Tue', 'Wed'
                          ]
    cutoff_date = datetime.strftime(datetime.today() - timedelta(days=2), "%Y-%m-%d")  # where to cutoff data
    test_date = datetime.strftime(datetime.today() - timedelta(days=1),
                                  "%Y-%m-%d")  # where we sample from snap date for all horizons
    # find the horizon to search
    horizon = calendar.monthrange(datetime.strptime(test_date, "%Y-%m-%d").year,
                                  datetime.strptime(test_date, "%Y-%m-%d").month)[1] - \
              datetime.strptime(test_date, "%Y-%m-%d").day + 1
    # temp horizon
    horizon = np.int((pd.to_datetime('2020-04-30') - pd.to_datetime(test_date)) / np.timedelta64(1, 'D'))
    result = []
    import time

    for lead in range(1, horizon):
        lead_date = datetime.strftime(datetime.strptime(test_date, '%Y-%m-%d') + timedelta(lead), '%Y-%m-%d')
        print(lead_date)
        lead_order = orders[(orders.PLN_DEL_DT - orders.SNAP_DT).dt.days == lead].copy()
        lead_order = lead_order.merge(shipment, on=['SNAP_DT', 'CUSTOMER'], how='left').fillna(0)
        updated_features, lead_order = prep_data(lead_order, holidays, exogenous_features, scale_type, pca_dim)
        if pca_dim > 0:
            updated_features = [col for col in updated_features if col not in ['IP', 'CON'] ]

        if type == 'parallel':
            dict_reg = {'XgBoost': xgb,
                        'Poly': poly_linear,
                        'Prophet': prophetCV,
                        'Spline': mars,
                        'Stacked': model_stack,
                        'CatBoost': cb,
                        # 'LightBoost': lgbm
                        }
            series_train = [lead_order.loc[(lead_order.SNAP_DT <= cutoff_date) & (lead_order.CUSTOMER == cus)]. \
                                rename(columns={'SHIP_QTY': 'y', 'PLN_DEL_DT': 'ds'}) for cus in
                            lead_order.CUSTOMER.unique()]
            series_valid = [lead_order.loc[(lead_order.SNAP_DT == test_date) & (lead_order.CUSTOMER == cus)]. \
                                rename(columns={'SHIP_QTY': 'y', 'PLN_DEL_DT': 'ds'}) for cus in
                            lead_order.CUSTOMER.unique()]
            # define executor
            executor = Parallel(n_jobs=8, backend='multiprocessing')
            start_time = time.time()
            for key, reg in dict_reg.items():
                tasks = (delayed(reg)(df_train, df_valid, updated_features) for df_train, df_valid in
                         zip(series_train, series_valid))
                forecast = executor(tasks)


                day_forecast = pd.DataFrame({'CUSTOMER': lead_order.CUSTOMER.unique().tolist(),
                                             'Forecast': forecast, 'Date': lead_date, 'model': key})
                result.append([day_forecast])
            print("--- %s seconds ---" % (time.time() - start_time))

    forecast_total = pd.concat([pd.concat(x) for x in result])
    forecast_total.to_excel('results.xlsx')
    summary_table = pd.pivot_table(forecast_total, values='Forecast', index=['Date', 'CUSTOMER'], columns='model')
    # perform resampling from normal distribution
    data_resampling(summary_table, shipment, cutoff_date, link)
