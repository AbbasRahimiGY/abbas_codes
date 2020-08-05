from zipfile import ZipFile
import wget
import pandas as pd
from datetime import datetime, timedelta
import pyodbc
import numpy as np
import os, warnings
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
import warnings
from sklearn import preprocessing
from sklearn.metrics import max_error

plt.style.use('fivethirtyeight')

warnings.filterwarnings('ignore')

os.chdir(r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast\covid')
# Make a connection
link = ('DSN=EDWTDPRD;UID=AA68383;PWD=ilivein339westberry')
pyodbc.pooling = False
sns.set_context("notebook", font_scale=0.1, rc={"lines.linewidth": 3.5})
sns.set_context("poster")


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


def timeseries_train_test_split(X, test_size):
    """
        Perform train-test split with respect to time series structure
    """

    # get the index after which test set starts
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    X_test = X.iloc[test_index:]
    return X_train, X_test


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
                                OR  NA_BI_VWS.CUSTOMER.DISTR_CHAN_CD = '32'
                            GROUP BY 1,2
                            ORDER BY 1,2'''
    with pyodbc.connect(link, autocommit=True) as connect:
        df = pd.read_sql(query, connect)
    df = df.pivot_table('SHIPPED_QTY', index='SHIP_DT', columns='CUSTOMER',
                        aggfunc='sum', dropna=False, fill_value=0)

    df.index = pd.to_datetime(df.index)
    df = df.resample('D').sum()
    df1 = df.resample('y').sum()
    active_customers = [col for col in df1.columns if df1['2019':][col].sum() > 100]
    # print('active customer are={}'.format(active_customers))
    df = df[active_customers]
    df = df['2014':]
    df[df < 0] = 0
    return df


def prophet_optim(df_train, features):
    param_grid = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 15, 18, 20, 22, 25, 30, 35]
    scores = []
    for cp_scale in param_grid:
        # split data in few different chuncks
        training, valid = timeseries_train_test_split(df_train, test_size=0.15)
        grid_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
                             changepoint_prior_scale=cp_scale,
                             seasonality_mode='multiplicative').add_country_holidays('US')
        for feature in features:
            grid_model.add_regressor(feature, mode='additive')
        with suppress_stdout_stderr():
            forecast = grid_model.fit(training[["ds", "y"] + features]). \
                predict(valid[["ds"] + features])
            error = max_error(valid['y'].values, forecast['yhat'].values)
            scores.append([cp_scale, error])
    scores.sort(key=lambda tup: tup[1])
    # done with grid search
    print('Best prior scale = {}'.format(scores[0][0]), 'Maximum error = {}'.format(int(scores[0][1])))
    return scores[0][0]


def read_covid():
    def plot_results(df):
        cases = df.copy()
        cases[['time']] = cases[['time']].astype('Datetime64')
        cases.set_index('time', inplace=True)
        cases_stack = cases.stack().reset_index().rename(columns={'level_1': 'Scenario', 0: 'Count'})
        cases_stack.to_excel(r'\\AKRTABLEAUPNA01\Americas_Market_Analytics$\COVID\admission_scenarios.xlsx')
        sns_plot = sns.relplot(x='time', y='Count',
                               hue='Scenario', palette="ch:2.5,.25",
                               height=6, aspect=3, legend="full",
                               kind="line", data=cases_stack[cases_stack.time >= '2020-01-01'])
        sns_plot.savefig(r'\\AKRTABLEAUPNA01\Americas_Market_Analytics$\COVID\admission_scenarios.png')

    url = "https://ihmecovid19storage.blob.core.windows.net/latest/ihme-covid19.zip"
    wget.download(url, 'ihme-covid19.zip')
    with ZipFile('ihme-covid19.zip', 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        # Iterate over the file names
        for fileName in listOfFileNames:
            # Check filename endswith csv
            if 'Reference' in fileName:
                # Extract a single file from zip
                df = pd.read_csv(zipObj.extract(fileName, 'temp_csv'))
    df = df[df.location_name == 'United States of America'][
        ['date', 'admis_mean', 'admis_lower', 'admis_upper']].rename(columns={'date': 'time'})
    df[['time']] = df[['time']].astype('Datetime64')
    plot_results(df)
    start = datetime.strftime(df['time'].iloc[0] - timedelta(days=1), '%Y-%m-%d')
    end = datetime.strftime(df['time'].iloc[-1] + timedelta(days=1), '%Y-%m-%d')
    for col in ['admis_mean', 'admis_lower', 'admis_upper']:
        # scale between 0 and 1
        st_scaler = preprocessing.MinMaxScaler()  #
        df.loc[:, col] = st_scaler.fit_transform(df.loc[:, col].values.reshape(-1, 1))

    before = pd.DataFrame({'time': pd.date_range('2014-01-01', start, freq='d'),
                           'admis_mean': 0, 'admis_lower': 0, 'admis_upper': 0})

    after = pd.DataFrame({'time': pd.date_range(end, '2020-12-31', freq='d'),
                          'admis_mean': 0, 'admis_lower': 0, 'admis_upper': 0})

    df = pd.concat([before, df, after])
    os.remove('ihme-covid19.zip')
    return df


def read_LV():
    active_cust = ['Chrysler', 'Ford', 'GM', 'HYUNDAI', 'Honda', 'Nissan', 'OE Auto Gen Cust', 'OE Mobile Home',
                   'OE Non Auto Gen Cust', 'SIA', 'Tesla', 'Toyota', 'Tredit', 'VW']
    df = pd.read_excel(r'202007NAPAK.xls', header=8, sheet_name='Mth-Qtr-CY')
    df.columns = df.columns.str.strip().str.upper().str.replace('VP: ', '').str.replace(' ', '_'). \
        str.replace('-', '_').str.replace('___', '_')
    mon_name = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    months = [col for name in mon_name for col in df.columns if name in col]
    index = [datetime.strptime(date, '%b_%Y') for date in months]
    df1 = df.copy()
    other_col = ['SALES_PARENT']
    columns = months + other_col
    df1 = df1.groupby(other_col)[months].sum().T
    df1.index = index
    df1.index.rename('PR_DATE', inplace=True)
    if 'Fiat' in df1.columns:
        df1['Chrysler'] = df1[['Chrysler', 'Fiat']].sum(axis=1)
    df1.rename(columns={'General Motors': 'GM', 'Subaru': 'SIA', 'Volkswagen': 'VW', 'Hyundai': 'HYUNDAI'},
               inplace=True)
    df1 = df1.stack(level=[i for i in range(len(other_col))]).reset_index().rename(columns={0: 'PRD'}).fillna(0)
    df1 = df1[df1.SALES_PARENT.isin(active_cust)].rename(columns={'SALES_PARENT': 'CUSTOMER'})
    # summarize total
    df1 = df1.groupby('PR_DATE')['PRD'].sum()
    df1.to_excel(r'\\AKRTABLEAUPNA01\Americas_Market_Analytics$\COVID\LV.xlsx')
    # df1['month']=pd.to_datetime(df1['PR_DATE'], format='%Y-%m')
    start_date = df1.index.min() - pd.DateOffset(day=1)
    end_date = df1.index.max() + pd.DateOffset(day=31)
    dates = pd.date_range(start_date, end_date, freq='D')
    dates.name = 'date'
    df1 = pd.DataFrame(df1.reindex(dates, method='ffill'))
    st_scaler = preprocessing.MinMaxScaler()
    df1.loc[:, "PRD"] = st_scaler.fit_transform(df1.loc[:, 'PRD'].values.reshape(-1, 1))
    df1 = df1.reset_index().rename(columns={'date': 'time'})
    return df1


shipment = direct_shipment_query()
cases = read_covid()
LV = read_LV()
cutoff_date = datetime.strftime(datetime.today() - timedelta(days=1), "%Y-%m-%d")

ship_sum = pd.DataFrame(shipment.sum(axis=1)).reset_index().rename(columns={'SHIP_DT': 'time', 0: 'shipped_qty'})
ship_sum = ship_sum[ship_sum.time <= cutoff_date]
combo = pd.merge(ship_sum, cases, on='time', how='right').fillna(0)
# join with LV data
combo = pd.merge(combo, LV, on='time')
combo['cap'] = max(combo.shipped_qty) * 1.1

df_train, df_test = combo[combo.time <= cutoff_date].rename(columns={'shipped_qty': 'y', 'time': 'ds'}), \
                    combo[combo.time > cutoff_date].rename(columns={'shipped_qty': 'y', 'time': 'ds'})

# Idea is to include effect of covid in the past but set it to 0 for future when we look into no_covid case
df_train['no_COVID'] = df_train['admis_mean']
df_test['no_COVID'] = 0.0
exogenous_features = ['admis_mean', 'admis_lower', 'admis_upper', 'no_COVID']
data = {}
#pr_scale_covid = prophet_optim(df_train, ['admis_mean', 'PRD'])
# added a case where COVID has no effect
# pr_scale_nocvid = prophet_optim(df_train, [])
for name in exogenous_features:
    if name == 'no_COVID':
        pr_scale = 0.1
    else:
        pr_scale = 0.1

    model = Prophet(daily_seasonality=False,
                    changepoint_prior_scale=pr_scale,
                    weekly_seasonality=True, yearly_seasonality=True,
                    seasonality_mode='multiplicative').add_country_holidays('US')
    model.add_regressor(name, mode='additive')
    model.add_regressor('PRD', mode='additive')

    with suppress_stdout_stderr():
        model.fit(df_train[["ds", "y"] + [name, 'PRD']])
    forecast = model.predict(df_test[["ds"] + [name, 'PRD']])
    forecast.loc[forecast.yhat < 0, "yhat"] = 0
    data[f'{name}'] = forecast.set_index('ds').resample('d').sum()['yhat']
data = pd.DataFrame.from_dict(data)
past_future = pd.concat([df_train, data.reset_index()]).fillna(0).drop(columns='cap')
past_future.set_index('ds', inplace=True)
for col in ['admis_lower', 'admis_mean', 'admis_upper','no_COVID']:
    past_future.loc[:, col] = past_future.loc[:, [col, 'y']].sum(axis=1)
past_future.drop(columns='y', inplace=True)
# past_future[['ds']] = past_future[['ds']].astype('Datetime64')
past_future.to_excel(r'\\AKRTABLEAUPNA01\Americas_Market_Analytics$\COVID\daily_forecast.xlsx')
data_stack = data.resample('d').sum()
data_stack = data_stack.stack().reset_index().rename(columns={'level_1': 'Scenario', 0: 'Volume'})
data_stack.pivot_table(values='Volume', index='ds', columns='Scenario').to_excel('today_forecast.xlsx')
# Plot the lines on two facets
sns_plot = sns.relplot(x='ds', y='Volume',
                       hue='Scenario', palette="ch:2.5,.25",
                       height=6, aspect=3, legend="full",
                       kind="line", data=data_stack)
sns_plot.set_xlabels(label='Date')
sns_plot.set_ylabels(label='Daily Quantity')
sns_plot.savefig("Monthly_Forecast.png")
