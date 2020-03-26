import pandas as pd
from datetime import datetime, timedelta
import pyodbc
import os
import matplotlib.pyplot as plt
from dateutil import relativedelta
import seaborn as sns
import numpy as np
from fbprophet import Prophet

plt.style.use('fivethirtyeight')

os.chdir(r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast')
# Make a connection
link = ('DSN=EDWTDPRD;UID=AA68383;PWD=baradarkhobvaghashang1364')
pyodbc.pooling = False


def direct_shipment_query():
    query = '''SELECT "ACTL_GOODS_ISS_DT" AS "SHIP_DT",
            SUM("DELIV_QTY") AS "SHIPPED_QTY",
            "CUST_GRP_ID" AS "CUST_GRP_ID",
            "CUST_GRP_NAME" AS "CUST_GRP_NAME"
            FROM (
            SELECT "ACTL_GOODS_ISS_DT",
                "CUST_GRP_ID",
                "CUST_GRP_NAME",
                "DELIV_QTY",
                "MKT_CTGY_MKT_AREA_NBR"
                FROM (
                SELECT *
                    FROM (
                    SELECT *
                        FROM (
                        SELECT "Left"."ACTL_GOODS_ISS_DT",
                            "Left"."CUST_GRP_ID",
                            "Left"."DELIV_QTY",
                            "Left"."DISTR_CHAN_CD",
                            "Left"."MATL_ID",
                            "Left"."SHIP_TO_CUST_ID",
                            "Left"."R_CUST_GRP_ID",
                            "Left"."CUST_GRP_NAME",
                            "Left"."DISTR_CHAN_NAME",
                            "Left"."SALES_ORG_CD",
                            "Left"."R_SHIP_TO_CUST_ID",
                            "Right"."MATL_ID" AS "R_MATL_ID",
                            "Right"."MKT_CTGY_MKT_AREA_NBR",
                            "Right"."PBU_NBR",
                            "Right"."SUPER_BRAND_ID"
                            FROM (
                            SELECT "Left"."ACTL_GOODS_ISS_DT",
                                "Left"."CUST_GRP_ID",
                                "Left"."DELIV_QTY",
                                "Left"."DISTR_CHAN_CD",
                                "Left"."MATL_ID",
                                "Left"."SHIP_TO_CUST_ID",
                                "Right"."CUST_GRP_ID" AS "R_CUST_GRP_ID",
                                "Right"."CUST_GRP_NAME",
                                "Right"."DISTR_CHAN_NAME",
                                "Right"."SALES_ORG_CD",
                                "Right"."SHIP_TO_CUST_ID" AS "R_SHIP_TO_CUST_ID"
                                FROM (
                                SELECT "ACTL_GOODS_ISS_DT",
                                    "CUST_GRP_ID",
                                    "DELIV_QTY",
                                    "DISTR_CHAN_CD",
                                    "MATL_ID",
                                    "SHIP_TO_CUST_ID"
                                    FROM (
                                    SELECT *
                                        FROM NA_BI_VWS.DELIVERY) AS "a") AS "Left" INNER JOIN (
                                SELECT "CUST_GRP_ID",
                                    "CUST_GRP_NAME",
                                    "DISTR_CHAN_NAME",
                                    "SALES_ORG_CD",
                                    "SHIP_TO_CUST_ID"
                                    FROM (
                                    SELECT *
                                        FROM NA_BI_VWS.CUSTOMER) AS "a") AS "Right" ON "Left"."SHIP_TO_CUST_ID" = "Right"."SHIP_TO_CUST_ID") AS "Left" INNER JOIN (
                            SELECT "MATL_ID",
                                "MKT_CTGY_MKT_AREA_NBR",
                                "PBU_NBR",
                                "SUPER_BRAND_ID"
                                FROM (
                                SELECT *
                                    FROM NA_BI_VWS.MATERIAL) AS "a") AS "Right" ON "Left"."MATL_ID" = "Right"."MATL_ID") AS "a"
                        WHERE "ACTL_GOODS_ISS_DT" >= '2011-02-01') AS "a"
                    WHERE "PBU_NBR" = '01'
                        AND  "SUPER_BRAND_ID" IN ('01','02','03')
                        AND  "SALES_ORG_CD" IN('N302', 'N312','N322')
                        AND  "MKT_CTGY_MKT_AREA_NBR" IN ('31','32','33')) AS "a") AS "a"
            GROUP BY "ACTL_GOODS_ISS_DT",
                "CUST_GRP_ID",
                "CUST_GRP_NAME" '''

    with pyodbc.connect(link, autocommit=True) as connect:
        df = pd.read_sql(query, connect)

    # pivit based on dist channels
    df = df.pivot_table('SHIPPED_QTY', index='SHIP_DT', columns='CUST_GRP_NAME',
                        aggfunc='sum', dropna=False, fill_value=0)
    df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_'). \
        str.replace('-', '_').str.replace('___', '_')
    df.index = pd.to_datetime(df.index)
    df = df.resample('D').sum()
    for col in df.columns:
        df[col][df[col] < 0.0] = 0.0
        if df[col].mean(axis=0) == 0.0:
            df.drop(columns=col, inplace=True)
    # combine DCs with low quantities or non-existence
    return df


#ds = direct_shipment_query()
#ds.to_excel('direct_shipment.xlsx')
s = pd.read_excel('direct_shipment.xlsx')
s.index=pd.to_datetime(s.SHIP_DT)
s.drop(columns=['SHIP_DT'],inplace=True)
main = ['CHRYSLER', 'FORD', 'GM', 'HYUNDAI', 'HONDA', 'NISSAN',
        'SIA', 'TESLA', 'TOYOTA', 'VW']
sides = ['OE_AUTO_GEN_CUST', 'OE_MOBILE_HOME', 'OE_NON_AUTO_GEN_CUST', 'TREDIT']
s = s[main]

sales = pd.read_excel('LVSALE.xlsx')
sales.index = pd.to_datetime(sales['Unnamed: 0'])
sales.drop(columns=['Unnamed: 0'], inplace=True)
sales = sales[sales.index >= s.index[0]]
sales.columns = sales.columns.str.upper()
sales.rename(columns={'VOLKSWAGEN': 'VW',
                      'SUBARU': 'SIA',
                      'GENERAL MOTORS': 'GM'}, inplace=True)
sales = sales[main]
sales_main = [name + str('_sale') for name in main]
sales.columns = sales_main

combo_not_scaled = pd.merge(np.log(sales.resample("M").sum() + 1),
                            np.log(s.resample("M").sum() + 1),
                            how='outer', left_index=True, right_index=True)

def prophet(df_train, df_test):
    model = Prophet(
        seasonality_mode="multiplicative",
        changepoint_prior_scale=25,
        holidays_prior_scale=30,
        seasonality_prior_scale=30,
        yearly_seasonality=False,
        daily_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        growth='logistic'
          ).add_seasonality(name='yearly',
          period=365.25,
          fourier_order=30
          ).add_seasonality(name='quarterly',
                            period=365.25 / 4,
                            fourier_order=25
                            # its 15, I just changed it to see the potential impact of quaretr end
                            )
    df_train['cap'] = max(df_train['y']) * 1.1
    model.add_regressor('GM_sale', mode="multiplicative", prior_scale=10)
    model.fit(df_train)
    forecast = model.make_future_dataframe(periods=len(df_test), include_history=False, freq='M')
    forecast['cap'] = max(df_train['y']) * 1.1
    forecast = pd.merge(forecast, df_test['GM_sale'],
                        left_on=forecast['ds'],
                        right_on=df_test.index)
    forecast = model.predict(forecast)
    forecast['yhat'] = np.exp(forecast['yhat'])
    return forecast

def prep_data_per_model(series, y_name, regs_name, lag_start, lag_end, cutoff_date, SM_start, SM_end):
    data = series.copy()
    data = data[:cutoff_date]
    X = data[regs_name]
    y = np.log(data[y_name])
    PRD_NAME = [name for name in regs_name if "_PRD" in name]
    X['PRD'] = np.log(X[PRD_NAME])
    X['PRD_LAG_1'] = X['PRD'].shift(1)
    X['PRD_LAG_2'] = X['PRD'].shift(2)
    X['PRD_LAG_3'] = X['PRD'].shift(3)
    X['GAS_PRICE'] = np.log(X.GAS_PRICE)
    X['GAS_PRICE_LAG1'] = X.GAS_PRICE.shift(1)
    for i in range(lag_start, lag_end + 1):
        X["LAG_{}".format(i)] = y.shift(i)
    for i in range(SM_start, SM_end + 1):
        X["ET_{}M".format(i)] = y.ewm(span=i, adjust=False).mean()

    X['TREND'] = y.ewm(span=12, adjust=False).mean()
    X = X.iloc[lag_end:, :]
    y = y.iloc[lag_end:]
    X_train = X.ix[:-1]
    y_train = y.ix[:-1]
    X_test = X.tail(1)
    y_test = y.tail(1)
    return X_train, X_test, y_train, y_test

holidays = pd.read_excel('holidays.xlsx')






GM = combo_not_scaled[['GM', 'GM_sale']]
GM['ds'] = GM.index
GM.rename(columns={'GM': 'y'}, inplace=True)
month=2
GM_train = GM[GM.index < datetime.strptime('2019-08', '%Y-%m')]
GM_test = GM[(GM.index.year == 2019) & (GM.index.month == 8)]
#GM_train.to_excel('GM_train.xlsx')
#GM_test.to_excel('GM_test.xlsx')
forecast = prophet(GM_train, GM_test)
GM_test['y'] = np.exp(GM_test['y'])

print('Forecast={}'.format(forecast['yhat'].values),' Actual={}'.format(GM_test['y'].values))
MAPE = (forecast['yhat'].values - GM_test[ 'y'].values) / GM_test[ 'y'].values
print(MAPE)
