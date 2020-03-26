import pandas as pd
from datetime import datetime, timedelta
import pyodbc
import os
import matplotlib.pyplot as plt
from dateutil import relativedelta
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge,BayesianRidge,HuberRegressor
from sklearn.pipeline import Pipeline

import joblib
from xgboost import XGBRegressor
from vecstack import StackingTransformer
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
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
    df = df.resample('MS').sum()
    for col in df.columns:
        df[col][df[col] < 0.0] = 0.0
        if df[col].mean(axis=0) == 0.0:
            df.drop(columns=col, inplace=True)
    # combine DCs with low quantities or non-existence
    return df


def events_holidays(link, date):
    date = shipment_main.index[0]
    query = '''Select CAL_DT,HLDY_DESC 
            From GDYR_BI_VWS.GDYR_HLDY_CURR                    
            order By CAL_DT
            where CAL_DT >= Cast('2011-01-01' As Date)'''
    with pyodbc.connect(link, autocommit=True) as connect:
        GY_Holiday = pd.read_sql(query, connect)
    GY_Holiday = GY_Holiday[GY_Holiday['HLDY_DESC'] != 'Sunday']
    df = pd.DataFrame({
        'holiday': GY_Holiday['HLDY_DESC'],
        'Date': pd.to_datetime(GY_Holiday['CAL_DT'])
    })
    df.index = pd.to_datetime(df.Date)
    df.drop(columns='Date', inplace=True)
    df = df.apply(lambda x: x.str.upper().str.replace('-', '').str.replace('  ', ' ').str.replace(' ', '_'))
    df = pd.get_dummies(df, prefix='is_', prefix_sep='')
    df = df.resample('MS').first()
    df = df[(df.index >= date) & (df.index < '2022-01-01')].replace(np.nan, 0)
    names = [name for name in df.columns if df[name].sum() != 0]
    df = df[names]
    return df


def get_sales(names, date):
    df = pd.read_excel('LVSALE.xlsx')
    df.index = pd.to_datetime(df['Unnamed: 0'])
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df = df[df.index >= date]
    df.columns = df.columns.str.upper()
    df.rename(columns={'VOLKSWAGEN': 'VW',
                       'SUBARU': 'SIA',
                       'GENERAL MOTORS': 'GM'}, inplace=True)
    df = df[names]
    df_main = [name + str('_sale') for name in names]
    df.columns = df_main
    df = df.resample('MS').sum()
    return df


def get_gas_price(date):
    df = pd.read_excel('US_GAS_Price.xlsx')
    df.index = pd.to_datetime(df.Month)
    df.drop(columns='Month', inplace=True)
    df = df[df.index >= date]
    df = df.resample('MS').sum()
    return df


def get_months_dummy(df):
    months = ['is_JAN', 'is_FEB', 'is_MAR', 'is_APR', 'is_MAY', 'is_JUN',
              'is_JUL', 'is_AUG', 'is_SEP', 'is_OCT', 'is_NOV', 'is_DEC']
    for i, month in enumerate(months):
        df[month] = np.where(df.index.month == i + 1, 1, 0)
    return df


def get_strike(df):
    df.loc[:, 'is_STRIKE'] = 0
    df.loc[(df.index.month == 9) & (df.index.year == 2019), 'is_STRIKE'] = 1
    df.loc[(df.index.month == 10) & (df.index.year == 2019), 'is_STRIKE'] = 1
    return df


def get_quarters(df):
    quarters = [1, 2, 3, 4]
    for i in quarters:
        df[str('is_Q{}'.format(i))] = np.where(df.index.quarter == i, 1, 0)
    return df

def get_stocks(date):
    GT_STOCK=pd.read_excel('GT_stock.xlsx')
    GT_STOCK.index=pd.to_datetime(GT_STOCK.Date)
    GT_STOCK=GT_STOCK['Volume'].ewm(span=180,adjust=False).mean()
    GT_STOCK=GT_STOCK[GT_STOCK.index>=date]
    GT_STOCK=GT_STOCK.resample('MS').mean()
    return GT_STOCK

def merger(long, short):
    '''long: contains forecast
        short:variable to be predicted'''
    combo = pd.merge(long, short, how='outer', left_index=True, right_index=True)
    return combo


def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """

    # get the index after which test set starts
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test

shipment = direct_shipment_query()
main = ['CHRYSLER', 'FORD', 'GM', 'HYUNDAI', 'HONDA', 'NISSAN',
        'SIA', 'TESLA', 'TOYOTA', 'VW']
sides = ['OE_AUTO_GEN_CUST', 'OE_MOBILE_HOME', 'OE_NON_AUTO_GEN_CUST', 'TREDIT']
shipment_main = shipment[main]

sales = get_sales(main, shipment_main.index[0])
gas_price = get_gas_price(shipment_main.index[0])
holidays = events_holidays(link, shipment_main.index[0])
combo = merger(sales, shipment_main)
combo = merger(combo, gas_price)
combo = merger(combo, holidays)
combo = get_months_dummy(combo)
# ---------------------------
dummies = ['is_', 'GAS', 'GM_sale']
regs_name = [name for name in combo.columns if any(x in name for x in dummies)]
GM = combo[regs_name]
GM = pd.merge(combo['GM'], GM, left_index=True, right_index=True)
from scipy import stats

GM.loc[:, 'GM'] = np.log(GM['GM'])
GM.loc[:, 'GM_sale'] = np.log(GM['GM_sale'])
GM.loc[:, 'GAS_PRICE'] = np.log(GM['GAS_PRICE'])
for i in range(1, 15):
    GM["LAG_{}".format(i)] = GM.GM.shift(i)
# -------------------------#

y = GM.dropna().GM
X = GM.dropna().drop(['GM'], axis=1)

# reserve 30% of data for testing
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
import scipy.stats as st
# for time-series cross-validation set 5 folds
tscv = TimeSeriesSplit(n_splits=5)
print('optmization')
params_grid = {
    'learning_rate': st.uniform(0.1, 0.5),
    'max_depth': list(range(4, 30, 2))
    }
skrg=XGBRegressor(**params_grid,objective='reg:squarederror',
                  tree_method= 'gpu_hist',random_state=0, n_jobs=-1)
search_skrg = RandomizedSearchCV(skrg, params_grid, cv=tscv) # 5 fold cross validation
search_skrg.fit(X_train, y_train)
joblib.dump(search_skrg.best_estimator_,'xgboos.pkl')
##----------------Extra Trees
params_Ex = {
    'n_estimators': list(range(100, 1000, 100)),
    'max_depth': list(range(4, 20, 2))}
EXrg=ExtraTreesRegressor(**params_Ex,random_state=0, n_jobs=-1)
search_EX = RandomizedSearchCV(EXrg, params_Ex, cv=tscv) # 5 fold cross validation
search_EX.fit(X_train, y_train)
joblib.dump(search_EX.best_estimator_,'ExTree.pkl')
##----------------RandomF
params_RF = {
    'n_estimators': list(range(100, 1000, 100)),
    'max_depth': list(range(4, 40, 2))}
RFrg=RandomForestRegressor(**params_RF,random_state=0, n_jobs=-1)
search_RF = RandomizedSearchCV(RFrg, params_RF, cv=tscv) # 5 fold cross validation
search_RF.fit(X_train, y_train)
joblib.dump(search_RF.best_estimator_,'RF.pkl')
params_Ridge = {
    'alpha': list(range(0.1, 10, 0.1))}
search_Ridge=Ridge(**params_Ridge,random_state=0)
search_Ridge = RandomizedSearchCV(search_Ridge, params_Ridge, cv=tscv) # 5 fold cross validation
search_Ridge.fit(X_train, y_train)
joblib.dump(search_Ridge.best_estimator_,'Ridge.pkl')
params_Lasso = {
    'alpha': list(range(0.1, 10, 0.1))}
search_Lasso=Lasso(**params_Lasso,random_state=0)
search_Lasso = RandomizedSearchCV(search_Lasso, params_Lasso, cv=tscv) # 5 fold cross validation
search_Lasso.fit(X_train, y_train)
joblib.dump(search_Lasso.best_estimator_,'Lasso.pkl')

print('optimization is done!')
# Caution! All estimators and parameter values are just
# demonstrational and shouldn't be considered as recommended.

# This is list of tuples
# Each tuple contains arbitrary unique name and estimator object
estimators_L1 = [
    ('lr',LinearRegression(n_jobs=-1)),
    ('hr',HuberRegressor()),
    ('lb',BayesianRidge()),
    ('rid',Ridge(**search_Ridge.best_params_,random_state=0)),
    ('lasso',Lasso(search_Lasso.best_params_,random_state=0)),
    ('svm',SVR(kernel='poly',degree=2)),
    ('kn', KNeighborsRegressor(n_jobs=-1, n_neighbors=4)),
    ('et', ExtraTreesRegressor(**search_EX.best_params_, random_state=0, n_jobs=-1)),

    ('rf', RandomForestRegressor(**search_RF.best_params_, random_state=0, n_jobs=-1)),

    ('xgb', XGBRegressor(**search_skrg.best_params_,objective='reg:squarederror',
                  tree_method= 'gpu_hist',random_state=0, n_jobs=-1))
]

stack = StackingTransformer(estimators=estimators_L1,   # base estimators
                            regression=True,            # regression task (if you need
                                                        #     classification - set to False)
                            variant='B',                # oof for train set, predict test
                                                        #     set in each fold and find mean
                            metric=mean_absolute_error, # metric: callable
                            n_folds=5,                  # number of folds
                            shuffle=False,               # shuffle the data
                            random_state=0,             # ensure reproducibility
                            verbose=1)

stack = stack.fit(X_train, y_train)
S_train = stack.transform(X_train)
S_test = stack.transform(X_test)
final_estimator = XGBRegressor(random_state=0, n_jobs=-1, learning_rate=0.4,tree_method= 'gpu_hist',
                         objective='reg:squarederror',n_estimators=300, max_depth=18)
final_estimator = final_estimator.fit(S_train, y_train)

# Predict
y_pred = final_estimator.predict(S_test)

# Final prediction score
print('Final prediction score: [%.8f]' % mean_absolute_error(y_test, y_pred))
# Specify steps of Pipeline
steps = [('stack', stack),
         ('final_estimator', final_estimator)]

# Init Pipeline
pipe = Pipeline(steps)

pipe = pipe.set_params(stack__verbose=2)

# Fit
pipe = pipe.fit(X_train, y_train)

# Predict
y_pred_pipe = pipe.predict(X_test)

# Final prediction score
print('Final prediction score using Pipeline: [%.8f]' % mean_absolute_error(np.exp(y_test.values),np.exp(y_pred_pipe)) )
plt.plot(np.exp(y_pred_pipe))
plt.plot(np.exp(y_test.values),'ko')
plt.show()


