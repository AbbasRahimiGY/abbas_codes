import calendar
import pandas as pd
from datetime import datetime, timedelta
import pyodbc
import numpy as np
import os, warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from fbprophet import Prophet
from pyearth import Earth
import lightgbm as lgb
from xgboost import plot_importance
from multiprocessing import cpu_count
from joblib import Parallel, delayed

os.environ['NUMEXPR_NUM_THREADS'] = '8'
warnings.simplefilter('ignore')
# ---------------------------------------------------#
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
os.chdir(r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast')
# Make a connection
link = ('DSN=EDWTDPRD;UID=AA68383;PWD=baradarkhobvaghashang1364')
pyodbc.pooling = False


# ---------------------------------------------------#
class Models:

    def __init__(self, df_train, df_test, exogenous_features, scale_list=None):
        if scale_list is None:
            scale_list = []
        if len(scale_list) > 0:
            for col in scale_list:
                df_train.loc[df_train[col] < 0, col] = 0
                df_test.loc[df_test[col] < 0, col] = 0
                df_train[col] = np.log(df_train[col] + 1)
                df_test[col] = np.log(df_test[col] + 1)

        self.train = df_train
        self.test = df_test
        self.scale_list = scale_list
        self.features = exogenous_features
        self.X_train = df_train[exogenous_features]
        self.X_test = df_test[exogenous_features]
        self.y_train = df_train['y']

    def poly_linear(self):
        model = Lasso(alpha=1e-2)  # ElasticNet(alpha=1e-2,l1_ratio=0.7)#Lasso(alpha=1e-2)
        poly = PolynomialFeatures(2)
        X_poly_train = poly.fit_transform(self.X_train)
        # Fit
        model = model.fit(X_poly_train, self.y_train)
        X_poly_test = poly.fit_transform(self.X_test)
        # Predict
        forecast = model.predict(X_poly_test)
        if forecast < 0:
            forecast[0] = 0
        if len(self.scale_list) > 0:
            forecast[0] = np.exp(forecast[0])
        return np.round(forecast.item(), 0)

    def mars(self, max_degree=2):
        model = Earth(max_degree=max_degree, allow_missing=True, enable_pruning=True,
                      minspan_alpha=.5, thresh=.001, smooth=False, verbose=False)

        model = model.fit(self.X_train, self.y_train)
        forecast = model.predict(self.X_test)
        if forecast < 0:
            forecast[0] = 0
        if len(self.scale_list) > 0:
            forecast[0] = np.exp(forecast[0])
        return np.round(forecast.item(), 0)

    def prophet(self, cp_scale=0.01):
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

        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
                        changepoint_prior_scale=cp_scale, seasonality_mode='multiplicative')
        for feature in self.features:
            model.add_regressor(feature, prior_scale=0.5)
        with suppress_stdout_stderr():
            model.fit(self.train[["ds", "y"] + self.features])
        forecast = model.predict(self.test[["ds"] + self.features])
        forecast.loc[forecast.yhat < 0, "yhat"] = 0
        if len(self.scale_list) > 0:
            forecast = np.exp(forecast["yhat"].item())
        else:
            forecast = forecast["yhat"].item()

        return np.round(forecast, 0)

    def xgb(self):
        min_X = self.X_train.min()
        max_X = self.X_train.max()
        X_train_sc=(self.X_train-min_X)/(max_X-min_X)
        X_test_sc = (self.X_test - min_X) / (max_X - min_X)
        model = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.15,
                             max_depth=5, alpha=10, n_estimators=200)
        model = model.fit(X_train_sc, self.y_train)
        # Predict
        forecast = model.predict(X_test_sc)
        if forecast < 0:
            forecast[0] = 0

        if len(self.scale_list) > 0:
            forecast[0] = np.exp(forecast[0])
        return np.round(forecast.item(), 0)


# all models for parallel runs
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


def poly_linear(df_train, df_test, exogenous_features, scale_list=None):
    if scale_list is None:
        scale_list = []
    if len(scale_list) > 0:
        for col in scale_list:
            df_train.loc[df_train[col] < 0, col] = 0
            df_test.loc[df_test[col] < 0, col] = 0
            df_train[col] = np.log(df_train[col] + 1)
            df_test[col] = np.log(df_test[col] + 1)

    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']
    model = Lasso(alpha=1e-2)  # ElasticNet(alpha=1e-2,l1_ratio=0.7)#Lasso(alpha=1e-2)
    poly = PolynomialFeatures(2)
    X_poly_train = poly.fit_transform(X_train)
    # Fit
    model = model.fit(X_poly_train, y_train)
    X_poly_test = poly.fit_transform(X_test)
    # Predict
    forecast = model.predict(X_poly_test)
    if forecast < 0:
        forecast[0] = 0
    if len(scale_list) > 0:
        forecast[0] = np.exp(forecast[0])
    return np.round(forecast.item(), 0)


def xgb(df_train, df_test, exogenous_features, scale_list=None):
    if scale_list is None:
        scale_list = []
    if len(scale_list) > 0:
        for col in scale_list:
            df_train.loc[df_train[col] < 0, col] = 0
            df_test.loc[df_test[col] < 0, col] = 0
            df_train[col] = np.log(df_train[col] + 1)
            df_test[col] = np.log(df_test[col] + 1)

    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']
    model = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.15,
                         max_depth=5, alpha=10, n_estimators=200)
    model = model.fit(X_train, y_train)
    # Predict
    forecast = model.predict(X_test)
    '''fig1, ax = plt.subplots(figsize=(12, 8))
    plot_importance(model, max_num_features=10, ax=ax)
    plt.axis('tight')
    fig1.savefig('xgboost.png')'''
    if forecast < 0:
        forecast[0] = 0
    if len(scale_list) > 0:
        forecast[0] = np.exp(forecast[0])
    return np.round(forecast.item(), 0)


def xgb_old(df_train, df_test, exogenous_features, scale_list=None):
    if scale_list is None:
        scale_list = []
    if len(scale_list) > 0:
        for col in scale_list:
            df_train.loc[df_train[col] < 0, col] = 0
            df_test.loc[df_test[col] < 0, col] = 0
            df_train[col] = np.log(df_train[col] + 1)
            df_test[col] = np.log(df_test[col] + 1)

    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']

    tscv = TimeSeriesSplit(n_splits=4)
    # model 1 optimization
    xgb_grid = {
        'min_child_weight': [1, 3, 6],
        'max_depth': [5, 10, 15, 20, 30, 40]
    }
    xgb_model = XGBRegressor(**xgb_grid, objective='reg:squarederror', n_estimators=100, colsample_bytree=0.3,
                             learning_rate=0.15, random_state=0, n_jobs=-1, alpha=10)
    grid = GridSearchCV(xgb_model, xgb_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=0)
    grid.fit(X_train, y_train)
    # Predict
    forecast = grid.predict(X_test)
    fig1, ax = plt.subplots(figsize=(12, 8))
    plot_importance(grid.best_estimator_, max_num_features=10, ax=ax)
    plt.axis('tight')
    fig1.savefig('xgboost.png')
    if forecast < 0:
        forecast[0] = 0
    if len(scale_list) > 0:
        forecast[0] = np.exp(forecast[0])
    return np.round(forecast.item(), 0)


def mars(df_train, df_test, exogenous_features, scale_list=None, max_degree=2):
    if scale_list is None:
        scale_list = []
    if len(scale_list) > 0:
        for col in scale_list:
            df_train.loc[df_train[col] < 0, col] = 0
            df_test.loc[df_test[col] < 0, col] = 0
            df_train[col] = np.log(df_train[col] + 1)
            df_test[col] = np.log(df_test[col] + 1)

    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']
    model = Earth(max_degree=max_degree, allow_missing=True, enable_pruning=True,
                  minspan_alpha=.5, thresh=.001, smooth=False, verbose=False)
    model = model.fit(X_train, y_train)
    # Predict
    forecast = model.predict(X_test)
    if forecast < 0:
        forecast[0] = 0

    if len(scale_list) > 0:
        forecast[0] = np.exp(forecast[0])
    return np.round(forecast.item(), 0)


def model_stack(df_train, df_test, exogenous_features, scale_list=None):
    if scale_list is None:
        scale_list = []
    if len(scale_list) > 0:
        for col in scale_list:
            df_train.loc[df_train[col] < 0, col] = 0
            df_test.loc[df_test[col] < 0, col] = 0
            df_train[col] = np.log(df_train[col] + 1)
            df_test[col] = np.log(df_test[col] + 1)

    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']
    # splitting data
    training, valid, ytraining, yvalid = timeseries_train_test_split(X_train, y_train, test_size=0.25)

    model1 = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.15,
                          max_depth=5, alpha=10, n_estimators=100)
    # poly features
    poly = PolynomialFeatures(3)
    lr = Lasso(alpha=1e-2)
    model2 = Pipeline([("polynomial_features", poly),
                       ("linear_regression", lr)])

    # fit models
    model1.fit(training, ytraining)
    model2.fit(training, ytraining)

    # make prediction for validation
    preds1 = model1.predict(valid)
    preds2 = model2.predict(valid)

    # make prediction for test data
    test_preds1 = model1.predict(X_test)
    test_preds2 = model2.predict(X_test)

    # form a new dataset for valid and test via stacking the predictions
    stacked_predictions = np.column_stack((preds1, preds2))
    stacked_test_predictions = np.column_stack((test_preds1, test_preds2))
    # specify meta model
    meta_model = LinearRegression()
    # fit meta model on stacked predictions
    meta_model.fit(stacked_predictions, yvalid)
    forecast = meta_model.predict(stacked_test_predictions)
    if forecast < 0:
        forecast[0] = 0
    if len(scale_list) > 0:
        forecast[0] = np.exp(forecast[0])
    return np.round(forecast.item(), 0)


def cb(df_train, df_test, exogenous_features, scale_list=None):
    if scale_list is None:
        scale_list = []
    if len(scale_list) > 0:
        for col in scale_list:
            df_train.loc[df_train[col] < 0, col] = 0
            df_test.loc[df_test[col] < 0, col] = 0
            df_train[col] = np.log(df_train[col] + 1)
            df_test[col] = np.log(df_test[col] + 1)

    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']
    model = CatBoostRegressor(iterations=100,
                              learning_rate=0.15,
                              depth=5, verbose=0)
    model = model.fit(X_train, y_train)
    # Predict
    forecast = model.predict(X_test)
    if forecast < 0:
        forecast[0] = 0

    if len(scale_list) > 0:
        forecast[0] = np.exp(forecast[0])
    return np.round(forecast.item(), 0)


def lgbm(df_train, df_test, exogenous_features, scale_list=None):
    if scale_list is None:
        scale_list = []
    if len(scale_list) > 0:
        for col in scale_list:
            df_train.loc[df_train[col] < 0, col] = 0
            df_test.loc[df_test[col] < 0, col] = 0
            df_train[col] = np.log(df_train[col] + 1)
            df_test[col] = np.log(df_test[col] + 1)

    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']
    training, valid, ytraining, yvalid = timeseries_train_test_split(X_train, y_train, test_size=0.15)
    lgb_train = lgb.Dataset(training, ytraining)
    lgb_eval = lgb.Dataset(valid, yvalid, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 511,
        'learning_rate': 0.15,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 10,
        'verbose': -1
    }

    # train
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=20,
                      valid_sets=lgb_eval,
                      early_stopping_rounds=5, verbose_eval=False)
    # predict
    forecast = model.predict(X_test, num_iteration=model.best_iteration)

    if forecast < 0:
        forecast[0] = 0

    if len(scale_list) > 0:
        forecast[0] = np.exp(forecast[0])
    return np.round(forecast.item(), 0)


def prophet(df_train, df_test, exogenous_features, scale_list=None, cp_scale=0.05):
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

    if scale_list is None:
        scale_list = []
    if len(scale_list) > 0:
        for col in scale_list:
            df_train.loc[df_train[col] < 0, col] = 0
            df_test.loc[df_test[col] < 0, col] = 0
            df_train[col] = np.log(df_train[col] + 1)
            df_test[col] = np.log(df_test[col] + 1)

    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
                    changepoint_prior_scale=cp_scale, seasonality_mode='multiplicative')
    for feature in exogenous_features:
        model.add_regressor(feature)
    with suppress_stdout_stderr():
        model.fit(df_train[["ds", "y"] + exogenous_features])
    forecast = model.predict(df_test[["ds"] + exogenous_features])
    forecast.loc[forecast.yhat < 0, "yhat"] = 0
    if len(scale_list) > 0:
        forecast = np.exp(forecast["yhat"].item())
    else:
        forecast = forecast["yhat"].item()

    return np.round(forecast, 0)


# end of models

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


def prep_data(data, holiday):
    '''
    current columns
    ['index', 'SNAP_DT', 'CUSTOMER', 'CON', 'IP', 'SHIP_QTY', 'day', 'month',
       'num_day_in_month', 'day_left', 'tot_ship', 'stright_line',
       'tot_ship_time_day', 'working', 'final_week', 'final_week_ip',
       'final_week_con', 'Christmas', 'Christmas Eve', 'Easter',
       'Independence Day', 'Labor Day', 'Mday', 'New Year Eve', 'NY',
       'Thanksgiving', 'Apr', 'Aug', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May',
       'Nov', 'Oct', 'Sep', 'Fri', 'Mon', 'Sat', 'Thu', 'Tue', 'Wed']
    :param data:
    :param holiday:
    :return:
    '''
    df = data.copy()
    if df.index.dtype != 'Datetime64':
        df = df.set_index('PLN_DEL_DT')
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
    # print(df.columns[df.isnull().any()])
    df = pd.concat([df, pd.get_dummies(df['month'])], axis=1)

    df = pd.concat([df, pd.get_dummies(df['weekday'])], axis=1)
    df.drop(columns=['day', 'month', 'Sun', 'year', 'Dec', 'weekday'], inplace=True)
    return df


def data_resampling(df, df_shipment, current):
    '''
    Perform resampling on result of several models fit into a normal distrubution
    :param df_shipment:
    :param current:
    :param df:
    :return:
    '''
    current_month = datetime.strftime(datetime.strptime(current, '%Y-%m-%d'), '%Y-%m')
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
    # now merge with current month shipment
    df = df.groupby('CUSTOMER')['forecast_low', 'forecast_mean', 'forecast_high'].sum()
    mtd_ship = df_shipment.copy()
    mtd_ship.set_index('SNAP_DT', inplace=True)
    mtd_ship = mtd_ship[current_month].groupby('CUSTOMER').resample('M').sum()
    df = pd.merge(df, mtd_ship, on=['CUSTOMER'])
    df['forecast_mean'] = df.loc[:, ['forecast_mean', 'SHIP_QTY']].sum(axis=1).astype('int')
    df['forecast_low'] = df.loc[:, ['forecast_low', 'SHIP_QTY']].sum(axis=1).astype('int')
    df['forecast_high'] = df.loc[:, ['forecast_high', 'SHIP_QTY']].sum(axis=1).astype('int')
    df.drop(columns='SHIP_QTY', inplace=True)
    df.to_excel('monthly_forecast_asof_{}.xlsx'.format(current))


if __name__ == '__main__':
    # -------call input here
    holidays = events_holidays_update(link)
    shipment = direct_shipment_query(link)
    active_CUSTOMER = shipment.CUSTOMER.unique().tolist()
    # orders = prep_order_data(active_CUSTOMER)
    orders = pd.read_pickle('historical_orders\historical_order_Rachael_filled_missing.pickle')
    type = 'parallel'
    # -----------------------
    exogenous_features = ['CON', 'IP', 'past_seven', 'past_twentyeight','past_two','yday_gain',
                          'Apr', 'Aug', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May',
                          'Nov', 'Oct', 'Sep'
                          ]

    scale_list = ['y']

    max_terms = len(exogenous_features)
    cutoff = datetime.strftime(datetime.today() - timedelta(days=1), "%Y-%m-%d")  # where to cutoff data
    test = datetime.strftime(datetime.today() - timedelta(days=0),
                             "%Y-%m-%d")  # where we sample from snap date for all horizons
    # find the horizon to search
    horizon = calendar.monthrange(datetime.strptime(test, "%Y-%m-%d").year,
                                    datetime.strptime(test, "%Y-%m-%d").month)[1] - \
                                    datetime.strptime(test, "%Y-%m-%d").day + 1
    # temp horizon
    horizon = np.int((pd.to_datetime('2020-04-30') - pd.to_datetime(test)) / np.timedelta64(1, 'D'))
    result = []
    for lead in range(1, horizon):
        lead_date = datetime.strftime(datetime.strptime(test, '%Y-%m-%d') + timedelta(lead), '%Y-%m-%d')
        print(lead_date)
        lead_order = orders[(orders.PLN_DEL_DT - orders.SNAP_DT).dt.days == lead].copy()
        lead_order = lead_order.merge(shipment, on=['SNAP_DT', 'CUSTOMER'], how='left').fillna(0)
        lead_order = prep_data(lead_order, holidays)

        if type == 'parallel':
            dict_reg = {'XgBoost': xgb,
                        'Poly': poly_linear,
                        'Prophet': prophet,
                        'Spline': mars,
                        'Stacked': model_stack,
                        'CatBoost': cb,
                        'LightBoost': lgbm
                        }
            series_train = [lead_order.loc[(lead_order.SNAP_DT <= cutoff) & (lead_order.CUSTOMER == cus)]. \
                                rename(columns={'SHIP_QTY': 'y', 'PLN_DEL_DT': 'ds'}) for cus in
                            lead_order.CUSTOMER.unique()]
            series_valid = [lead_order.loc[(lead_order.SNAP_DT == test) & (lead_order.CUSTOMER == cus)]. \
                                rename(columns={'SHIP_QTY': 'y', 'PLN_DEL_DT': 'ds'}) for cus in
                            lead_order.CUSTOMER.unique()]
            # define executor
            executor = Parallel(n_jobs=8, backend='multiprocessing')
            for key, reg in dict_reg.items():
                tasks = (delayed(reg)(df_train, df_valid, exogenous_features) for df_train, df_valid in
                         zip(series_train, series_valid))
                forecast = executor(tasks)

                day_forecast = pd.DataFrame({'CUSTOMER': lead_order.CUSTOMER.unique().tolist(),
                                             'Forecast': forecast, 'Date': lead_date, 'model': key})
                result.append([day_forecast])

        else:
            y_hat = 0
            for cus in orders.CUSTOMER.unique():

                df_train = lead_order.loc[(lead_order.SNAP_DT <= cutoff) & (lead_order.CUSTOMER == cus)]. \
                    rename(columns={'SHIP_QTY': 'y', 'PLN_DEL_DT': 'ds'})
                df_valid = lead_order.loc[(lead_order.SNAP_DT == test) & (lead_order.CUSTOMER == cus)]. \
                    rename(columns={'SHIP_QTY': 'y', 'PLN_DEL_DT': 'ds'})
                my_model = Models(df_train, df_valid, exogenous_features, scale_list=None)
                yhat_cus = my_model.xgb()  # mars(max_degree=1)
                if yhat_cus > 0:
                    y_hat += yhat_cus
            print(lead_date, y_hat)
    forecast_total = pd.concat([pd.concat(x) for x in result])
    forecast_total.to_excel('results.xlsx')
    summary_table = pd.pivot_table(forecast_total, values='Forecast', index=['Date', 'CUSTOMER'], columns='model')
    # perform resampling from normal distribution
    data_resampling(summary_table, shipment, test)
