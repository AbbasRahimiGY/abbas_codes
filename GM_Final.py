import pandas as pd
from datetime import datetime, timedelta
import pyodbc
import os
import matplotlib.pyplot as plt
from dateutil import relativedelta
import seaborn as sns
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from pandas.core.common import SettingWithCopyWarning
import warnings
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
import scipy.stats as st
warnings.simplefilter('ignore')
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('fivethirtyeight')
os.chdir(r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast')
# Make a connection
link = ('DSN=EDWTDPRD;UID=AA68383;PWD=baradarkhobvaghashang1364')
pyodbc.pooling = False


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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
    df = df.resample('M').sum()
    for col in df.columns:
        df[col][df[col] < 0.0] = 0.0
        if df[col].mean(axis=0) == 0.0:
            df.drop(columns=col, inplace=True)
    # combine DCs with low quantities or non-existence
    return df


def events_holidays(link, date):
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
    df = df.resample('M').first()
    df = df[(df.index >= date) & (df.index < '2022-01-01')].replace(np.nan, 0)
    names = [name for name in df.columns if df[name].sum() != 0]
    df = df[names]
    return df


def data_prep():
    def merger(long, short):
        '''long: contains forecast
        short:variable to be predicted'''
        combo = pd.merge(long, short, how='outer', left_index=True, right_index=True)
        return combo

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
        df_main = [name + str('_PRD') for name in names]
        df.columns = df_main
        df = df.resample('M').sum()
        return df

    def get_gas_price(date):
        df = pd.read_excel('US_GAS_Price.xlsx')
        df.index = pd.to_datetime(df.Month)
        df.drop(columns='Month', inplace=True)
        df = df[df.index >= date]
        df = df.resample('M').sum()
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


def model_stack(X_train, y_train, X_test):
    # splitting data
    training, valid, ytraining, yvalid = timeseries_train_test_split(X_train, y_train, test_size=0.5)
    tscv = TimeSeriesSplit(n_splits=4)
    # model 1 optimization
    xgb_grid = {
        'n_estimators': list(range(100, 500, 50)),
        'learning_rate': st.uniform(0.1, 0.5),
        'max_depth': list(range(4, 30, 2)),
        'min_child_weight': list(range(0, 100, 1))
    }
    xgb_model = XGBRegressor(**xgb_grid, objective='reg:squarederror', random_state=0, n_jobs=-1)
    model1 = RandomizedSearchCV(xgb_model, xgb_grid, cv=tscv,scoring='neg_mean_squared_error')
    # model 2 optimization
    rf_grid = {
        'max_features': list(range(2, 20, 1)),
        'max_depth': list(range(4, 30, 2)),
        'max_leaf_nodes': list(range(2, 100, 1)),
        'min_samples_leaf': list(range(2, 100, 1))
    }
    rf_model = RandomForestRegressor(**rf_grid, n_jobs=-1)
    model2 = RandomizedSearchCV(rf_model, rf_grid, cv=tscv, scoring='neg_mean_squared_error')

    lasso_grid = {
        'alpha': st.uniform(0.00001, 1)
    }
    lasso_model = Lasso(**lasso_grid)
    model3 = RandomizedSearchCV(lasso_model, lasso_grid, cv=tscv,scoring='neg_mean_squared_error')
    model4 = LinearRegression()

    # poly features
    poly = PolynomialFeatures(3)
    X_poly_train = poly.fit_transform(training)
    X_poly_valid = poly.fit_transform(valid)
    X_poly_test = poly.fit_transform(X_test)

    # fit models
    model1.fit(training, ytraining)
    model2.fit(training, ytraining)
   # model3.fit(X_poly_train, ytraining)
    #model4.fit(training, ytraining)
    # make prediction for validation
    preds1 = model1.predict(valid)
    preds2 = model2.predict(valid)
   # preds3 = model3.predict(X_poly_valid)
    #preds4 = model4.predict(valid)
    # make prediction for test data
    test_preds1 = model1.predict(X_test)
    test_preds2 = model2.predict(X_test)
    #test_preds3 = model3.predict(X_poly_test)
    #test_preds4 = model4.predict(X_test)
    # form a new dataset for valid and test via stacking the predictions
    stacked_predictions = np.column_stack((preds1, preds2))
    stacked_test_predictions = np.column_stack((test_preds1, test_preds2))
    # specify meta model
    meta_model = LinearRegression()
    # fit meta model on stacked predictions
    meta_model.fit(stacked_predictions, yvalid)
    final_predictions = meta_model.predict(stacked_test_predictions)

    return final_predictions


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


def prep_data_lag0(series, y_name, regs_name, lag_start, lag_end, cutoff_date, SM_start, SM_end):
    data = series.copy()
    ts_end = cutoff_date + pd.offsets.MonthEnd(1)
    data = data[:ts_end]
    X = data[regs_name]
    y = np.log(data[y_name])
    PRD_NAME = [name for name in regs_name if "_PRD" in name]
    # convert to log the data and lagging
    X['PRD'] = np.log(X[PRD_NAME])
    X['GAS_PRICE'] = np.log(X.GAS_PRICE)
    X['GAS_PRICE_LAG1'] = X.GAS_PRICE.shift(1)
    for i in range(lag_start, lag_end + 1):
        X["LAG_{}".format(i)] = y.shift(i)
    for i in range(SM_start, SM_end + 1):
        X["ET_{}M".format(i)] = y.ewm(span=i, adjust=False).mean()
    for i in range(lag_start, lag_end + 1):
        X["PRD_LAG_{}".format(i)] = X['PRD'].shift(i)

    X['TREND_min'] = y.rolling(3).min()
    X['TREND_max'] = y.rolling(3).max()
    X['TREND_median'] = y.rolling(3).median()
    X['3Months_AVG'] = y.rolling(3).mean()
    # forecast month
    # make sure we begin ts after lag end
    X = X.iloc[lag_end+1:, :]
    y = y.iloc[lag_end+1:]
    X_train = X[:-1]
    y_train = y[:-1]
    X_test = X.tail(1)
    return X_train, X_test, y_train


customers = ['CHRYSLER', 'FORD', 'GM', 'HONDA', 'TOYOTA', 'NISSAN']
combo = data_prep()


def back_testing():
    table = {}
    for customer in customers:
        dummies = ['is_', 'GAS', str('{}_PRD').format(customer)]
        regs_name = [name for name in combo.columns if any(x in name for x in dummies)]
        dates = pd.date_range(start='2019-01-01', end='2019-11-1', freq='MS')
        combo_copy = combo.copy()
        combo_copy.loc['2019-03-01':, customer] = np.nan

        error, time = [], []

        for i in range(len(dates) - 1):
            X_train, X_test, y_train, _ = prep_data_per_model(combo_copy, customer, regs_name, lag_start=1, lag_end=12,
                                                              cutoff_date=dates[i + 1], SM_start=1, SM_end=10)
            y_pred = model_stack(X_train, y_train, X_test)
            combo_copy.loc[dates[i + 1]:, customer] = np.round(np.exp(y_pred), 0)
            y_test = combo.loc[dates[i + 1], customer]
            err = mean_absolute_percentage_error(y_test, np.exp(y_pred))
            error.append(np.round(err, 2))
            time.append(dates[i + 1].strftime('%y-%m-%d'))
        chunck = {'Date': time, 'MAPE': error}
        table[customer] = chunck
        print('Forecasting {} is completed!'.format(customer))

    df = pd.DataFrame([[i, j, name] for name in table.keys() for i, j in zip(table[name]['MAPE'], table[name]['Date'])])
    df.columns = ['MAPE', 'Date', 'customer']
    df.set_index('Date', inplace=True)
    fig, ax = plt.subplots(figsize=(16, 4))
    df[(df.index != '19-10-01') & (df.index != '19-09-01')].groupby(['Date', 'customer'])['MAPE']. \
        mean().unstack().plot(kind='line', ax=ax, logy=False, cmap='jet')
    ax.legend(loc=(0.1, -0.5), ncol=5)
    plt.show()

def model_poly(X_train, y_train,X_test):
    from sklearn.model_selection import GridSearchCV
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = [{'alpha':[0.0001,0.001,0.01,0.02,0.05,0.09,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]}]
    param_grid = [{'n_estimators': list(range(100, 500, 100)),
                   'learning_rate': np.linspace(0.01,0.5,20).tolist(),
                   'min_child_weight=': list(range(1, 20, 1))}]
    model=XGBRegressor(n_estimators=500, objective='reg:squarederror') #, booster='gblinear',feature_selector='shuffle', eval_metric='mae'
    # Fit
   # model=Lasso()
    grid_search=GridSearchCV(model, param_grid,cv=tscv,scoring='neg_mean_squared_error',return_train_score=True)
    grid_search.fit(X_train, y_train)
    # Predict
    y_pred = grid_search.predict(X_test)
    return y_pred,grid_search

def lag0_forecast():
    lag0_table = {}
    for customer in customers:
        dummies = ['is_', 'GAS', str('{}_PRD').format(customer)]
        regs_name = [name for name in combo.columns if any(x in name for x in dummies)]
        cutoff_date = pd.datetime.date(datetime.now()) - pd.offsets.MonthEnd(1)
        X_train, X_test, y_train = prep_data_lag0(combo, customer, regs_name, lag_start=1, lag_end=3,
                                                  cutoff_date=cutoff_date, SM_start=1, SM_end=6)
        model=joblib.load('models\{}.pkl'.format(customer))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        chunck = {'Date': X_test.index[0].date().strftime('%y-%m'), 'Forecast': np.round(np.exp(y_pred)/1000,0).flatten()}
        lag0_table[customer] = chunck
        print('Forecasting {} is completed!'.format(customer))

    pd.DataFrame.from_dict(lag0_table).to_excel('lag0_forcas.xlsx')
    return


lag0_forecast()
