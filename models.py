from catboost import CatBoostRegressor
from fbprophet import Prophet
from pyearth import Earth
from sklearn.metrics import max_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import numpy as np
import lightgbm as lgb
import os, warnings


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
        model = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.15,
                             max_depth=5, alpha=10, n_estimators=200)
        model = model.fit(self.X_train, self.y_train)
        # Predict
        forecast = model.predict(self.X_test)
        if forecast < 0:
            forecast[0] = 0

        if len(self.scale_list) > 0:
            forecast[0] = np.exp(forecast[0])
        return np.round(forecast.item(), 0)


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


def poly_linearCV(df_train, df_test, exogenous_features):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast
    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']
    tscv = TimeSeriesSplit(n_splits=3)
    # model 1 optimization
    param_grid = {
        'fit__alpha': [0.1, 0.01, 0.001]
    }
    poly = PolynomialFeatures(2)

    grid_model = Pipeline([("polynomial_features", poly),
                           ("fit", Lasso())])
    model = GridSearchCV(grid_model, param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=0)
    # Fit & predict
    forecast = model.fit(X_train, y_train).best_estimator_.predict(X_test)

    if forecast < 0:
        forecast[0] = 0
    return np.round(forecast.item(), 0)


def xgb(df_train, df_test, exogenous_features):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast
    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']
    model = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.15,
                         max_depth=5, alpha=10, n_estimators=200)
    model = model.fit(X_train, y_train)
    # Predict
    forecast = model.predict(X_test)
    if forecast < 0:
        forecast[0] = 0
    return np.round(forecast.item(), 0)


def xgbCV(df_train, df_test, exogenous_features):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast
    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']

    tscv = TimeSeriesSplit(n_splits=3)
    # model 1 optimization
    xgb_grid = {
        'min_child_weight': [1, 3, 6],
        'max_depth': [5, 10]
    }
    xgb_model = XGBRegressor(**xgb_grid, objective='reg:squarederror', n_estimators=200, colsample_bytree=0.3,
                             learning_rate=0.15, random_state=0, n_jobs=-1, alpha=10)
    grid = GridSearchCV(xgb_model, xgb_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=0)
    grid.fit(X_train, y_train)
    # Predict
    forecast = grid.best_estimator_.predict(X_test)
    if forecast < 0:
        forecast[0] = 0
    return np.round(forecast.item(), 0)


def cbCV(df_train, df_test, exogenous_features):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast
    X_train = df_train[exogenous_features]
    X_test = df_test[exogenous_features]
    y_train = df_train['y']

    # model 1 optimization
    param_grid = {
        'l2_leaf_reg': [1, 2, 3],
        'depth': [5, 10]
    }
    grid_model = CatBoostRegressor(iterations=100, learning_rate=0.15)
    grid = grid_model.grid_search(param_grid, X=X_train, y=y_train, shuffle=True,
                                  cv=3, verbose=0, refit=False)
    # Predict

    model = CatBoostRegressor(verbose=0, depth=grid['params']['depth'], iterations=100,
                              l2_leaf_reg=grid['params']['l2_leaf_reg'], learning_rate=0.15)
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)
    if forecast < 0:
        forecast[0] = 0

    return np.round(forecast.item(), 0)


def prophetCV(df_train, df_test, exogenous_features):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast

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

    # perform a mini grid search
    param_grid = [0.01, 0.05, 0.1]
    scores = []
    for cp_scale in param_grid:
        # split data in few different chuncks
        training, valid, _, __ = timeseries_train_test_split(df_train, df_test, test_size=0.15)
        grid_model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
                             changepoint_prior_scale=cp_scale, seasonality_mode='multiplicative')
        for feature in exogenous_features:
            grid_model.add_regressor(feature)
        with suppress_stdout_stderr():
            forecast = grid_model.fit(training[["ds", "y"] + exogenous_features]). \
                predict(valid[["ds"] + exogenous_features])
            error = max_error(valid['y'].values, forecast['yhat'].values)
            scores.append([cp_scale, error])
    scores.sort(key=lambda tup: tup[1])
    # done with grid search
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
                    changepoint_prior_scale=scores[0][0], seasonality_mode='multiplicative')
    for feature in exogenous_features:
        model.add_regressor(feature)
    with suppress_stdout_stderr():
        model.fit(df_train[["ds", "y"] + exogenous_features])
    forecast = model.predict(df_test[["ds"] + exogenous_features])
    forecast.loc[forecast.yhat < 0, "yhat"] = 0
    forecast = forecast["yhat"].item()

    return np.round(forecast, 0)


def mars(df_train, df_test, exogenous_features, max_degree=2):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast
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
    return np.round(forecast.item(), 0)


def model_stack(df_train, df_test, exogenous_features):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast
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
    return np.round(forecast.item(), 0)


def lgbm(df_train, df_test, exogenous_features):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast
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

    return np.round(forecast.item(), 0)


def prophet(df_train, df_test, exogenous_features, cp_scale=0.05):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast

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
    for feature in exogenous_features:
        model.add_regressor(feature)
    with suppress_stdout_stderr():
        model.fit(df_train[["ds", "y"] + exogenous_features])
    forecast = model.predict(df_test[["ds"] + exogenous_features])
    forecast.loc[forecast.yhat < 0, "yhat"] = 0
    forecast = forecast["yhat"].item()

    return np.round(forecast, 0)


def cb(df_train, df_test, exogenous_features):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast
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
    return np.round(forecast.item(), 0)


def poly_linear(df_train, df_test, exogenous_features):
    if (df_test['IP'].values == 0) & (df_test['CON'].values == 0):
        forecast = 0
        return forecast
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
    return np.round(forecast.item(), 0)
