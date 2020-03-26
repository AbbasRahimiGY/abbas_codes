import pandas as pd
import numpy as np
from fbprophet import Prophet
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
import logging
import os
from datetime import datetime, timedelta
import scipy.stats as st

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

def data_prep(df):
    date = pd.to_datetime(df.index)
    y = df.tolist()
    y = [np.log(x + 1) for x in y]
    data = pd.DataFrame({'ds': date, 'y': y}, index=pd.to_datetime(date))
    data.loc[(data.y < data.y.quantile(.05)) & (data.y > data.y.quantile(.95)), 'y'] = np.nan
    return data


def prophet_configs():
    models = list()
    # define config lists
    n_changepoint = np.linspace(10, 50, 15).tolist()
    preScale_changepoint = np.linspace(0.01, 20, 20).tolist()
    season_params = np.linspace(10, 30, 5).tolist()
    holiday_params = np.linspace(10, 30, 5).tolist()
    growth_params = ['logistic', 'linear']
    season_mode = ['additive', 'multiplicative']
    monthFreq_params = [1,5, 10, 15,24]
    weekFreq_params = [1,5, 10, 15,20]
    quarterFreq_params = [1,5, 10, 15]
    yearFreq_params = [1,5, 10,15, 20]
    # create config instances
    for cp in n_changepoint:
        for pscale in preScale_changepoint:
            for sp in season_params:
                for sm in season_mode:
                    for hp in holiday_params:
                        for g in growth_params:
                            for mf in monthFreq_params:
                                for wf in weekFreq_params:
                                    for qf in quarterFreq_params:
                                        for yf in yearFreq_params:
                                            cfg = [cp, pscale, sp, sm, hp, g, mf, wf, qf, yf]
                                            models.append(cfg)
    return models


def prophet(df, holidays,period,config):
    cp, pscale, season, season_m, holiday_sc, growth, monthFreq, weekFreq, quarterFreq, yearFreq = config
    timeseries = data_prep(df)
    timeseries['cap'] = max(timeseries['y']) * 1.1

    model = Prophet(
        seasonality_mode=season_m,
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
                                                                            ).add_country_holidays('US')
    with suppress_stdout_stderr():
        model.fit(timeseries)
    forecast = model.make_future_dataframe(periods=period, include_history=False)
    forecast['cap'] = max(timeseries['y']) * 1.1
    forecast = model.predict(forecast)
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast.index = pd.to_datetime(forecast['ds'])
    return forecast


def measure_mape(actual, predicted):
    return abs(actual-predicted)/actual


def prophet_validation(data, holidays, cfg):
#    cutoff = pd.date_range(start='2019-01-31', end='2019-12-31', freq='M')
#    test_start = [i + timedelta(days=1) for i in cutoff]
#    test_end = pd.date_range(start='2019-02-29', end='2020-01-31', freq='M')
    test_start = pd.date_range(start='2019-02-01', end='2020-01-31', freq='MS')
    test_end = pd.date_range(start='2019-02-01', end='2020-01-31', freq='M')
    cutoff = [pd.date_range(start=st, end=en, freq='B')[4] for st, en in zip(test_start, test_end)]
    test_start = [i + timedelta(days=1) for i in cutoff]  # lag cutoff date by 1 day
    error = []
    for i in range(len(cutoff)):
        cut_date = datetime.strftime(cutoff[i], '%Y-%m-%d')
        ts = datetime.strftime(test_start[i], '%Y-%m-%d')
        te = datetime.strftime(test_end[i], '%Y-%m-%d')
        train = data[:cut_date]
        test = data[ts:te]
        predictions = prophet(train, holidays, len(test), cfg)
        # print(ts, te, 'error of shipment={}'.format(measure_mape(test.values, predictions['yhat'].values)))
        error.append(measure_mape(test.sum(), predictions['yhat'].sum()))
    return np.round(np.mean(error),3)


def score_model(data, holidays, cfg):
    result = None
    key = str(cfg)
    try:
        result = prophet_validation(data, holidays, cfg)
    except:
        error = None
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


def grid_search(data, holidays, cfg_list, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs = 8, backend='multiprocessing')
        tasks = (delayed(score_model)(data, holidays,cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, holidays, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

