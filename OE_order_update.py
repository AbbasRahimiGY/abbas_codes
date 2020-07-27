import pandas as pd
from datetime import datetime, timedelta
import pyodbc
import numpy as np
import os

os.chdir(r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast')
# Make a connection
link = ('DSN=EDWTDPRD;UID=AA68383;PWD=ilivein339westberry')
pyodbc.pooling = False


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


def prep_order_data(df):
    def add_missing_dates(grp):
        _ = grp.set_index('PLN_DEL_DT')
        _ = _.reindex(idx)
        return _

    df_update = df.copy()
    df_update['PLN_DEL_DT'] = df_update['PLN_DEL_DT'].astype('datetime64')
    df_update['SNAP_DT'] = df_update['SNAP_DT'].astype('datetime64')

    _ = df_update.pivot_table('PRI_AND_CURR_MTH_IN_PROC_QTY', index='SNAP_DT', columns='CUSTOMER',
                       aggfunc='sum', dropna=False, fill_value=0).resample('Y').sum()

    active_customer = [col for col in _.columns if _['2019':][col].sum() > 100]
    # read orders
    df_update[['PRI_AND_CURR_MTH_COMMIT_QTY',
               'PRI_AND_CURR_MTH_IN_PROC_QTY']] = df_update[['PRI_AND_CURR_MTH_COMMIT_QTY',
                                                             'PRI_AND_CURR_MTH_IN_PROC_QTY']].astype('int32')
    df_update = df_update[df_update.CUSTOMER.isin(active_customer)]
    df_update['PLN_DEL_DT'] = df_update.PLN_DEL_DT.astype('Datetime64')
    df_update = df_update.drop_duplicates()
    # fill in for days some customers are missing
    df_update = df_update[['SNAP_DT', 'CUSTOMER', 'PLN_DEL_DT',
                           'PRI_AND_CURR_MTH_COMMIT_QTY', 'PRI_AND_CURR_MTH_IN_PROC_QTY']]
    df_update = pd.pivot_table(df_update, index=['SNAP_DT', 'PLN_DEL_DT'], columns='CUSTOMER',
                               values=['PRI_AND_CURR_MTH_IN_PROC_QTY', 'PRI_AND_CURR_MTH_COMMIT_QTY']).fillna(0)
    df_update = df_update.stack(level=['CUSTOMER']).reset_index()

    df_update.PLN_DEL_DT = pd.to_datetime(df_update.PLN_DEL_DT)
    df_update = df_update[
        (df_update.PLN_DEL_DT >= '2014') & (df_update.PLN_DEL_DT < '2021') & (df_update.SNAP_DT >= '2014')]

    # fill in all missing PLN_DEL_DT
    idx = pd.date_range(df_update.PLN_DEL_DT.min(), df_update.PLN_DEL_DT.max(), freq='d')
    # Group by country name and extend
    df_update = df_update.groupby(['SNAP_DT', 'CUSTOMER']).apply(add_missing_dates).drop(
        columns=['SNAP_DT', 'CUSTOMER']). \
        reset_index().fillna(0).rename(columns={'level_2': 'PLN_DEL_DT'})
    # fill in for missing customers
    df_update[['PRI_AND_CURR_MTH_COMMIT_QTY',
               'PRI_AND_CURR_MTH_IN_PROC_QTY']] = df_update[['PRI_AND_CURR_MTH_COMMIT_QTY',
                                                             'PRI_AND_CURR_MTH_IN_PROC_QTY']].astype('int32')
    df_update.to_pickle('historical_orders\historical_order_Rachael_filled_missing.pickle')


hist_orders = pd.read_pickle(r'historical_orders\new_historical_order.pickle')
df = hist_orders.copy()
df['PLN_DEL_DT'] = df['PLN_DEL_DT'].astype('datetime64')
df['SNAP_DT'] = df['SNAP_DT'].astype('datetime64')
last_snap = datetime.strftime(df['SNAP_DT'].tail(1).tolist()[0] + timedelta(days=1), "%Y-%m-%d")
current_date = datetime.strftime(datetime.today() - timedelta(days=1), "%Y-%m-%d")

dates = pd.date_range(start=last_snap, end=current_date, freq='d')

if len(dates) > 0:
    orders = pd.DataFrame()
    for i in range(len(dates)):
        day_order = order_query_rachael(dates[i])
        #order['SNAP_DT'] = pd.to_datetime(order['SNAP_DT'], errors='coerce')
        orders = orders.append(day_order, ignore_index=True)
    hist_orders = hist_orders.append(orders)
    hist_orders.to_pickle(r'historical_orders\new_historical_order.pickle')
    prep_order_data(hist_orders) # update the orders with additional info and filled missing dates



