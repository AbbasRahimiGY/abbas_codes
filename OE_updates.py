import pandas as pd
from datetime import datetime, timedelta
import pyodbc
import numpy as np
import os, csv
import warnings
import win32com.client as win32

warnings.filterwarnings('ignore')

link = ('DSN=EDWTDPRD;UID=AA68383;PWD=baradarkhobvaghashang1364')
pyodbc.pooling = False
os.chdir(r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast\OE_updates')


def order_cat_query():
    query = '''SELECT
                 
                  CAST(COALESCE(O1.SNAP_DT, D1.SNAP_DT) AS DATE) AS SNAP_DT,
                 -- C.CUST_GRP_NAME AS CUSTOMER,
                  M.MKT_CTGY_MKT_AREA_NAME AS CTGY,
                 
                  SUM(ZEROIFNULL(O1.BPI_COMMIT)) AS PRI_AND_CURR_MTH_COMMIT_QTY,
                 
                 SUM(             ZEROIFNULL(D1.PRI_AND_CURR_MTH_INPROC_QTY)        ) AS PRI_AND_CURR_MTH_IN_PROC_QTY,
                  PRI_AND_CURR_MTH_COMMIT_QTY+PRI_AND_CURR_MTH_IN_PROC_QTY AS WORKING,
                  SUM(D1.DELIV_QTY) AS SHIP_QTY,
                  SHIP_QTY +WORKING AS shipped_plus_working
               
            FROM
                          (
                                         SELECT 
                                                       O.SHIP_TO_CUST_ID,
                                                       O.SNAP_DT,
                                                     --  O.PLN_DELIV_DT,
                                                       O.MATL_ID,
                                                       --O.DELIV_BLK_CD,
                                                       SUM(O.RPT_ORDER_QTY) AS ORD_QTY,
                                                       SUM(O.RPT_CNFRM_QTY) AS CONFIRM_QTY,
                                                       SUM(
                                                       CASE
                                                                                    -- WHEN O.PLN_GOODS_ISS_DT < ADD_MONTHS((CURRENT_DATE -1) - EXTRACT(DAY FROM CURRENT_DATE -1) + 1, 1)
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
                                                       AND O.SNAP_DT BETWEEN DATE-2 AND DATE-1 
                                                   
                                                       --  AND  TRUNC(O.SNAP_DT,'MM') =TRUNC(O.PLN_DELIV_DT,'mm')
                                         GROUP BY 
                                                       O.SHIP_TO_CUST_ID,
                                                       O.SNAP_DT,
                                                     
                                                       O.MATL_ID
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
                                                                     -- D.PLN_GOODS_MVT_DT AS PLN_DELIV_DT,
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
                                                                      D.SNAP_DT BETWEEN DATE-2 AND DATE-1
                                                                      AND TRUNC(COALESCE(D.ACTL_GOODS_ISS_DT, CURRENT_DATE-1) ,'MM') = TRUNC(CURRENT_DATE-1,'MM')
                                                                     -- AND D.GOODS_ISS_IND = 'N'
                                                       GROUP BY
                                                                      D.SHIP_TO_CUST_ID,
                                                                      D.MATL_ID,
                                                                      D.SNAP_DT
                                                                     
                                         )
                                         D1
                          ON
                                         O1.SNAP_DT = D1.SNAP_DT
                                         AND O1.SHIP_TO_CUST_ID = D1.SHIP_TO_CUST_ID
                                         AND O1.MATL_ID = D1.MATL_ID
                  
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
                          2
            ORDER BY
                          1,
                          2;
                 '''

    with pyodbc.connect(link, autocommit=True) as connect:
        df = pd.read_sql(query, connect)
    return df  # old model #


def order_query_update():
    query = '''SELECT
                 
                  CAST(COALESCE(O1.SNAP_DT, D1.SNAP_DT) AS DATE) AS SNAP_DT,
                  C.CUST_GRP_NAME AS CUSTOMER,
                 
                  SUM(ZEROIFNULL(O1.BPI_COMMIT)) AS PRI_AND_CURR_MTH_COMMIT_QTY,
                 
                 SUM(             ZEROIFNULL(D1.PRI_AND_CURR_MTH_INPROC_QTY)        ) AS PRI_AND_CURR_MTH_IN_PROC_QTY,
                  PRI_AND_CURR_MTH_COMMIT_QTY+PRI_AND_CURR_MTH_IN_PROC_QTY AS WORKING,
                  SUM(D1.DELIV_QTY) AS SHIP_QTY,
                  SHIP_QTY +WORKING AS shipped_plus_working
               
            FROM
                          (
                                         SELECT 
                                                       O.SHIP_TO_CUST_ID,
                                                       O.SNAP_DT,
                                                     --  O.PLN_DELIV_DT,
                                                       O.MATL_ID,
                                                       --O.DELIV_BLK_CD,
                                                       SUM(O.RPT_ORDER_QTY) AS ORD_QTY,
                                                       SUM(O.RPT_CNFRM_QTY) AS CONFIRM_QTY,
                                                       SUM(
                                                       CASE
                                                                                    -- WHEN O.PLN_GOODS_ISS_DT < ADD_MONTHS((CURRENT_DATE -1) - EXTRACT(DAY FROM CURRENT_DATE -1) + 1, 1)
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
                                                       AND O.SNAP_DT BETWEEN DATE-2 AND DATE-1 
                                                   
                                                       --  AND  TRUNC(O.SNAP_DT,'MM') =TRUNC(O.PLN_DELIV_DT,'mm')
                                         GROUP BY 
                                                       O.SHIP_TO_CUST_ID,
                                                       O.SNAP_DT,
                                                     
                                                       O.MATL_ID
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
                                                                     -- D.PLN_GOODS_MVT_DT AS PLN_DELIV_DT,
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
                                                                      D.SNAP_DT BETWEEN DATE-2 AND DATE-1
                                                                      AND TRUNC(COALESCE(D.ACTL_GOODS_ISS_DT, CURRENT_DATE-1) ,'MM') = TRUNC(CURRENT_DATE-1,'MM')
                                                                     -- AND D.GOODS_ISS_IND = 'N'
                                                       GROUP BY
                                                                      D.SHIP_TO_CUST_ID,
                                                                      D.MATL_ID,
                                                                      D.SNAP_DT
                                                                     
                                         )
                                         D1
                          ON
                                         O1.SNAP_DT = D1.SNAP_DT
                                         AND O1.SHIP_TO_CUST_ID = D1.SHIP_TO_CUST_ID
                                         AND O1.MATL_ID = D1.MATL_ID
                  
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
                          2
            ORDER BY
                          1,
                          2;
                 '''

    with pyodbc.connect(link, autocommit=True) as connect:
        df = pd.read_sql(query, connect)
    return df


def order_line_query():
    query = '''SELECT

                  CAST(COALESCE(O1.SNAP_DT, D1.SNAP_DT) AS DATE) AS SNAP_DT,
                  C.CUST_GRP_NAME AS CUSTOMER,
                  M.MKT_CTGY_PROD_LINE_NAME AS LINE,

                  SUM(ZEROIFNULL(O1.BPI_COMMIT)) AS PRI_AND_CURR_MTH_COMMIT_QTY,

                 SUM(  ZEROIFNULL(D1.PRI_AND_CURR_MTH_INPROC_QTY)        ) AS PRI_AND_CURR_MTH_IN_PROC_QTY,
                  Zeroifnull(PRI_AND_CURR_MTH_COMMIT_QTY)+ zeroifnull(PRI_AND_CURR_MTH_IN_PROC_QTY) AS WORKING,
                  SUM(zeroifnull(D1.DELIV_QTY)) AS SHIP_QTY,
                  SHIP_QTY +WORKING AS shipped_plus_working


            FROM
                          (
                                         SELECT 
                                                       O.SHIP_TO_CUST_ID,
                                                       O.SNAP_DT,
                                                     --  O.PLN_DELIV_DT,
                                                       O.MATL_ID,
                                                       --O.DELIV_BLK_CD,
                                                       SUM(O.RPT_ORDER_QTY) AS ORD_QTY,
                                                       SUM(O.RPT_CNFRM_QTY) AS CONFIRM_QTY,
                                                       SUM(
                                                       CASE
                                                                                    -- WHEN O.PLN_GOODS_ISS_DT < ADD_MONTHS((CURRENT_DATE -1) - EXTRACT(DAY FROM CURRENT_DATE -1) + 1, 1)
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
                                                       AND O.SNAP_DT BETWEEN DATE-2 AND DATE-1 

                                                       --  AND  TRUNC(O.SNAP_DT,'MM') =TRUNC(O.PLN_DELIV_DT,'mm')
                                         GROUP BY 
                                                       O.SHIP_TO_CUST_ID,
                                                       O.SNAP_DT,

                                                       O.MATL_ID
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
                                                                     -- D.PLN_GOODS_MVT_DT AS PLN_DELIV_DT,
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
                                                                      D.SNAP_DT BETWEEN DATE-2 AND DATE-1
                                                                      AND TRUNC(COALESCE(D.ACTL_GOODS_ISS_DT, CURRENT_DATE-1) ,'MM') = TRUNC(CURRENT_DATE-1,'MM')
                                                                     -- AND D.GOODS_ISS_IND = 'N'
                                                       GROUP BY
                                                                      D.SHIP_TO_CUST_ID,
                                                                      D.MATL_ID,
                                                                      D.SNAP_DT

                                         )
                                         D1
                          ON
                                         O1.SNAP_DT = D1.SNAP_DT
                                         AND O1.SHIP_TO_CUST_ID = D1.SHIP_TO_CUST_ID
                                         AND O1.MATL_ID = D1.MATL_ID

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
                          2,3
            ORDER BY
                          1,
                          2,3;
                 '''

    with pyodbc.connect(link, autocommit=True) as connect:
        df = pd.read_sql(query, connect)
    return df  # old model #


def plan_line_query():
    query = '''SELECT
               SP.PERD_BEGIN_MTH_DT AS "DATE",
               --SP.LAG_DESC,
               SP.FORC_TYP_DESC,
               --SP.SALES_ORG_CD,
               --SP.DISTR_CHAN_CD,
               --SP.CUST_GRP_ID,
               CG.CUST_GRP_NAME AS "CUSTOMER",
               M.MKT_CTGY_PROD_LINE_NAME AS LINE,
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
                       --AND SP.SALES_ORG_CD IN('N302', 'N312')
                       AND SP.DATA_SRC_CD = 'CUST_SLS_PLN_MTH'
                       AND CG.CUST_HIER_GRP_2_DESC ='oe'
                GROUP BY
                       SP.PERD_BEGIN_MTH_DT,
                       --SP.LAG_DESC,
                       SP.FORC_TYP_DESC,
                       --SP.SALES_ORG_CD,
                       --SP.DISTR_CHAN_CD,
                       --SP.CUST_GRP_ID,
                       CG.CUST_GRP_NAME,
                       M.MKT_CTGY_PROD_LINE_NAME
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


def plan_query():
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


def plan_cat_query():
    query = '''SELECT
               SP.PERD_BEGIN_MTH_DT AS "DATE",
               --SP.LAG_DESC,
               SP.FORC_TYP_DESC,
               --SP.SALES_ORG_CD,
               --SP.DISTR_CHAN_CD,
               --SP.CUST_GRP_ID,
               M.MKT_CTGY_MKT_AREA_NAME AS CTGY,
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
                       M.MKT_CTGY_MKT_AREA_NAME
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


# prep data at different aggregation
def prep_cat_level():  # old model
    order = order_cat_query()
    plan_CT = plan_cat_query()
    df = order.copy()
    lastday = datetime.strftime(datetime.now() - timedelta(days=1), '%Y-%m-%d')
    twodaysago = datetime.strftime(datetime.now() - timedelta(days=2), '%Y-%m-%d')
    df.index = pd.to_datetime(df['SNAP_DT'])
    # combine other companies for plan
    part_a = df[df.index == lastday]
    part_b = df[df.index == twodaysago]
    # make sure we only look at current month

    name_sop = plan_CT['FORC_TYP_DESC'].tail(1).values.tolist()[0]  # use the current month pln as header name
    if df.index[-1].day > 1:
        part_a.loc[:, 'VS_Prior_Day'] = (part_a['shipped_plus_working'].values -
                                         part_b['shipped_plus_working'].values).astype('int64')
        part_a = part_a.merge(plan_CT[['SOP_QTY', 'CTGY']], left_on='CTGY',
                              right_on='CTGY', how='right').rename(columns={'SOP_QTY': name_sop})
        part_a.fillna(value=0, inplace=True)
        part_a[f'VS {name_sop}'] = (part_a['shipped_plus_working'] - part_a[name_sop]).astype('int64')
        part_a = part_a[['CTGY', name_sop, 'shipped_plus_working', f'VS {name_sop}', 'VS_Prior_Day']]

    else:
        part_a = part_a.merge(plan_CT[['SOP_QTY', 'CTGY']], left_on='CTGY',
                              right_on='CTGY', how='right').rename(columns={'SOP_QTY': name_sop})
        part_a.fillna(value=0, inplace=True)
        part_a[f'VS {name_sop}'] = part_a['shipped_plus_working'] - part_a[name_sop]
        part_a = part_a[['CTGY', name_sop, 'shipped_plus_working', f'VS {name_sop}']]

    part_a = part_a.sort_values(by='shipped_plus_working', ascending=False)
    custom_dict = {'Commuter / Touring': 0, 'All Terrain': 1, 'High Performance': 2}

    part_a = part_a.iloc[part_a['CTGY'].map(custom_dict).argsort()].rename(columns={'CTGY': 'Category'})
    part_a.to_excel('OE_Update_CAT_{}.xlsx'.format(datetime.strftime(datetime.now(), '%Y-%m-%d')), index=False)


def prep_customer_level():
    plan = plan_query()
    # shipment = shipment_query()
    orders = order_query_update()  # order_query()
    df = orders.copy()
    lastday = datetime.strftime(datetime.now() - timedelta(days=1), '%Y-%m-%d')
    twodaysago = datetime.strftime(datetime.now() - timedelta(days=2), '%Y-%m-%d')
    df.loc[df.CUSTOMER == 'Chrysler', 'CUSTOMER'] = 'FCA'
    # combine other companies for orders
    main = ['FCA', 'GM', 'Ford', 'Honda', 'Toyota', 'Nissan']
    others = [customer for customer in df.CUSTOMER.unique() if customer not in main]
    df.loc[df.CUSTOMER.isin(others), 'CUSTOMER'] = 'Other'
    df = df.groupby(['SNAP_DT', 'CUSTOMER'], as_index=False).sum()
    df.index = pd.to_datetime(df['SNAP_DT'])
    # combine other companies for plan
    plan.loc[plan.CUSTOMER.isin(others), 'CUSTOMER'] = 'Other'
    plan = plan.groupby(['DATE', 'CUSTOMER', 'FORC_TYP_DESC'], as_index=False).sum()
    plan.loc[plan.CUSTOMER == 'Chrysler', 'CUSTOMER'] = 'FCA'

    part_a = df[df.index == lastday]
    part_b = df[df.index == twodaysago]
    # make sure we only look at current month

    name_sop = plan['FORC_TYP_DESC'].tail(1).values.tolist()[0]  # use the current month pln as header name
    if df.index[-1].day > 1:
        part_a.loc[:, 'VS_Prior_Day'] = (part_a['shipped_plus_working'].values -
                                         part_b['shipped_plus_working'].values).astype('int64')
        part_a = part_a.merge(plan[['SOP_QTY', 'CUSTOMER']], left_on='CUSTOMER',
                              right_on='CUSTOMER', how='right').rename(columns={'SOP_QTY': name_sop})
        part_a.fillna(value=0, inplace=True)
        part_a[f'VS {name_sop}'] = (part_a['shipped_plus_working'] - part_a[name_sop]).astype('int64')
        part_a = part_a[['CUSTOMER', name_sop, 'shipped_plus_working', f'VS {name_sop}', 'VS_Prior_Day']]

    else:
        part_a = part_a.merge(plan[['SOP_QTY', 'CUSTOMER']], left_on='CUSTOMER',
                              right_on='CUSTOMER', how='right').rename(columns={'SOP_QTY': name_sop})
        part_a.fillna(value=0, inplace=True)
        part_a[f'VS {name_sop}'] = part_a['shipped_plus_working'] - part_a[name_sop]
        part_a = part_a[['CUSTOMER', name_sop, 'shipped_plus_working', f'VS {name_sop}']]

    part_a = part_a.sort_values(by='shipped_plus_working', ascending=False)
    if 'Other' in part_a.CUSTOMER.unique():
        custom_dict = {'FCA': 0, 'GM': 1, 'Ford': 3,
                       'Honda': 4, 'Toyota': 5, 'Nissan': 6, 'Other': 7}
    else:
        custom_dict = {'FCA': 0, 'GM': 1, 'Ford': 3,
                       'Honda': 4, 'Toyota': 5, 'Nissan': 6}

    part_a = part_a.iloc[part_a['CUSTOMER'].map(custom_dict).argsort()]

    ''' real_columns = [col for col in part_a.columns if 'CUSTOMER' not in col]
    for i in real_columns:
        if i in ([f'VS {name_sop}', 'VS_Prior_Day']):
            part_a[i] = part_a[i].apply(
                lambda x: np.ceil(x / 1000) if (x / 1000) > 0 else np.floor(x / 1000))
        else:
            part_a[i] = part_a[i].apply(lambda x: np.round(x / 1000,0))
     '''
    part_a.to_excel('OE_Update_{}.xlsx'.format(datetime.strftime(datetime.now(), '%Y-%m-%d')), index=False)


def prep_line_level():
    order = order_line_query()
    plan_CT = plan_line_query()
    df = order.copy()
    lastday = datetime.strftime(datetime.now() - timedelta(days=1), '%Y-%m-%d')
    twodaysago = datetime.strftime(datetime.now() - timedelta(days=2), '%Y-%m-%d')
    df.index = pd.to_datetime(df['SNAP_DT'])
    # combine other companies for plan

    part_a = df[df.index == lastday]
    part_b = df[df.index == twodaysago]
    # make sure there is no missing lines comparing yesterday vs today
    check_point = pd.merge(part_a, part_b, on=['CUSTOMER', 'LINE'], how='outer', suffixes=('', '_db')).fillna(0)
    part_a = check_point[[col for col in check_point if 'db' not in col]]
    part_b = check_point[[col for col in check_point if 'db' in col]]
    part_b.columns = part_b.columns.str.rstrip('_db')
    # make sure we only look at current month

    name_sop = plan_CT['FORC_TYP_DESC'].tail(1).values.tolist()[0]  # use the current month pln as header name
    if df.index[-1].day > 1:
        part_a.loc[:, 'VS_Prior_Day'] = (part_a['shipped_plus_working'].values -
                                         part_b['shipped_plus_working'].values).astype('int64')
        part_a = part_a.merge(plan_CT[['SOP_QTY', 'CUSTOMER', 'LINE']], left_on=['CUSTOMER', 'LINE'],
                              right_on=['CUSTOMER', 'LINE'], how='right').rename(columns={'SOP_QTY': name_sop})
        part_a.fillna(value=0, inplace=True)
        part_a[f'VS {name_sop}'] = (part_a['shipped_plus_working'] - part_a[name_sop]).astype('int64')
        part_a = part_a[['CUSTOMER', 'LINE', name_sop, 'shipped_plus_working', f'VS {name_sop}', 'VS_Prior_Day']]
        part_a.sort_values(by='VS_Prior_Day')
    else:
        part_a = part_a.merge(plan_CT[['SOP_QTY', 'CUSTOMER', 'LINE']], left_on=['CUSTOMER', 'LINE'],
                              right_on=['CUSTOMER', 'LINE'], how='right').rename(columns={'SOP_QTY': name_sop})
        part_a.fillna(value=0, inplace=True)
        part_a[f'VS {name_sop}'] = part_a['shipped_plus_working'] - part_a[name_sop]
        part_a = part_a[['CTGY', name_sop, 'shipped_plus_working', f'VS {name_sop}']]

    part_a.to_excel('OE_Update_LINE_{}.xlsx'.format(datetime.strftime(datetime.now(), '%Y-%m-%d')), index=False)


# prep mails out
def line_mail():
    # prep html file for category
    def color_negative_red(value):
        """
        Colors elements in a dateframe
        green if positive and red if
        negative. Does not color NaN
        values.
        """
        if value < 0:
            color = 'red'
        elif value > 0:
            color = 'green'
        else:
            color = 'black'

        return 'color: %s' % color

    attachment = r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast\OE_updates\OE_Update_LINE_{}.xlsx'. \
        format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
    df = pd.read_excel(attachment)
    df = df.sort_values(by='VS_Prior_Day', ascending=False)
    df = pd.concat([df.head(5), df.tail(5)])
    df = df[abs(df['VS_Prior_Day']) > 0]
    numeric_cols = [col for col in df.columns if df[col].dtype in ('int64', 'float64')]
    vs_cols = [col for col in df.columns if 'VS' in col]
    df.reset_index(inplace=True, drop=True)

    styles = [
        dict(selector="th", props=[("font-size", "105%"),
                                   ('border-collapse', 'collapse'),
                                   ('border-spacing', '200px 200px'),
                                   ("text-align", "center"),
                                   ]),  # header
        dict(selector='table',
             props=[('border', '1px solid lightgrey'),
                    ('border-collapse', 'collapse')]),

        dict(selector='tbody td',
             props=[('border', '1px solid lightgrey'),
                    ('font-size', '12.5px'),
                    ('font-family', 'arial'),
                    ('text-align', 'center'),
                    ('width', '160')])
    ]

    percent = {}
    for n_col in numeric_cols:
        percent[n_col] = "{:20,.0f}"  # make comma between 1000

    # df.style.apply(highlight, axis=1)
    html = (
        df.style
            .format(percent)
            .set_table_styles(styles)
            .applymap(color_negative_red, subset=vs_cols)
            .set_properties(**{'font-size': '10pt', 'font-family': 'Calibri'})
            .hide_index()
            .render()
    )
    return html


def category_mail():
    # prep html file for category
    def color_negative_red(value):
        """
        Colors elements in a dateframe
        green if positive and red if
        negative. Does not color NaN
        values.
        """
        if value < 0:
            color = 'red'
        elif value > 0:
            color = 'green'
        else:
            color = 'black'

        return 'color: %s' % color

    def bold_font(s):
        if s.Category == 'Grand Total':
            return ['font-weight:bold'] * len(s)
        else:
            return ['background-color: white'] * len(s)

    def highlight(s):
        if s.Category == 'Grand Total':
            return ['background-color: wheat'] * len(s)
        else:
            return ['background-color: white'] * len(s)

    attachment = r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast\OE_updates\OE_Update_CAT_{}.xlsx'. \
        format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
    df = pd.read_excel(attachment)
    numeric_cols = [col for col in df.columns if df[col].dtype in ('int64', 'float64')]
    vs_cols = [col for col in df.columns if 'VS' in col]

    df = pd.concat([df, pd.DataFrame(df.sum(axis=0), columns=['Grand Total']).T])
    df.Category[-1] = 'Grand Total'
    df.reset_index(inplace=True, drop=True)

    styles = [
        dict(selector="th", props=[("font-size", "105%"),
                                   ('border-collapse', 'collapse'),
                                   ('border-spacing', '200px 200px'),
                                   ("text-align", "center"),
                                   ]),  # header
        dict(selector='table',
             props=[('border', '1px solid lightgrey'),
                    ('border-collapse', 'collapse')]),

        dict(selector='tbody td',
             props=[('border', '1px solid lightgrey'),
                    ('font-size', '12.5px'),
                    ('font-family', 'arial'),
                    ('text-align', 'center'),
                    ('width', '160')])
    ]

    percent = {}
    for n_col in numeric_cols:
        percent[n_col] = "{:20,.0f}"  # make comma between 1000

    # df.style.apply(highlight, axis=1)
    html = (
        df.style
            .format(percent)
            .set_table_styles(styles)
            .applymap(color_negative_red, subset=vs_cols)
            .set_properties(**{'font-size': '10pt', 'font-family': 'Calibri'})
            .apply(highlight, axis=1)
            .apply(bold_font, axis=1)
            .hide_index()
            .render()
    )
    return html


def customer_mail():
    def color_negative_red(value):
        """
        Colors elements in a dateframe
        green if positive and red if
        negative. Does not color NaN
        values.
        """
        if value < 0:
            color = 'red'
        elif value > 0:
            color = 'green'
        else:
            color = 'black'

        return 'color: %s' % color

    def bold_font(s):
        if s.CUSTOMER == 'Grand Total':
            return ['font-weight:bold'] * len(s)
        else:
            return ['background-color: white'] * len(s)

    def highlight(s):
        if s.CUSTOMER == 'Grand Total':
            return ['background-color: wheat'] * len(s)
        else:
            return ['background-color: white'] * len(s)

    attachment = r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast\OE_updates\OE_Update_{}.xlsx'. \
        format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
    df = pd.read_excel(attachment)
    numeric_cols = [col for col in df.columns if df[col].dtype in ('int64', 'float64')]
    vs_cols = [col for col in df.columns if 'VS' in col]

    df = pd.concat([df, pd.DataFrame(df.sum(axis=0), columns=['Grand Total']).T])
    df.CUSTOMER[-1] = 'Grand Total'
    df.reset_index(inplace=True, drop=True)

    styles = [
        dict(selector="th", props=[("font-size", "105%"),
                                   ('border-collapse', 'collapse'),
                                   ('border-spacing', '200px 200px'),
                                   ("text-align", "center"),
                                   ]),  # header
        dict(selector='table',
             props=[('border', '1px solid lightgrey'),
                    ('border-collapse', 'collapse')]),

        dict(selector='tbody td',
             props=[('border', '1px solid lightgrey'),
                    ('font-size', '12.5px'),
                    ('font-family', 'arial'),
                    ('text-align', 'center'),
                    ('width', '160')])
    ]

    percent = {}
    for n_col in numeric_cols:
        percent[n_col] = "{:20,.0f}"  # make comma between 1000

    # df.style.apply(highlight, axis=1)
    html = (
        df.style
            .format(percent)
            .set_table_styles(styles)
            .applymap(color_negative_red, subset=vs_cols)
            .set_properties(**{'font-size': '10pt', 'font-family': 'Calibri'})
            .apply(highlight, axis=1)
            .apply(bold_font, axis=1)
            .hide_index()
            .render()
    )
    return html


def send_out_email():
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    email_t4 = 'abbas_rahimi@goodyear.com;stacey_francesconi@goodyear.com;' \
               'nikki_viar@goodyear.com;dusty_smith@goodyear.com;patrick_handley@goodyear.com;' \
               'kenneth_carter@goodyear.com;josh_mottor@goodyear.com;blake_housel@goodyear.com;'
    # email_t4 = 'abbas_rahimi@goodyear.com'
    mail.To = email_t4

    mail.Subject = 'OE Update as of {}'.format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
    mail.Body = 'Attached please find the OE update.'

    if datetime.today().day - 1 > 1:
        customer_html = customer_mail()
        CTGY_html = category_mail()
        line_html = line_mail()
        body = '<html><body>' + \
               '<h2> Per OE Categories Update <h2>' + \
               CTGY_html + \
               '<h2> Per OE Customers Update <h2>' + \
               customer_html + \
               '<h2> Top Moved Lines <h2>' + \
               line_html + \
               '</body></html>'
        mail.HTMLbody = body
        attachment = r'T:\Marketing\GBA-Share\BA Portal Files\OE_Forecast\OE_updates\OE_Update_LINE_{}.xlsx'. \
            format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
        mail.Attachments.Add(attachment)
    else:
        customer_html = customer_mail()
        CTGY_html = category_mail()
        body = '<html><body>' + \
               '<h2> Per OE Categories Update <h2>' + \
               CTGY_html + \
               '<h2> Per OE Customers Update <h2>' + \
               customer_html + \
               '</body></html>'
        mail.HTMLbody = body

    mail.Send()


# prep_customer_level(plan, shipment, orders)
prep_customer_level()
prep_cat_level()
prep_line_level()
send_out_email()
