import pandas as pd
import numpy as np
import datetime as dt
import sys
import pyodbc
import time

sys.path.append('S:/Stoxx/Product Development and Research/Team/ChristianM/Documentation/QAD')
import qadconnect34plus as q
import myfunctions as myf


#dloc = '//res03/proj/Misc/INDEX_DOCS/Workflowtool/2667/05_BT/01_calc_steps/'
#dloc = 'S:/Stoxx/Product Development and Research/Projects/2667 Deka MultiFactor/05_BT/01_calc_steps/'
dloc = '//teams.deutsche-boerse.de@SSL/DavWWWRoot/sites/sp0056/WFT Data/2667/05_BT/01_calc_steps/'

histloc = dloc + 'universe/01_gl1800.csv'

creds = 'DRIVER={SQL Server};SERVER=delacroix.prod.ci.dom;DATABASE=qai;UID=stx-txg2a;PWD=stx-txg2a'
con = pyodbc.connect(creds)


def get_tr_series(ic, startdate, enddate, currency):
    #startdate_ = startdate - dt.timedelta(days=15)
    #startdate = startdate_ - dt.timedelta(days=(startdate_.day)) + dt.timedelta(days=1)

    c = get_currency(ic, ic)
    if (c == currency) or (currency == 'loc'):
        sql = """
        SELECT MarketDate, RI
        FROM DS2PrimQtRI
        WHERE InfoCode = ?
            AND MarketDate <= ?
            AND MarketDate >= ?
        """
        df = pd.read_sql(sql, con, index_col='MarketDate', params=[str(ic), str(enddate), str(startdate)])
        df.columns = [ic]
    else:
        sql = """
        SELECT r.MarketDate, r.RI / fxr.MidRate
        FROM DS2PrimQtRI r, DS2FXCode fxc, DS2FXRate fxr
        WHERE fxc.FromCurrCode = ?
            AND fxc.ToCurrCode = ?
            AND fxc.RateTypeCode = 'SPOT'
            AND fxr.ExRateIntCode = fxc.ExRateIntCode
            AND fxr.ExRateDate = r.MarketDate
            AND r.InfoCode = ?
            AND r.MarketDate >= ?
            AND r.MarketDate <= ?
        """
        df = pd.read_sql(sql, con, index_col='MarketDate',
                         params=[str(c), str(currency), str(ic), str(startdate), str(enddate)])
        df.columns = [ic]
    if c == 'GBP':
        df = df / 100.
    df.index.name = None
    return df.sort_index()


def get_pr_series(ic, startdate, enddate, currency):
    startdate_ = startdate - dt.timedelta(days=15)
    startdate = startdate_ - dt.timedelta(days=(startdate_.day)) + dt.timedelta(days=1)

    c = get_currency(ic, ic)
    if (c == currency) or (currency == 'loc'):
        sql = """
        SELECT MarketDate, Close_
        FROM DS2PrimQtPrc
        WHERE InfoCode = ?
            AND MarketDate <= ?
            AND MarketDate >= ?
        """
        df = pd.read_sql(sql, con, index_col='MarketDate', params=[str(ic), str(enddate), str(startdate)])
        df.columns = [ic]
    else:
        sql = """
        SELECT r.MarketDate, r.Close_ / fxr.MidRate
        FROM DS2PrimQtPrc r, DS2FXCode fxc, DS2FXRate fxr
        WHERE fxc.FromCurrCode = ?
            AND fxc.ToCurrCode = ?
            AND fxc.RateTypeCode = 'SPOT'
            AND fxr.ExRateIntCode = fxc.ExRateIntCode
            AND fxr.ExRateDate = r.MarketDate
            AND r.InfoCode = ?
            AND r.MarketDate >= ?
            AND r.MarketDate <= ?
        """
        df = pd.read_sql(sql, con, index_col='MarketDate',
                         params=[str(c), str(currency), str(ic), str(startdate), str(enddate)])
        df.columns = [ic]
    if c == 'GBP':
        df = df / 100.
    df.index.name = None
    return df.sort_index()


def get_moend_tr_series(ic, startdate, enddate, currency):
    try:
        df1 = get_tr_series(ic, startdate, enddate, currency)
        dates = pd.DatetimeIndex(df1.index)
        ismonthend = (dates.day[0:len(dates) - 1] > dates.day[1:len(dates)])
        ismonthend = list(ismonthend)
        ismonthend.append(True)
        return df1[df1.index.isin(dates[ismonthend])]
    except:
        return np.nan


def get_moend_pr_series(ic, startdate, enddate, currency):
    try:
        df1 = get_pr_series(ic, startdate, enddate, currency)
        dates = pd.DatetimeIndex(df1.index)
        ismonthend = (dates.day[0:len(dates) - 1] > dates.day[1:len(dates)])
        ismonthend = list(ismonthend)
        ismonthend.append(True)
        return df1[df1.index.isin(dates[ismonthend])]
    except:
        return np.nan


def get_currency(identifier, infoc=0):
    try:
        if infoc == 0:
            ic = q.get_infocode(identifier)
        elif infoc != 0:
            ic = infoc

        if np.isnan(int(ic)):
            return np.nan
        else:
            sql = """
            SELECT PrimISOCurrCode
            FROM Ds2CtryQtInfo
            WHERE Infocode = ?
            """
            res = pd.read_sql(sql, con, params=[str(ic)]).values
            if len(res) > 0:
                return res[0][0]
            else:
                return np.nan
    except:
        return np.nan


def get_fxrate(fromcurr, tocurr, date):
    """Return exchange rate
    Most recent if not available on date

    Keyword arguements:
    date -- (datetime.date)
    fromcurr -- (string)
    tocurr -- (string)
    """
    if fromcurr == tocurr:
        return 1
    else:
        sqlcode = """
        SELECT ExRateIntCode
        FROM DS2FXCode
        WHERE FromCurrCode = '%s'
            AND ToCurrCode = '%s'
            AND RateTypeCode = 'SPOT'
        """ % (fromcurr, tocurr)
        try:
            exrateintcode = pd.io.sql.read_sql(sqlcode, con).loc[0, 'ExRateIntCode']
        except:
            return np.nan
        sqlrate = """
        SELECT MidRate
        FROM DS2FxRate
        WHERE ExRateIntCode = '%s'
            AND ExRateDate <= '%s'
            ORDER BY ExRateDate DESC
        """ % (str(exrateintcode), dt.date.strftime(date, '%Y-%m-%d'))
        try:
            return pd.io.sql.read_sql(sqlrate, con).loc[0, 'MidRate']
        except:
            return np.nan


def get_vencode(sedol, venType):
    """Return the vendor code corresponding to a specified vendor - no date filtering
    """
    sql = """
    SELECT VenCode
    FROM %sSecSdlChg%s ssc, %sSecMapX smx
    WHERE ssc.Sedol = '%s'
        AND ssc.SecCode = smx.SecCode
        AND smx.Rank = 1
        AND smx.VenType = %s
    """
    vc = pd.io.sql.read_sql(sql % ('', 'X', '', sedol[:6], venType), con)
    if len(vc) == 0:
        vc = pd.io.sql.read_sql(sql % ('G', '', 'G', sedol[:6], venType), con)
    if len(vc) == 0:
        return np.nan
    else:
        return vc.loc[0, 'VenCode']


def get_vencode_from_infocode(ic, vtype):
    """Return the vendor code corresponding to a specified vendor based on infocode
    """
    try:
        sql = "SELECT SecCode from %sSecMapX WHERE VenType=33 and VenCode = '%s'"

        sc = pd.io.sql.read_sql(sql % ('', ic), con)
        if len(sc) == 0:
            sc = pd.io.sql.read_sql(sql % ('G', ic), con)
        if len(sc) == 0:
            return np.nan
        else:
            seccode = sc.loc[0, 'SecCode']
            sql1 = "SELECT VenCode from %sSecMapX WHERE VenType = '%s' and SecCode = '%s'"

            vc = pd.io.sql.read_sql(sql1 % ('', str(vtype), seccode), con)
            if len(vc) == 0:
                vc = pd.io.sql.read_sql(sql1 % ('G', str(vtype), seccode), con)
            if len(vc) == 0:
                return np.nan
            else:
                return vc.loc[0, 'VenCode']
    except:
        return np.nan


def get_wspit_currency(vencode):
    sqlstr = """
    select ISOCurrCode from wspitinfo where Code = '%s'
    """ % (vencode)
    try:
        res = pd.io.sql.read_sql(sqlstr, con)
        return res.loc[0, 'ISOCurrCode']
    except:
        return np.nan


# #60m stdev
#                try:
#                    dfri = get_moend_tr_series(ic, myf.add_months(d,-60), d, 'loc')
#                    actret= np.array(dfri.iloc[1:len(dfri),0])/np.array(dfri.iloc[0:len(dfri)-1,0])-1
#                    if len(actret)>2:
#                        wsdata[index][2] = np.std(actret, ddof=1)
#                    else:
#                        wsdata[index][2] = np.nan
#                except:
#                    wsdata[index][2] = np.nan
#
#                #12m return
#                try:
#                    dfri_ = get_moend_tr_series(ic, myf.add_months(d,-12), d, 'loc')
#                    if len(dfri_)>1:
#                        wsdata[index][3] = dfri_.iloc[len(dfri_)-1,0] / dfri_.iloc[0,0] - 1
#                    else:
#                        wsdata[index][3] = np.nan
#                except:
#                    wsdata[index][3] = np.nan
#

def get_reg_filter(reg):
    if reg == 'Europe':
        get_reg_filter = list(['EA', 'EB', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI', 'EK', 'EL', 'EM', 'EO',
                               'EP', 'ER', 'ES', 'EU', 'EY', 'AT', 'BE', 'CH', 'CZ', 'DE', 'DK', 'FI',
                               'FR', 'GB', 'GR', 'IE', 'IT', 'LU', 'NL', 'NO', 'PT', 'SE'])
    if reg == 'America':
        get_reg_filter = list(['AA', 'AC', 'US', 'CA'])

    elif reg == 'AsiaPac':
        get_reg_filter = list(['PJ', 'PA', 'PH', 'PS', 'PZ', 'JP', 'AU', 'CN', 'HK', 'NZ', 'SG'])

    return get_reg_filter

    ##example
    ##dfjp.loc[dfjp[dfjp.country.isin(get_reg_filter('Europe'))].index, 'region'] = 'Europe'
    #
    ##zscore
    # value2_med = df2.mean()[len(df2.median())-1]
    # value2_std = df2.std()[len(df2.std())-1]
    #
    # df2.iloc[:, 1] = (df2.iloc[:, 1] - value2_med) / value2_std