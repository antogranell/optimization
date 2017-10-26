import pandas as pd
import numpy as np
import datetime as dt
import pyodbc
import myfunctions as myf

# connect to Database
# DRIVER={SQL Server};SERVER=mpzhwindex01;DATABASE=qai;UID=XXXXXXXXXX;PWD=XXXXXXX
# PDDB: 'DRIVER={SQL Server};SERVER=Zurix04.bat.ci.dom\STOXXDBDEV2,55391;DATABASE=TSTXENG02;UID=stx-txg2a;PWD=stx-txg2a'
creds = 'DRIVER={SQL Server};SERVER=delacroix.prod.ci.dom;DATABASE=qai;UID=stx-txg2a;PWD=stx-txg2a'
con = pyodbc.connect(creds)

def get_prod_wswspit(item, sedstr, dat, finalval, ws_type, minmth): 
    
    """Arguments: 
    sedstr: first 6 char of Sedol or Sedol list (text); dat is a str like '2001-01-01'; 
    finalval is 'Y' (retreieve the last valid value available), 'N' (perform aggregation);
    ws_type is the set of tables 'ws' or 'wspit'
    
    Output:
    the field 'epsReportDate' corresponds to the 'pointdate' or 'startdate' field on wspit, 
    'fiscalPeriodEndDate' is the 'Calperiodenddate' or 'enddate' on wspit
    WSPIT tables covered: WSPITFinVal, WSPITCmpIssFData, WSPITSupp
    """
    
    creds1 = 'DRIVER={SQL Server};SERVER=Zurix04.bat.ci.dom\STOXXDBDEV2,55391;DATABASE=TSTXENG03;UID=stx-txg2a;PWD=stx-txg2a' #dbag

    conex = pyodbc.connect(creds1)
    
    if ws_type=='ws':
        db_ = 'usp_get_cumulative_fundamental_generic'
        freq_ = 'A,Q,S,R'
    elif ws_type=='wspit':
        db_ = 'usp_get_cum_fundamental_wspit_fin'
        freq_ = '1,2,8,3,10,5,9,4'

    sql = """
    set nocount on
    if object_id('tempdb..#fundamentals') is not null
    begin
        drop table #fundamentals;
    end;
    create table #fundamentals(
    sedol varchar(6),
    sedol7 varchar(7),
    isin nvarchar(48), 
    dj_id nvarchar(12),
    name VARCHAR(61),
    code INT NOT NULL,
    currencyOfDocument VARCHAR(12),
    epsReportDate DATETIME,
    fiscalPeriodEndDate DATETIME,
    value FLOAT,
    year_ smallint,
    freq varchar(1) ,
    item INT NOT NULL,
    seq SMALLINT NOT NULL ,
    periodUpdateFlag VARCHAR(12),
    itemUnits VARCHAR(9),
    latest_value smallint DEFAULT 0
    );

	exec %s ?, ?, ?, ?, ?, ?;
    select * from #fundamentals
    """
    try:
        res = pd.io.sql.read_sql(sql % db_, conex, params=[item, sedstr, dat, freq_, finalval, minmth])
        return res
    except:
        return pd.DataFrame()
		

#gets a sedol using the isin
def get_sedol(identifier):
    sql = """
    SELECT Sedol
    FROM %sSecMstrX AS t1, %sSecMapX AS t2, DS2CtryQtInfo AS t3
    WHERE t1.SecCode = t2.SecCode
        AND t2.VenType = 33
        AND t2.VenCode = t3.InfoCode
        AND t3.IsPrimQt = 1
        AND Isin = ?
    """
    sql_us = sql % ('','')
    res = pd.io.sql.read_sql(sql_us,con, params=[str(identifier)]).values
    if len(res) > 0:
        return res[0][0]
    else:
        sql_g = sql % ('G','G')
        res = pd.io.sql.read_sql(sql_g, con, params=[str(identifier)]).values
        if len(res) > 0:
            return res[0][0]
        else:
            sql_ch = """
            SELECT t1.Sedol
            FROM DS2SedolChg t1, DS2CtryQtInfo t2
            WHERE t1.Infocode = ?
            AND t2.Infocode = ?
            AND IsPrimQt = 1
            ORDER BY EndDate DESC
            """ #% (str(identifier))
            ic = str(get_infocode(str(identifier)))
            try:
                res = pd.io.sql.read_sql(sql_ch,con, params=[ic, ic]).values
                if len(res) > 0:
                    return res[0][0]
                else:
                    return np.nan
            except:
                return np.nan

def get_localdps_table(identifier, infoc=0):
    if infoc==0:
        ic = get_infocode(identifier)
    elif infoc!=0:
        ic = infoc
        
    if np.isnan(ic):
        return pd.DataFrame(columns=['dt', 'curr', 'dps',])
    else:        
        sql = """
        SELECT a.EventDate, d.EffectiveDate, round(a.DPS, 5), b.DivRate as ext, c.PrimISOCurrCode as curr FROM DS2DPS a 
        LEFT JOIN (SELECT * FROM DS2Div WHERE InfoCode = '%s' and DivTypeCode='EXT') b
        ON a.EventDate = b.AnnouncedDate
        LEFT JOIN (SELECT * FROM DS2Div WHERE InfoCode = '%s' and not DivTypeCode='EXT') d
        ON a.EventDate = d.AnnouncedDate
        LEFT JOIN Ds2CtryQtInfo c on a.InfoCode = c.InfoCode
        WHERE a.InfoCode = '%s'
        ORDER BY EventDate DESC
        """ % (str(ic),str(ic),str(ic))
        res = pd.io.sql.read_sql(sql,con).values
        if len(res) > 0:
            df = pd.DataFrame(res, columns=['dt', 'exdt', 'dps_', 'ext', 'curr']).fillna(0)
            df['dps'] = df['dps_'] - df['ext']
            del df['ext']
            del df['dps_']
            return df
        else:
            return pd.DataFrame(columns=['dt', 'exdt', 'curr', 'dps',])

def get_adjpr_table(identifier, infoc=0):
    if infoc==0:
        ic = get_infocode(identifier)
    elif infoc!=0:
        ic = infoc
        
    c = get_currency(identifier,infoc)
    if np.isnan(ic):
        return pd.DataFrame(columns=['dt','close_', 'cumadj', 'adjclose'])
    else: 
        sql = """
        SELECT a.MarketDate, a.Close_, b.CumAdjFactor, a.Close_ * b.CumAdjFactor AS close_adjusted_loc
        FROM DS2PrimQtPrc a, DS2Adj b
        WHERE a.InfoCode = b.InfoCode
        AND a.MarketDate between b.AdjDate and isnull(b.EndAdjdate, '20790101')
        AND b.AdjType = 2
        AND a.InfoCode = '%s'      
        order by a.MarketDate desc
        """ % (str(ic))
        res = pd.io.sql.read_sql(sql,con).values
        if len(res) > 0:
            df = pd.DataFrame(res, columns=['dt','close_', 'cumadj', 'adjclose'])
            df['dt'] = df['dt'].map(lambda x: str(x)[:10])
            df['dt'] = pd.to_datetime(df['dt'], format='%Y-%m-%d', dayfirst=True)
            
            if c == 'GBP':
                df[['close_', 'adjclose']]  = df[['close_', 'adjclose']] / 100
            return df
        else:
            return pd.DataFrame(columns=['dt','close_', 'cumadj', 'adjclose'])
       

# returns the Datastream InfoCode of primary listing given ISIN or SEDOL
def get_infocode(identifier):
    sql = """
    SELECT VenCode
    FROM %sSecMstrX AS t1, %sSecMapX AS t2, DS2CtryQtInfo AS t3
    WHERE t1.SecCode = t2.SecCode
        AND t2.VenType = 33
        AND t2.VenCode = t3.InfoCode
        AND t3.IsPrimQt = 1
        AND %s = '%s'
    """
    if len(identifier) > 7:
        identifier_type = 'Isin'
        flag = ''
    else:
        identifier = str(identifier)[0:6]
        identifier_type = 'Sedol'
        flag = '--'
    sql_us = sql % ('','',identifier_type,str(identifier))
    res = pd.io.sql.read_sql(sql_us,con).values
    if len(res) > 0:
        return res[0][0]
    else:
        sql_g = sql % ('G','G',identifier_type,str(identifier))
        res = pd.io.sql.read_sql(sql_g,con).values
        if len(res) > 0:
            return res[0][0]
        else:
            if identifier_type == 'Sedol':
                sql_ch = """
                SELECT t1.Infocode
                FROM DS2SedolChg t1, DS2CtryQtInfo t2
                WHERE t1.Sedol = '%s'
                    AND IsPrimQt = 1
                """ % (str(identifier))
            elif identifier_type == 'Isin':
                sql_ch = """
                SELECT Infocode
                FROM DS2IsinChg t1, DS2CtryQtInfo t2
                WHERE t1.Isin = '%s'
                    AND t1.DsSecCode = t2.DsSecCode
                    AND IsPrimQt = 1
                """ % (str(identifier))
            res = pd.io.sql.read_sql(sql_ch,con).values
            if len(res) > 0:
                return res[0][0]
            else:
                return np.nan

#gets a isin using the infocode
def get_isin(identifier):
    sql = """
    SELECT Isin
    FROM %sSecMstrX AS t1, %sSecMapX AS t2, DS2CtryQtInfo AS t3
    WHERE t1.SecCode = t2.SecCode
        AND t1.Type_ in (10,1)
        AND t2.VenType = 33
        AND t2.VenCode = t3.InfoCode
        AND t3.IsPrimQt = 1
        AND t3.Infocode = ?
    """
    sql_us = sql % ('','')
    res = pd.io.sql.read_sql(sql_us,con, params=[str(identifier)]).values
    if len(res) > 0:
        return res[0][0]
    else:
        sql_g = sql % ('G','G')
        res = pd.io.sql.read_sql(sql_g, con, params=[str(identifier)]).values
        if len(res) > 0:
            return res[0][0]
        else:
            sql_ch = """
            SELECT t1.ISIN
            FROM DS2IsinChg t1, DS2CtryQtInfo t2
            WHERE t1.DsSecCode = t2.DsSecCode
            AND t2.Infocode = ?
            AND IsPrimQt = 1
            ORDER BY EndDate DESC
            """
            try:
                res = pd.io.sql.read_sql(sql_ch,con, params=[str(identifier)]).values
                if len(res) > 0:
                    return res[0][0]
                else:
                    return np.nan
            except:
                return np.nan
				
def get_dps_ts(identifier, startdate, enddate, all_=False):
#all_=True returns a dy based on each day's price;
#all_=False returns a dy for each day but based on the price of the last dividend's payment date

    ic = get_infocode(identifier)
    dfa = get_localdps_table(ic, ic)

    df1a = get_adjpr_table(ic, ic)
    dfdta = pd.DataFrame()
    df=dfa
    df1=df1a
    alldates = df1a[(df1a.dt>=startdate) & (df1a.dt<=enddate)]['dt']
    if all_ == False:
        dates= dfa[(dfa.dt>=startdate) & (dfa.dt<=enddate)]['dt']
    else:
        dates= alldates
    for dat in dates:

        dc = 0
        dftemp = pd.DataFrame()
        dat = dt.datetime.strptime(str(dat)[:10], '%Y-%m-%d')
        dat_1 = add_months(dat,-12)
        for d in [dat,dat_1]:
            if len(df[(df.dt<=d) & (df.dt>=add_months(d,-18)) & (-df.dps.isnull())])>0:
                dpsdt = df[(df.dt<=d) & (df.dt>=add_months(d,-18)) & (-df.dps.isnull())].iloc[0,0] #date of last dps
                curr = df[(df.dt<=d) & (df.dt>=add_months(d,-18)) & (-df.dps.isnull())].iloc[0,2] #curr
                dps1 = float(df[(df.dt<=d) & (df.dt>=add_months(d,-18)) & (-df.dps.isnull())].iloc[0,3]) #last dps
            else:
                dpsdt = np.nan
                exdt = np.nan
                curr = np.nan
                dps1 = np.nan

            if len(df1[(df1.dt<=d) & (-df1.close_.isnull())])>0:
                pr = np.round(float(df1[(df1.dt<=d) & (-df1.close_.isnull())].iloc[0,1]),6)
                try:
                    adjfact = np.round(float(df1[(df1.dt<=dpsdt) & (-df1.close_.isnull())].iloc[0,2]) / float(df1[(df1.dt<=d) & (-df1.close_.isnull())].iloc[0,2]),6)
                    #dtdelta = (d-dt.datetime.strptime(str(dpsdt)[:10], '%Y-%m-%d')).days
                except:
                    adjfact = 1
            else:
                pr = np.nan
                adjfact = np.nan
            if dc==0:
                dpsdt0 = dpsdt
            elif dc==1:
                dpsdt12 = dpsdt
                dps12 = dps1 * adjfact

            dps = np.round(dps1 * adjfact, 3)
            strdc = str(dc)
            dc = dc + 1
        try:    
            adjfact12m = np.round(float(df1[(df1.dt<=dpsdt12) & (-df1.close_.isnull())].iloc[0,2]) / float(df1[(df1.dt<=dpsdt0) & (-df1.close_.isnull())].iloc[0,2]),6) 
        except:
            adjfact12m = 1

        dfthisdate = pd.DataFrame(np.array([dat, pr, adjfact, dps])).transpose()
        dfthisdate.columns=['date', 'pr_','adjfact_', 'dps_']
        dftemp = pd.concat([dftemp, dfthisdate], axis=1)
        dfdpsadj = pd.DataFrame(np.array([adjfact12m, np.round(dps12 * adjfact12m, 3)])).transpose()
        dfdpsadj.columns=['adjfact_12m', 'dps_1adj']
        dftemp = pd.concat([dftemp, dfdpsadj], axis=1)

        dfdta = pd.concat([dfdta, dftemp], axis=0)
    try:
        dfdta['dy'] = dfdta['dps_'] / dfdta['pr_']
        dfdta['1y_gwth'] = dfdta['dps_'] / dfdta['dps_1adj'] - 1
        dfdta['date'] = dfdta['date'].map(lambda x: pd.to_datetime(str(x)[:10], format='%Y-%m-%d', dayfirst=True))
        dfdta.index = dfdta['date']
        dfdta.index.name= None
        dfdta = dfdta[['dy', 'dps_','1y_gwth']]
        dfdta = dfdta[(dfdta.index>=startdate) & (dfdta.index<=enddate)].sort()
        x = pd.merge(pd.DataFrame(alldates), dfdta, how='left', left_on='dt', right_index=True)
        x = x.fillna(method='pad')
        x.index=x.dt
        x.index.name=None
        del x['dt']
        x = x[-x.dy.isnull()]    
    except:
        x = np.nan

    return x#.sort()

# returns the number of shares reported of identifier most recently before date
def get_sharesout(identifier, date, infoc=0):
    if infoc==0:
        ic = get_infocode(identifier)
    elif infoc!=0:
        ic = infoc
    if np.isnan(ic):
        return np.nan
    else:
        sql = """
        SELECT NumShrs
        FROM DS2NumShares
        WHERE InfoCode = ?
            AND EventDate <= ?
        ORDER BY EventDate DESC
        """
        res = pd.io.sql.read_sql(sql, con, params=[str(ic), str(date)]).values
        if len(res) > 0:
            return res[0][0] * 1000
        else:
            return np.nan

# returns the freefloat percentage of identifier reported most recently of identifier before date
def get_freefloat(identifier, date):
    ic = get_infocode(identifier)
    if np.isnan(ic):
        return np.nan
    else:
        sql = """
        SELECT FreeFloatPct
        FROM DS2ShareHldgs
        WHERE InfoCode = '%s'
            AND ValDate <= '%s'
        ORDER BY ValDate DESC
        """ % (str(ic), str(date))
        res = pd.io.sql.read_sql(sql,con).values
        if len(res) > 0:
            return res[0][0] / 100.0
        else:
            return 1

# returns the currency code of identifier
def get_currency(identifier, infoc=0):
    
    if infoc==0:
        ic = get_infocode(identifier)
    elif infoc!=0:
        ic = infoc       
    
    if np.isnan(ic):
        return np.nan
    else:
        sql = """
        SELECT PrimISOCurrCode
        FROM Ds2CtryQtInfo
        WHERE Infocode = '%s'
        """ % (str(ic))
        res = pd.io.sql.read_sql(sql,con).values
        if len(res) > 0:
            return res[0][0]
        else:
            return np.nan

# returns the name of of identifier
def get_name(identifier):
    ic = get_infocode(identifier)
    if np.isnan(ic):
        return np.nan
    else:
        sql = """
        SELECT DsQtName
        FROM Ds2CtryQtInfo
        WHERE Infocode = '%s'
        """ % (str(ic))
        res = pd.io.sql.read_sql(sql,con).values
        if len(res) > 0:
            return res[0][0]
        else:
            return np.nan

# returns local closing price of identifier on date
def get_localclose(identifier, date):
    ic = get_infocode(identifier)
    if np.isnan(ic):
        return np.nan
    else:
        sql = """
        SELECT Close_
        FROM DS2PrimQtPrc
        WHERE InfoCode = '%s'
        AND MarketDate <= '%s'
        AND Close_ is not null
        ORDER BY MarketDate DESC
        """ % (str(ic), str(date))
        res = pd.io.sql.read_sql(sql,con).values
        if len(res) > 0:
            return res[0][0]
        else:
            return np.nan

def get_localdps(identifier, date):
    ic = get_infocode(identifier)
    if np.isnan(ic):
        return np.nan
    else:
        sql = """
        SELECT DPS
        FROM DS2DPS
        WHERE InfoCode = '%s'
        AND EventDate <= '%s'
        ORDER BY EventDate DESC
        """ % (str(ic), str(date))
        res = pd.io.sql.read_sql(sql,con).values
        if len(res) > 0:
            return res[0][0]
        else:
            return np.nan

# download timeseries of field of list of identifiers from startdate to enddate, returns dataframe
# field in ['adj close', 'close', 'open', 'high', 'low', 'volume', 'bid', 'ask', 'total return']
def get_timeseries(identifiers, field, startdate, enddate, currency):
    if type(identifiers) == str:
        identifiers = [identifiers]
    if field.lower() not in ['adj close', 'close', 'open', 'high', 'low', 'volume', 'bid', 'ask', 'total return']:
        print('Unknown field name ' + field)
        return np.nan
    elif field.lower() == 'adj close':
        return get_timeseries_adj_close(identifiers, startdate, enddate, currency)
    elif field.lower() == 'total return':
        return get_timeseries_total_return(identifiers, startdate, enddate, currency)	
    else:
        if field.lower() == 'close' or field.lower() == 'open':
            field = field + '_'		
        data = []
        for i in identifiers:
            ic = get_infocode(i)
            if np.isnan(ic):
                df = pd.DataFrame(np.nan,index=[],columns=[i])
            else:
                sql = """
                SELECT MarketDate, %s
                FROM DS2PrimQtPrc
                WHERE InfoCode = %s      
                AND MarketDate >= '%s'
                AND MarketDate <= '%s'
                """ % (field, str(ic), str(startdate), str(enddate))     
                df = pd.io.sql.read_sql(sql,con, index_col='MarketDate')
                df.columns = [i]
            data.append(df)
        return data[0].join(data[1:], how='outer')
		
def get_timeseries_adj_close(identifiers, startdate, enddate, currency):
    if type(identifiers) == str:
        identifiers = [identifiers]
    data = []
    for i in identifiers:
        ic = get_infocode(i)
        if np.isnan(ic):
            df = pd.DataFrame(np.nan,index=[],columns=[i])
        else:
            c = get_currency(i)
            if (c == currency) or (currency == 'loc'):
                sql = """
                SELECT a.MarketDate, a.Close_ * b.CumAdjFactor AS close_adjusted_loc
                FROM DS2PrimQtPrc a, DS2Adj b
                WHERE a.InfoCode = b.InfoCode
                    AND a.MarketDate between b.AdjDate and isnull(b.EndAdjdate, '20790101')
                    AND b.AdjType = 2
                    AND a.InfoCode = %s      
                    AND a.MarketDate >= '%s'
                    AND a.MarketDate <= '%s'
                """ % (str(ic), str(startdate), str(enddate))     
                df = pd.io.sql.read_sql(sql, con, index_col='MarketDate')
                df.columns = [i]
            else:
                sql = """
                SELECT a.MarketDate, a.Close_ * b.CumAdjFactor / fxr.MidRate AS close_adjusted
                FROM DS2PrimQtPrc a, DS2Adj b, DS2FXCode fxc, DS2FXRate fxr
                WHERE fxc.FromCurrCode = '%s'
                    AND fxc.ToCurrCode = '%s'
                    AND fxc.RateTypeCode = 'SPOT'
                    AND fxr.ExRateIntCode = fxc.ExRateIntCode
                    AND fxr.ExRateDate = a.MarketDate
                    AND a.InfoCode = b.InfoCode
                    AND a.MarketDate between b.AdjDate and isnull(b.EndAdjdate, '20790101')
                    AND b.AdjType = 2
                    AND a.InfoCode = '%s'      
                    AND a.MarketDate >= '%s'
                    AND a.MarketDate <= '%s'
                """ % (str(c), str(currency), str(ic), str(startdate), str(enddate))
                df = pd.io.sql.read_sql(sql, con, index_col='MarketDate')
                df.columns = [i]
        if c == 'GBP':
            df = df / 100.
        data.append(df)
    return data[0].join(data[1:], how='outer')

# download return indices of list of isins from startdate to enddate, returns dataframe
def get_timeseries_total_return(identifiers, startdate, enddate, currency):
    if type(identifiers) == str:
        identifiers = [identifiers]
    data = []
    for i in identifiers:
        ic = get_infocode(i)
        if np.isnan(ic):
            df = pd.DataFrame(np.nan,index=[],columns=[i])
        else:
            c = get_currency(i)
            if (c == currency) or (currency == 'loc'):
                sql = """
                SELECT MarketDate, RI
                FROM DS2PrimQtRI
                WHERE InfoCode = '%s'
                    AND MarketDate >= '%s'
                    AND MarketDate <= '%s'
                """ % (str(ic), str(startdate), str(enddate))     
                df = pd.io.sql.read_sql(sql, con, index_col='MarketDate')
                df.columns = [i]
            else:
                sql = """
                SELECT r.MarketDate, r.RI / fxr.MidRate
                FROM DS2PrimQtRI r, DS2FXCode fxc, DS2FXRate fxr
                WHERE fxc.FromCurrCode = '%s'
                    AND fxc.ToCurrCode = '%s'
                    AND fxc.RateTypeCode = 'SPOT'
                    AND fxr.ExRateIntCode = fxc.ExRateIntCode
                    AND fxr.ExRateDate = r.MarketDate
                    AND r.InfoCode = '%s'
                    AND r.MarketDate >= '%s'
                    AND r.MarketDate <= '%s'
                """ % (str(c), str(currency), str(ic), str(startdate), str(enddate))
                df = pd.io.sql.read_sql(sql, con, index_col='MarketDate')
                df.columns = [i]
        if c == 'GBP':
            df = df / 100.
        data.append(df)
    return data[0].join(data[1:], how='outer')

def get_close(identifier, ddate, currency,infoc=0):

    if infoc==0:
        ic = get_infocode(identifier)
    elif infoc!=0:
        ic = infoc       
    if np.isnan(ic):    
        #return np.nan
        pass
    else:
        c = get_currency(ic, ic)
        if (c == currency) or (currency == 'loc'):
            sql = """
            SELECT MarketDate, Close_
            FROM DS2PrimQtPrc
            WHERE InfoCode = '%s'
                AND MarketDate <= '%s'
            """ % (str(ic), str(ddate))     
            df = pd.io.sql.read_sql(sql, con)
            df.columns = ['Date', ic]
        else:
            sql = """
            SELECT r.MarketDate, r.Close_ / fxr.MidRate
            FROM DS2PrimQtPrc r, DS2FXCode fxc, DS2FXRate fxr
            WHERE fxc.FromCurrCode = '%s'
                AND fxc.ToCurrCode = '%s'
                AND fxc.RateTypeCode = 'SPOT'
                AND fxr.ExRateIntCode = fxc.ExRateIntCode
                AND fxr.ExRateDate = r.MarketDate
                AND r.InfoCode = '%s'
                AND r.MarketDate <= '%s'
            """ % (str(c), str(currency), str(ic), str(ddate))
            df = pd.io.sql.read_sql(sql, con)
            df.columns = ['Date', ic]
    if c == 'GBP':
        df = df / 100.

    try:
        return float(df.sort_values('Date', ascending=False).head(1)[ic])
    except:
        return np.nan
	
# returns (most recent) fx rate on date
def get_fxrate(fromcurr, tocurr, date):
    if fromcurr == tocurr:
        return 1
    else:
        sqlcode = """
        SELECT ExRateIntCode
        FROM DS2FXCode
        WHERE FromCurrCode = '%s'
            AND ToCurrCode = '%s'
            AND RateTypeCode = 'SPOT'
        """ % (str(fromcurr), str(tocurr))
        try:
       	    exrateintcode = pd.io.sql.read_sql(sqlcode,con).loc[0,'ExRateIntCode']
        except:
            return np.nan
        sqlrate = """
        SELECT MidRate
        FROM DS2FxRate
        WHERE ExRateIntCode = '%s'
            AND ExRateDate <= '%s'
            ORDER BY ExRateDate DESC
        """ % (str(exrateintcode),str(date))
        try:
            dffx = pd.io.sql.read_sql(sqlrate,con)
            dffx = dffx[-dffx.MidRate.isnull()].reset_index(drop=True).copy()
            return dffx.loc[0,'MidRate']
        except:
            return np.nan

# e.g. second Friday in month of the_date is nth_weekday(the_date,2,4)
def nth_weekday(the_date, nth_week, week_day):
    temp = the_date.replace(day=1)
    adj = (week_day - temp.weekday()) % 7
    temp += dt.timedelta(days=adj)
    temp += dt.timedelta(weeks=nth_week-1)
    return temp

def add_months(date, months):
    import calendar
    month = int(date.month - 1 + months)
    year = int(date.year + month / 12)
    month = int(month % 12 + 1)
    day = min(date.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)
        
def get_curr_rate(curr1, curr2, datefrom, dateto):
    sqlstr = """
    select a.ExRateDate as date_, b.ToCurrCode as curr1, b.FromCurrCode as curr2, a.MidRate as value_
    from Ds2FxRate a 
    left join Ds2FxCode b
    on b.ExRateIntCode=a.ExRateIntCode
    where b.FromCurrCode = '%s'
    and b.ToCurrCode = '%s'
    and b.RateTypeCode='SPOT'
    and a.exratedate>='%s' 
    and a.exratedate<='%s'
    order by a.exratedate asc
    """ % (str(curr2), str(curr1), str(datefrom), str(dateto))
    if curr1==curr2:
        d = {'curr1': curr1, 'curr2': curr2, 'value_': 1}
        return pd.DataFrame(data=d, index=[datefrom, dateto])
    else:
        try:
            res= pd.io.sql.read_sql(sqlstr,con)
            res.index = res.date_
            res.index.name = None
            del res['date_']
            return res
        except:
            d = {'curr1': curr1, 'curr2': curr2, 'value_': np.nan}
            return pd.DataFrame(data=d, index=[datefrom, dateto])

# returns ADTV (average daily traded value) in currency curr, window in month
def get_adtv(identifier, date, window, currency):
    ic = get_infocode(identifier)
    if np.isnan(ic):
        return np.nan
    else:
        startdate = str(add_months(dt.datetime.strptime(date, '%Y-%m-%d'), -window))
        c = get_currency(identifier)
        if (c.lower() == currency.lower()) or (currency.lower() == 'loc'):
            sqlprim = """
            SELECT AVG(Close_ * Volume) AS adtv
            FROM DS2PrimQtPrc
            WHERE InfoCode = '%s'
                AND MarketDate >= '%s'
                AND MarketDate <= '%s'
            """ % (str(ic), str(startdate), str(date))
            try:
                resprim = pd.io.sql.read_sql(sqlprim,con).replace([None],[np.nan]).loc[0,'adtv']
            except:
                resprim = np.nan
            sqlscd = """
            SELECT MAX(t.adtv) AS adtv
            FROM (SELECT AVG(Close_ * Volume) AS adtv
            FROM DS2ScdQtPrc
            WHERE InfoCode = '%s'
                AND MarketDate >= '%s'
                AND MarketDate <= '%s'
            GROUP BY ExchIntCode) as t
            """ % (str(ic), str(startdate), str(date))
            try:
                resscd = pd.io.sql.read_sql(sqlscd,con).replace([None],[np.nan]).loc[0,'adtv']
            except:
                resscd = np.nan
            try:
                res = np.nanmax([resprim,resscd])
            except:
                return np.nan
        else:
            sqlprim = """
            SELECT AVG(p.Close_ * p.Volume / fxr.MidRate) AS adtv
            FROM DS2PrimQtPrc p
			JOIN DS2FXCode fxc ON p.ISOCurrCode = fxc.FromCurrCode
            JOIN DS2FXRate fxr ON fxr.ExRateIntCode = fxc.ExRateIntCode
                AND fxr.ExRateDate = (
                SELECT MAX(sub.ExRateDate)
                FROM DS2FXRate AS sub
                WHERE sub.ExRateDate <= p.MarketDate
                AND sub.ExRateIntCode = fxr.ExRateIntCode
                )
            WHERE fxc.ToCurrCode = '%s'
            AND fxc.RateTypeCode = 'SPOT'
            AND p.InfoCode = '%s'
            AND p.MarketDate >= '%s'
            AND p.MarketDate <= '%s'
            """ % (currency, str(ic), str(startdate), str(date))
            try:
                resprim = pd.io.sql.read_sql(sqlprim,con).replace([None],[np.nan]).loc[0,'adtv']
            except:
                resprim = np.nan
            sqlscd = """
            SELECT MAX(t.adtv)
            FROM(SELECT ExchIntCode, AVG(p.Close_ * p.Volume / fxr.MidRate) as adtv
            FROM DS2ScdQtPrc p
			JOIN DS2FXCode fxc ON p.ISOCurrCode = fxc.FromCurrCode
            JOIN DS2FXRate fxr ON fxr.ExRateIntCode = fxc.ExRateIntCode
                AND fxr.ExRateDate = (
                SELECT MAX(sub.ExRateDate)
                FROM DS2FXRate AS sub
                WHERE sub.ExRateDate <= p.MarketDate
                AND sub.ExRateIntCode = fxr.ExRateIntCode
                )
            WHERE fxc.ToCurrCode = '%s'
            AND fxc.RateTypeCode = 'SPOT'
            AND p.InfoCode = '%s'
            AND p.MarketDate >= '%s'
            AND p.MarketDate <= '%s'
            """ % (currency, str(ic), str(startdate), str(date))
            try:
                resscd = pd.io.sql.read_sql(sqlscd,con).replace([None],[np.nan]).loc[0,'adtv']
            except:
                resscd = np.nan
            try:
                res = np.nanmax([resprim,resscd])
            except:
                return np.nan
        if c == 'GBP':
            try:
                return res / 100.
            except:
                return np.nan
        else:
            return res
            
# with a from and to date and gets the timeseries of price and vols from primary exchange
def get_adtv_prim_plus(identifier, date, startdate, currency, infoc=0):
    if infoc==0:
        ic = get_infocode(identifier)
    elif infoc!=0:
        ic = infoc
    if np.isnan(ic):
        return pd.DataFrame(columns=['marketdate','close_','volume','exch'])
    else:
        c = get_currency(identifier, ic)
        if (c.lower() == currency.lower()) or (currency.lower() == 'loc'):
            sqlprimts = """
            SELECT MarketDate, Close_, Volume, ExchIntCode as Exch
            FROM DS2PrimQtPrc
            WHERE InfoCode = ?
                AND MarketDate >= ?
                AND MarketDate <= ?
            """ #% (str(ic), str(startdate), str(date))
            try:   
                #resprimts = pd.io.sql.read_sql(sqlprimts,con) #this is the corresponding time series #old  
                resprimts = pd.io.sql.read_sql(sqlprimts,con, params=[str(ic), str(startdate), str(date)])
                resprimts.columns = ['marketdate','close_','volume','exch']
            except:
                resprimts = pd.DataFrame(columns=['marketdate','close_','volume','exch']) 
        else:
            sqlprimts = """
            SELECT MarketDate, (p.Close_/fxr.MidRate) as Close_, p.Volume, p.ExchIntCode as Exch
            FROM DS2PrimQtPrc p
            JOIN DS2FXCode fxc ON p.ISOCurrCode = fxc.FromCurrCode
            JOIN DS2FXRate fxr ON fxr.ExRateIntCode = fxc.ExRateIntCode
                AND fxr.ExRateDate = (
                SELECT MAX(sub.ExRateDate)
                FROM DS2FXRate AS sub
                WHERE sub.ExRateDate <= p.MarketDate
                AND sub.ExRateIntCode = fxr.ExRateIntCode
                )
            WHERE fxc.ToCurrCode = ?
            AND fxc.RateTypeCode = 'SPOT'
            AND p.InfoCode = ?
            AND p.MarketDate >= ?
            AND p.MarketDate <= ?
            """ #% (currency, str(ic), str(startdate), str(date))
            try:
                #resprimts = pd.io.sql.read_sql(sqlprimts,con)
                resprimts = pd.io.sql.read_sql(sqlprimts,con, params=[currency, str(ic), str(startdate), str(date)])
                resprimts.columns = ['marketdate','close_','volume','exch']
            except:
                resprimts = pd.DataFrame(columns=['marketdate','close_','volume','exch'])
                
        if c == 'GBP':
            try:
                resprimts[resprimts.columns[1]]=resprimts[resprimts.columns[1]]/100.
                return resprimts
            except:
                return pd.DataFrame(columns=['marketdate','close_','volume','exch'])
        else:
            return resprimts
			
# with a from and to date and gets the timeseries of price and vols from secondary exchanges
def get_adtv_sec_plus(identifier, date, startdate, currency, infoc=0):
    if infoc==0:
        ic = get_infocode(identifier)
    elif infoc!=0:
        ic = infoc
    if np.isnan(ic):
        return pd.DataFrame(columns=['marketdate','close_','volume','exch'])
    else:
        c = get_currency(identifier, ic)
        if (c.lower() == currency.lower()) or (currency.lower() == 'loc'):
            sqlscdts = """
            SELECT MarketDate, Close_, Volume, ExchIntCode as Exch
            FROM DS2ScdQtPrc
            WHERE InfoCode = '%s'
                AND MarketDate >= '%s'
                AND MarketDate <= '%s'
            ORDER BY ExchIntCode DESC
            """ % (str(ic), str(startdate), str(date))
            try:
                resscdts = pd.io.sql.read_sql(sqlscdts,con)
                resscdts.columns = ['marketdate','close_','volume','exch']
            except:
                resscdts = pd.DataFrame(columns=['marketdate','close_','volume','exch'])
        else:
            sqlscdts = """
            SELECT MarketDate, (p.Close_/fxr.MidRate) as Close_, p.Volume, p.ExchIntCode as Exch
            FROM DS2ScdQtPrc p
			JOIN DS2FXCode fxc ON p.ISOCurrCode = fxc.FromCurrCode
            JOIN DS2FXRate fxr ON fxr.ExRateIntCode = fxc.ExRateIntCode
                AND fxr.ExRateDate = (
                SELECT MAX(sub.ExRateDate)
                FROM DS2FXRate AS sub
                WHERE sub.ExRateDate <= p.MarketDate
                AND sub.ExRateIntCode = fxr.ExRateIntCode
                )
            WHERE fxc.ToCurrCode = '%s'
            AND fxc.RateTypeCode = 'SPOT'
            AND p.InfoCode = '%s'
            AND p.MarketDate >= '%s'
            AND p.MarketDate <= '%s'
            """ % (currency, str(ic), str(startdate), str(date))         
            try:
                resscdts = pd.io.sql.read_sql(sqlscdts,con)
                resscdts.columns = ['marketdate','close_','volume','exch']
            except:
                resscdts = pd.DataFrame(columns=['marketdate','close_','volume','exch'])              
                
        if c == 'GBP':
            try:
                resscdts[resscdts.columns[1]]=resscdts[resscdts.columns[1]]/100.
                return resscdts
            except:
                return pd.DataFrame(columns=['marketdate','close_','volume','exch'])
        else:
            return resscdts

def get_adtv_fromto(identifier, date, startdate, currency):
    ic = get_infocode(identifier)
    if np.isnan(ic):
        return np.nan
    else:
        c = get_currency(identifier)
        if (c.lower() == currency.lower()) or (currency.lower() == 'loc'):
            sqlprim = """
            SELECT AVG(Close_ * Volume) AS adtv
            FROM DS2PrimQtPrc
            WHERE InfoCode = '%s'
                AND MarketDate >= '%s'
                AND MarketDate <= '%s'
            """ % (str(ic), str(startdate), str(date))
            try:
                resprim = pd.io.sql.read_sql(sqlprim,con).replace([None],[np.nan]).loc[0,'adtv']
            except:
                resprim = np.nan
                
            sqlscd = """
            SELECT MAX(t.adtv) AS adtv
            FROM (SELECT AVG(Close_ * Volume) AS adtv
            FROM DS2ScdQtPrc
            WHERE InfoCode = '%s'
                AND MarketDate >= '%s'
                AND MarketDate <= '%s'
            GROUP BY ExchIntCode) as t
            """ % (str(ic), str(startdate), str(date))
            try:
                resscd = pd.io.sql.read_sql(sqlscd,con).replace([None],[np.nan]).loc[0,'adtv']
            except:
                resscd = np.nan

        else:
            sqlprim = """
            SELECT AVG(p.Close_ * p.Volume / fxr.MidRate) AS adtv
            FROM DS2PrimQtPrc p, DS2FXCode fxc, DS2FXRate fxr
            WHERE p.ISOCurrCode = fxc.FromCurrCode
                AND fxc.ToCurrCode = '%s'
                AND fxc.RateTypeCode = 'SPOT'
                AND fxr.ExRateIntCode = fxc.ExRateIntCode
                AND fxr.ExRateDate = p.MarketDate
                AND p.InfoCode = '%s'
                AND p.MarketDate >= '%s'
                AND p.MarketDate <= '%s'
            """ % (currency, str(ic), str(startdate), str(date))
            try:
                resprim = pd.io.sql.read_sql(sqlprim,con).replace([None],[np.nan]).loc[0,'adtv']
            except:
                resprim = np.nan

            sqlscd = """
            SELECT MAX(t.adtv)
            FROM(SELECT ExchIntCode, AVG(p.Close_ * p.Volume / fxr.MidRate) as adtv
            FROM DS2ScdQtPrc p, DS2FXCode fxc, DS2FXRate fxr
            WHERE p.ISOCurrCode = fxc.FromCurrCode
                AND fxc.ToCurrCode = '%s'
                AND fxc.RateTypeCode = 'SPOT'
                AND fxr.ExRateIntCode = fxc.ExRateIntCode
                AND fxr.ExRateDate = p.MarketDate
                AND p.InfoCode = '%s'
                AND p.MarketDate >= '%s'
                AND p.MarketDate <= '%s'
            GROUP BY ExchIntCode) as t
            """ % (currency, str(ic), str(startdate), str(date))            
            try:
                resscd = pd.io.sql.read_sql(sqlscd,con).replace([None],[np.nan]).loc[0,'adtv']
            except:
                resscd = np.nan
                
        try:
            res = np.nanmax([resprim,resscd])
        except:
            return np.nan
                
                
        if c == 'GBP':
            try:
                return res / 100.
            except:
                return np.nan
        else:
            return res
         
		 
def get_div_yield(d, infocode): # d is date in date dt.date format
    sql = """
    DECLARE 
        @cutoff_date DATE = '%s',
        @18m_back DATE = '%s',
        @target_currency NVARCHAR(3) = 'EUR'


    ;WITH allPrices AS (
                SELECT 
                    c.InfoCode,
                    p.MarketDate,
                    CASE
                        WHEN ex.PriceUnit = 'E-02' THEN (p.Close_/ISNULL(r.MidRate,1)) / 100
                        ELSE (p.Close_/ISNULL(r.MidRate,1))
                    END AS close_EUR
                FROM Ds2CtryQtInfo AS c
                JOIN DS2PrimQtPrc AS p
                    ON p.InfoCode = c.InfoCode
                    AND p.MarketDate = (SELECT MAX(sub.MarketDate) FROM DS2PrimQtPrc AS sub
                                        WHERE sub.InfoCode = p.InfoCode 
                                            AND sub.MarketDate BETWEEN @18m_back AND @cutoff_date)
                    AND p.RefPrcTypCode = 1
                JOIN DS2ExchQtInfo AS ex
                    ON ex.InfoCode = p.InfoCode
                    AND ex.ExchIntCode = p.ExchIntCode
                LEFT JOIN DS2FXCode AS fxc
                    ON fxc.RateTypeCode = 'SPOT'
                    AND (fxc.FromCurrCode = ex.ISOCurrCode AND fxc.ToCurrCode = @target_currency )
                LEFT JOIN DS2FXRate AS r
                    ON r.ExRateIntCode = fxc.ExRateIntCode
                    AND r.ExRateDate = (
                        SELECT MAX(sub.ExRateDate)
                        FROM DS2FXRate AS sub
                        WHERE sub.ExRateIntCode = r.ExRateIntCode
                        AND sub.ExRateDate <= p.MarketDate
                    )
            ), allDPS AS (
                SELECT
                    c.InfoCode,
                    a.DivTypeCode,
                    ISNULL(a.ISOCurrCode, d.ISOCurrCode) AS ISOCurrCode,
                    b.EventDate AS DPS_date,
                    a.EffectiveDate,
                    a.DivRate,
                    b.DPS
                FROM Ds2CtryQtInfo AS c
                JOIN DS2DPS AS b
                    ON b.InfoCode = c.InfoCode
                    AND b.EventDate BETWEEN @18m_back AND @cutoff_date
                JOIN DS2Div AS d
                    ON d.InfoCode = b.InfoCode
                    AND d.EffectiveDate = (
                        SELECT MAX(EffectiveDate)
                        FROM DS2Div AS sub
                        WHERE sub.InfoCode = d.InfoCode
                            AND sub.EffectiveDate < @cutoff_date
                    )
                LEFT JOIN DS2Div AS a
                    ON a.InfoCode = b.InfoCode
                    AND a.AnnouncedDate = b.EventDate
            ), filterDPS AS (
                SELECT DISTINCT
                    a.InfoCode,
                    a.DPS_date,
                    a.EffectiveDate,
                    a.ISOCurrCode,
                    CASE
                        WHEN a.DPS - ISNULL(b.DivRate, 0) < 0 THEN a.DPS
                        ELSE a.DPS - ISNULL(b.DivRate, 0)
                    END AS DPS
                FROM allDPS AS a
                LEFT JOIN allDPS AS b
                    ON b.InfoCode = a.InfoCode
                    AND a.DPS_date = b.DPS_date
                    AND b.DivTypeCode = 'EXT'
                WHERE a.DPS_date =
                    (   SELECT MAX(sub.DPS_date) FROM allDPS AS sub
                        WHERE sub.InfoCode = a.InfoCode
                            AND sub.DPS_date <= @cutoff_date
                    )
            ), dpsPrice AS (
            SELECT
                p.InfoCode,
                p.MarketDate AS price_date,
                p.close_EUR,
                d.DPS_date,
                d.EffectiveDate,
                d.DPS,
                d.DPS / ISNULL(r.MidRate,1) AS DPS_EUR,
                (d.DPS * ISNULL((ad.CumAdjFactor/ap.CumAdjFactor),1))/ ISNULL(r.MidRate,1) AS DPS_EUR_adj
            FROM allPrices AS p
            JOIN filterDPS AS d
                ON d.InfoCode = p.InfoCode
            LEFT JOIN DS2Adj AS ap
                ON ap.InfoCode = p.InfoCode
                AND ap.AdjType = 2
                AND p.MarketDate BETWEEN ap.AdjDate AND ISNULL(ap.EndAdjDate,'9999-12-31')
            LEFT JOIN DS2Adj AS ad
                ON ad.InfoCode = p.InfoCode
                AND ad.AdjType = 2
                AND d.DPS_date BETWEEN ad.AdjDate AND ISNULL(ad.EndAdjDate,'9999-12-31')
            LEFT JOIN DS2FXCode AS fxc
                ON fxc.RateTypeCode = 'SPOT'
                AND (fxc.FromCurrCode = d.ISOCurrCode AND fxc.ToCurrCode =  @target_currency )
            LEFT JOIN DS2FXRate AS r
                ON r.ExRateIntCode = fxc.ExRateIntCode
                AND r.ExRateDate = (
                    SELECT MAX(sub.ExRateDate)
                    FROM DS2FXRate AS sub
                    WHERE sub.ExRateIntCode = r.ExRateIntCode
                    AND sub.ExRateDate <= p.MarketDate
                )
            )
            SELECT
                price_date,
                DPS_date,
                a.infoCode,
                b.Region,
                close_EUR,
                CASE
                    WHEN b.Region = 'BE' THEN DPS/0.8 
                    ELSE DPS 
                END AS DPS,
                CASE
                    WHEN b.Region = 'BE' THEN DPS_EUR_adj/0.8 
                    ELSE DPS_EUR_adj 
                END AS DPS_EUR_adj,
                CASE
                    WHEN b.Region = 'BE' THEN (DPS_EUR_adj/0.8 )/close_EUR
                    ELSE DPS_EUR_adj/close_EUR
                END AS div_yield
            FROM dpsPrice as a
            JOIN DS2CtryQtInfo AS b
                ON b.InfoCode = a.InfoCode
                AND b.IsPrimQt = 1
            WHERE DPS_date =
                (   SELECT MAX(sub.DPS_date) FROM dpsPrice AS sub
                    WHERE sub.InfoCode = a.InfoCode
                        AND sub.DPS_date = a.DPS_date
                )
            AND a.InfoCode IN ('%s')
    """
    try:
        #res = pd.io.sql.read_sql(sql % (dt.date.strftime(cutoff,'%Y-%m-%d'),dt.date.strftime(cutoff18,'%Y-%m-%d'),infoCode), con)
        res = pd.io.sql.read_sql(sql % (d, myf.add_months(d,-18), infocode), con)
        return res
    except:
        return pd.DataFrame(columns=['price_date','DPS_date','infoCode','Region','close_EUR','DPS','DPS_EUR_adj','div_yield'])
		
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
    vc = pd.io.sql.read_sql(sql % ('','X','', sedol[:6], venType), con)
    if len(vc)== 0:
        vc = pd.io.sql.read_sql(sql % ('G','','G', sedol[:6], venType), con)
    if len(vc) == 0:
        return np.nan
    else:
        return vc.loc[0,'VenCode']

def get_wspit_table(wspitItem):
    sql_table = """
    SELECT Desc_
    FROM WSPITDesc
    WHERE Code = (SELECT Left(TableCode,1)
                  FROM WSPITITEM
                  WHERE WSPITITEM.Item = %s)
        AND Type_ = 1
    """
    try:
        return pd.io.sql.read_sql(sql_table % str(wspitItem),con).loc[0,'Desc_']
    except:
        return np.nan

def get_multi_wspit(wspitCode, wspitItemlist, tabla):
    """Return the WSPIT item list
    """
    sql_value = """
    SELECT *
    FROM %s
    WHERE Item in (%s)
        AND Code = %s
    """
    try:
        return pd.io.sql.read_sql(sql_value % (tabla, wspitItemlist, str(wspitCode)), con)
    except:
        return pd.DataFrame()
        
def get_wspit_format(dfx):

    lstfreq = [[1,2,False,'A'], [8,3,True,'Q'], [10,5,True,'S'], [9,4,True,'R']]
    dfx_ = pd.DataFrame()

    for item in dfx.Item.drop_duplicates():
        dftemp1_ = dfx[dfx.Item==item].sort(['CalPrdEndDate', 'PointDate'], ascending=[False, False])
        for fper in dftemp1_.FiscalPrd.drop_duplicates():  
            dftemp2_ = dftemp1_[(dftemp1_.FiscalPrd==fper)]
            for freq_ in lstfreq:
                dftemp3_ = dftemp2_[(dftemp2_.FreqCode.isin([freq_[0], freq_[1]]))]
                dftemp3_ = dftemp3_.sort(['PointDate','FreqCode'], ascending = [False, freq_[2]]).reset_index(drop=True).head(1)
                if len(dftemp3_)>0:
                    dftemp3_['FreqCode'] = freq_[3]
                dfx_ = pd.concat([dfx_, dftemp3_], axis=0)

    dfx_ = dfx_[[0,4,2,3,6,1,5]]
    #dfx_.columns = ['code','item','freq','year_','seq','date_','value_']

    return dfx_.reset_index(drop=True)
	
def get_wspit_data(sedol6, item):
    vencode = get_vencode(sedol6, 35)
    tabla = get_wspit_table(item)
    dfx = get_multi_wspit(vencode, item, tabla)
    return dfx.sort_values(['CalPrdEndDate', 'PointDate'], ascending=[False, False])
	
	#example:
	#vencode = get_vencode('690064', 35)
	#tabla = get_wspit_table(1551)
	#dfx = get_multi_wspit(vencode, 1551, tabla)
	