import pandas as pd
import sys
import numpy as np
import datetime as dt
import qadconnect34plus as q 

def get_4d_comp_file(df, cutdt, indexsymbol, internal_symbol_col): #date in format: 20151013
    df = get_4d_wf_fields(df, indexsymbol)
    df = df[['valid_from', 'valid_to', 'index_symbol',  internal_symbol_col, 'size', 'description', 'not_rep_before']]
    df.columns = ['valid_from', 'valid_to', 'index_symbol', 'dj_id', 'size', 'description', 'not_rep_before']
    return df

def get_4d_wf_file(df, cutdt, indexsymbol, internal_symbol_col): #date in format: 20151013
    dfprod = get_prod_close_EUR(cutdt)
    df = pd.merge(df, dfprod[['ISIN','internal_key','close_eur']], left_on=internal_symbol_col, right_on='internal_key', how='left')
    df = get_4d_wf_fields(df, indexsymbol)
    df['weightfactor'] = np.around(100000000000*df['weight'] / df['close_eur'],0)
    df = df[['valid_from', 'valid_to', 'index_symbol', internal_symbol_col, 'weightfactor', 'capfactor', 'description', 'not_rep_before']]
    df.columns = ['valid_from', 'valid_to', 'index_symbol', 'dj_id', 'weightfactor', 'capfactor', 'description', 'not_rep_before']
    return df

def get_prod_close_EUR(dt): #date in format: 20151013
    df = pd.read_csv('M:/Production/FinalSheets/s6/archive/stoxx_global_'+ dt +'.txt', sep=';')
    df=trim_rows_cols(df)
    return df

def get_4d_wf_fields(df, indexsymbol): #date is the cut-off for wf calculation
    yesterday = str(dt.date.today()-dt.timedelta(days=1)).replace('-','')
    df['valid_from'] = yesterday
    df['valid_to'] = 99991231
    df['index_symbol'] = indexsymbol
    df['size'] = 'Y'
    df['description'] = np.nan
    df['not_rep_before'] = yesterday
    df['weightfactor'] = 1
    df['capfactor'] = 1
    return df

def get_im_sub(a_dir):
    import os, glob
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def trim_rows_cols(df):
    cols=df.columns
    cols=cols.map(lambda x: x.strip())
    df.columns=cols
    for c in df.columns:
        try: 
            df[c] = df[c].map(lambda x: x.strip())
        except:
            a=1
    return df

def get_web_file(fpath):
    import requests
    from requests.auth import HTTPBasicAuth
    with open('creds.txt') as c: #creds.txt file contains: name.surname@stoxx.com,pass
        creds = c.read()
    creds=creds.split(',')
    auth = HTTPBasicAuth(creds[0],creds[1])
    r = requests.get(fpath, auth=auth)
    text = r.text
    rows = text.split('\n')[1:]
    data = [x.split(';') for x in rows if x!='']
    return pd.DataFrame(data)

def get_web_h(idxlist, usedates=False, dfrom='1.1.1980', dto='1.1.2050', special=False):

    import requests
    from requests.auth import HTTPBasicAuth
    with open('C:/Users/iv822/Documents/Python Scripts/creds.txt') as c: #creds.txt file contains: name.surname@stoxx.com,pass
        creds = c.read()
    creds=creds.split(',')
    auth = HTTPBasicAuth(creds[0],creds[1])
    proxyDict = { 
                  "https"  : 'https://webproxy-fra.deutsche-boerse.de:8080'
                }

    if usedates==True:
        dfrom=dfrom
        dto=dto

    for f in idxlist:
        #url='https://www.stoxx.com/document/Indices/Current/HistoricalData/h_'+f.lower()+'.txt'
        url='https://www.stoxx.com/download/historical_data/h_'+f.lower()+'.txt'
        r = requests.get(url, auth=auth, proxies=proxyDict)
        text = r.text
        rows = text.split('\n')[1:]
        if special==False:
            try:
                data = [x.split(';')[:-1] for x in rows if x!='']
                df = pd.DataFrame(data, columns=['Date','Symbol','Indexvalue'])
            except:
                data = [x.split(';') for x in rows if x!='']
                df = pd.DataFrame(data, columns=['Date','Symbol','Indexvalue'])
            del df['Symbol']
        elif special==True:
            data = [x.split(';') for x in rows if x!='']
            df = pd.DataFrame(data, columns=['Date','Indexvalue'])
            df = df.loc[1:,:]
        df['Indexvalue'] = df['Indexvalue'].map(lambda x: float(x))
        df = df.rename(columns={'Indexvalue':f})
        if idxlist.index(f)==0:
            dfres=df
        else:
            dfres=pd.merge(dfres, df, how='outer', on='Date') 
    dfres['Date']=pd.to_datetime(dfres['Date'], format='%d.%m.%Y', dayfirst=True)
    if usedates:
        dfres=filterdts(dfrom, dto, dfres)
        dfres=dfres.sort('Date', ascending=True)
		
    return dfres


def get_h(idxlist, floc, usedates=False, dfrom='1.1.1980', dto='1.1.2050', special=False):

    for idx in idxlist:
        print('http://www.stoxx.com/download/historical_data/h_'+idx.lower()+'.txt')
        #pass

    if usedates==True:
        dfrom=dfrom
        dto=dto

    for f in idxlist:

        df = pd.read_csv(floc + 'h_'+f.lower()+'.txt', sep=';')
        df['Indexvalue'] = df['Indexvalue'].map(lambda x: float(x))
        df = df.rename(columns={'Indexvalue':f})

        df = df[['Date', f]]
        if idxlist.index(f)==0:
            dfres=df
        else:
            dfres=pd.merge(dfres, df, how='outer', on='Date') 
    dfres['Date']=pd.to_datetime(dfres['Date'], format='%d.%m.%Y', dayfirst=True)
    if usedates:
        dfres=filterdts(dfrom, dto, dfres)

    return dfres

#calculates DY for each year
#gets dataframe with columns: date, price, net, gross; outputs dataframe; Date need to be in date format (parse_dates=[0], dayfirst=True)
def calc_an_div_yields(df):
    dates=pd.DatetimeIndex(df.iloc[:,0])
    isyearend=(dates.month[0:len(dates)-1]>dates.month[1:len(dates)])
    dates1=dates[:len(dates)]
    yearend=dates1[isyearend]
    dflastday=pd.DataFrame(yearend)
    df1=pd.merge(dflastday, df, left_on=dflastday.columns[0], right_on=df.columns[0])
    del df1[df1.columns[0]]
    df1.index=df1[df1.columns[0]]
    df1.index.name = None
    del df1[df1.columns[0]]
    returns=np.array(df1.iloc[1:len(df1),:])/np.array(df1.iloc[0:len(df1)-1,:])-1
    dy=[returns[:,1]-returns[:,0], returns[:,2]-returns[:,0]]
    dfdy=np.round(pd.DataFrame(np.transpose(dy)),4)
    dfdy=dfdy.rename(columns={dfdy.columns[0]:'net_dy', dfdy.columns[1]:'gross_dy'})
    yrlist=((yearend[i].year+1) for i in range(len(yearend)-1))
    dfdy['yr']=list(yrlist)
    dfdy.index=dfdy['yr']
    dfdy.index.name=None
    del dfdy['yr']  
    return dfdy

def calc_an_rets(df):
    dates=pd.DatetimeIndex(df.iloc[:,0])
    isyearend=(dates.month[0:len(dates)-1]>dates.month[1:len(dates)])
    dates1=dates[:len(dates)]
    yearend=dates1[isyearend]
    dflastday=pd.DataFrame(yearend)
    df1=pd.merge(dflastday, df, left_on=dflastday.columns[0], right_on=df.columns[0])
    del df1[df1.columns[0]]
    df1.index=df1[df1.columns[0]]
    df1.index.name = None
    del df1[df1.columns[0]]
    returns=np.array(df1.iloc[1:len(df1),:])/np.array(df1.iloc[0:len(df1)-1,:])-1
    return pd.DataFrame(returns, columns = df1.columns, index=df1.index[1:])
	
#calculates DY for each month, quarter, year
#gets dataframe with columns: date, price, net, gross; outputs dataframe; Date need to be in date format (parse_dates=[0], dayfirst=True)
def calc_div_yields_ext(df):
    dates=pd.DatetimeIndex(df.iloc[:,0])
    datesq=dates[dates.map(lambda x: x.month==3 or x.month==6 or x.month==9 or x.month==12)]

    isyearend=(dates.month[0:len(dates)-1]>dates.month[1:len(dates)])
    isquarterend =(datesq.day[0:len(datesq)-1]>datesq.day[1:len(datesq)])
    ismonthend=(dates.day[0:len(dates)-1]>dates.day[1:len(dates)])

    dates1=dates[:len(dates)]
    dates2=datesq[:len(datesq)]

    period_end=[np.nan, np.nan, np.nan]
    period_end[0] = dates1[isyearend]
    period_end[1] = dates2[isquarterend]
    period_end[2] = dates1[ismonthend]

    dfres = pd.DataFrame()
    for k in range(len(period_end)):
        p_end = period_end[k]
        dflastday=pd.DataFrame(p_end)
        dflastday=pd.concat([dflastday, df.iloc[[0,len(df)-1],0]], axis=0).sort(0, ascending=True).drop_duplicates().reset_index(drop=True)
        df1=pd.merge(dflastday, df, left_on=dflastday.columns[0], right_on=df.columns[0])
        df1
        period = df1.loc[:,0].map(lambda x: str(x)[:7])

        del df1[df1.columns[0]]
        df1.index=df1[df1.columns[0]]
        df1.index.name = None
        del df1[df1.columns[0]]

        returns=np.array(df1.iloc[1:len(df1),:])/np.array(df1.iloc[0:len(df1)-1,:])-1
        annreturns=[] #observation count
        for i in range(len(df1.index)-1):
            ct = len(df[(df.Date>df1.index[i]) & (df.Date<=df1.index[i+1])])
            annreturns.append(np.round((returns[i]+1)**(250/ct)-1,18))

        dyact=[returns[:,1]-returns[:,0], returns[:,2]-returns[:,0]]
        dfdyann=pd.DataFrame(np.transpose([np.transpose(annreturns)[1]-np.transpose(annreturns)[0],np.transpose(annreturns)[2]-np.transpose(annreturns)[0]]))
        dfdyact=np.round(pd.DataFrame(np.transpose(dyact)),18)

        dfdy=pd.concat([dfdyact,dfdyann], axis=1)
        dfdy['period_end']= period[1:].reset_index(drop=True)
        if k==0:
            list1=np.array(dfdy)
        elif k==1:
            list2=np.array(dfdy)
        elif k==2:
            list3=np.array(dfdy) 

    dfres = pd.concat([pd.DataFrame(list1), pd.DataFrame(list2), pd.DataFrame(list3)], 
                           keys=['Yearly', 'Quarterly', 'Monthly'])
    dfres.columns=['DY_net_act','DY_gross_act','DY_net_ann','DY_gross_ann', 'period_end']
    cols=dfres.columns.tolist()
    dfres=dfres[cols[-1:] + cols[:-1]] 

    return dfres

def calc_roll_dy(df, freq, window):
    """Returns the rolling dividend yields actual and annualized with 250 observations for a year
    
    Keyword arguments:
    df -- dataframe with columns: date, price, net, gross
    freq -- 'y', 'q', 'm' (yearly, quarterly, monthly)
    window -- window in years (1, 2, 3 ..)
    """
    dates=pd.DatetimeIndex(df.iloc[:,0])

    if freq=='y':
        stp = 1
        isyearend=(dates.month[0:len(dates)-1]>dates.month[1:len(dates)])
        period_end = dates[numpy.append(isyearend, True)]
    elif freq=='q':
        stp = 4
        datesq=dates[dates.map(lambda x: x.month==3 or x.month==6 or x.month==9 or x.month==12)]
        isquarterend =(datesq.day[0:len(datesq)-1]>datesq.day[1:len(datesq)])
        period_end = datesq[np.append(isquarterend, True)]
    else:
        stp = 12
        ismonthend=(dates.day[0:len(dates)-1]>dates.day[1:len(dates)])
        period_end = dates[np.append(ismonthend, True)]

    dflastday=pd.DataFrame(period_end)
    dflastday=pd.concat([dflastday, df.iloc[[0,len(df)-1],0]], axis=0).sort(0, ascending=True).drop_duplicates().reset_index(drop=True)
    df1 = pd.merge(dflastday, df, left_on=dflastday.columns[0], right_on=df.columns[0])

    del df1[df1.columns[0]]
    df1.index=df1[df1.columns[0]]
    df1.index.name = None
    del df1[df1.columns[0]]

    per = []; returns = []; annreturns = []

    for x in range(1,len(df1)):
        stp1 = stp * window
        if x-stp1>=0:
            per.append((str(df1.index[x-stp1])[:10]+' / '+ str(df1.index[x])[:10]))
            ct = len(df[(df.Date>df1.index[x-stp1]) & (df.Date<=df1.index[x])])
            rets = np.array(df1.iloc[x,:]) / np.array(df1.iloc[x-stp1,:])
            returns.append(list(rets-1))
            annreturns.append(list((rets)**(250/ct)-1))
        else:
            per.append(np.nan)
            returns.append([np.nan,np.nan,np.nan])
            annreturns.append([np.nan,np.nan,np.nan])

    dyact = pd.DataFrame([np.array(returns)[:,1]-np.array(returns)[:,0], np.array(returns)[:,2]-np.array(returns)[:,0]])
    dyann = pd.DataFrame([np.array(annreturns)[:,1]-np.array(annreturns)[:,0], np.array(annreturns)[:,2]-np.array(annreturns)[:,0]])

    dfdy = pd.concat([pd.DataFrame(per), dyact.T, dyann.T], axis=1)
    dfdy.columns=['period', 'DY_net_act','DY_gross_act','DY_net_ann','DY_gross_ann']
    dfdy = dfdy[-dfdy.period.isnull()]
    
    return dfdy

#calculates DY for last 1y, 3y and 5y periods, receives a dataframe with columns: date, price, net, gross; outputs dataframe 
def calc_stats_years(df1):
    yrs =['5y','3y','1y']
    dto=df1.iloc[len(df1)-1,0]
    df1.reset_index(inplace=True,drop=True)
    df1.fillna(method='pad', inplace=True)
    
    dt5 = add_months(dto,-12*5)
    dt5 = df1[df1.Date>=dt5].iloc[0,0]
    dt3 = add_months(dto,-12*3)
    dt3 = df1[df1.Date>=dt3].iloc[0,0]
    dt1 = add_months(dto,-12)
    dt1 = df1[df1.Date>=dt1].iloc[0,0]
    dt0 = add_months(dto,-0)
    dt0 = df1[df1.Date>=dt0].iloc[0,0]
    
    if dt5.weekday()==5:
        dt5=dt5-dt.timedelta(days=1)
    elif dt5.weekday()==6:
        dt5=dt5-dt.timedelta(days=2)
        
    if dt3.weekday()==5:
        dt3=dt3-dt.timedelta(days=1)
    elif dt3.weekday()==6:
        dt3=dt3-dt.timedelta(days=2)
        
    if dt1.weekday()==5:
        dt1=dt1-dt.timedelta(days=1)
    elif dt1.weekday()==6:
        dt1=dt1-dt.timedelta(days=2)
        
    if dt0.weekday()==5:
        dt0=dt0-dt.timedelta(days=1)
    elif dt0.weekday()==6:
        dt0=dt0-dt.timedelta(days=2)
    
    dtlst=[dt5, dt3, dt1, dt0] #3 dates
    ct5y = len(df1[(df1.Date <= dtlst[3]) & (df1.Date > dtlst[0])]) #observation counts
    ct3y = len(df1[(df1.Date <= dtlst[3]) & (df1.Date > dtlst[1])])
    ct1y = len(df1[(df1.Date <= dtlst[3]) & (df1.Date > dtlst[2])])

    #dfvals=df1[df1.Date.isin(dtlst)==True]
    dfvals=pd.DataFrame()
    for fe in dtlst:
        dfvals=pd.concat([dfvals, pd.DataFrame(df1[df1.Date<=fe].sort('Date', ascending=False).reset_index(drop=True).iloc[0,:]).T], axis=0)
    dfvals.reset_index(inplace=True,drop=True)
    actret=np.array(dfvals.iloc[len(dfvals)-1,1:4])/np.array(dfvals.iloc[0:len(dfvals)-1,1:4])-1

    dfvals.reset_index(inplace=True,drop=True)
    actret=np.array(dfvals.iloc[len(dfvals)-1,1:])/np.array(dfvals.iloc[0:len(dfvals)-1,1:])-1
    
    #actual returns
    dfactret=pd.DataFrame(actret)
    dfactret.columns=df1.columns[1:]
    dfactret['years']=yrs 
    list1=np.array(dfactret)

    #annualized returns
    dfannret=dfactret
    dfannret.iloc[0,:3]=dfannret.iloc[0,:3].map(lambda x: (x+1)**(250/ct5y)-1)
    dfannret.iloc[1,:3]=dfannret.iloc[1,:3].map(lambda x: (x+1)**(250/ct3y)-1)
    dfannret.iloc[2,:3]=dfannret.iloc[2,:3].map(lambda x: (x+1)**(250/ct1y)-1)
    list2=np.array(dfannret)

    #vola
    returns=np.array(df1.iloc[1:len(df1),1:])/np.array(df1.iloc[0:len(df1)-1,1:])-1
    dfr=pd.DataFrame(returns)
    dfr['Date']=list(df1.loc[1:,'Date'])
    vol=[]
    vol.append(np.std(dfr[(dfr.Date <= dtlst[3]) & (dfr.Date > dtlst[0])].iloc[:,:3], ddof=1)*np.sqrt(250))
    vol.append(np.std(dfr[(dfr.Date <= dtlst[3]) & (dfr.Date > dtlst[1])].iloc[:,:3], ddof=1)*np.sqrt(250))
    vol.append(np.std(dfr[(dfr.Date <= dtlst[3]) & (dfr.Date > dtlst[2])].iloc[:,:3], ddof=1)*np.sqrt(250))
    dfvola=pd.DataFrame(vol)
    dfvola.columns=df1.columns[1:]
    dfvola['years']=yrs
    list3=np.array(dfvola)

    #dys
    dfdys=pd.DataFrame(list2)
    dfdys.iloc[:,1]=dfdys.iloc[:,1]-dfdys.iloc[:,0]
    dfdys.iloc[:,2]=dfdys.iloc[:,2]-dfdys.iloc[:,0]
    dfdys.iloc[:,0]=np.nan
    list4=np.array(dfdys)
    
    #max drawdown
    mxdd = []
    for d in range(len(dtlst[1:])):
        dft = df1[(df1.Date <= dtlst[3]) & (df1.Date > dtlst[d])]
        pk = np.zeros((len(dft)+1,3))
        dd = np.zeros((len(dft)+1,4))
        h = np.array(dft.iloc[:,1:])
        pk[0] = h[0]
        for i in range(len(h)):
            for j in range(3):
                pk[i+1,j] = h[i,j] if h[i,j] > pk[i,j] else pk[i,j]
                dd[i+1,j] = h[i,j] / pk[i+1,j] - 1 if h[i,j] < pk[i+1,j] else 0
        dd = dd[1:]
        mxdd.append((abs(dd[:,0].min()), abs(dd[:,1].min()), abs(dd[:,2].min()),yrs[d]))

    dfres = pd.concat([pd.DataFrame(list1), pd.DataFrame(list2), pd.DataFrame(list3), pd.DataFrame(list4), pd.DataFrame(mxdd)], 
                       keys=['return actual', 'return ann.', 'volatility ann.', 'dividend yield ann.', 'max drawdown'])
    dfres.columns=[df1.columns[1], df1.columns[2], df1.columns[3], 'years']    
    
    return dfres

def calc_stats_sharpe_years(df1, rfrate=True): #True or nothing means use EONIA from web
    import requests
    from requests.auth import HTTPBasicAuth
    yrstemp = ['1y','3y','5y']
    
    if rfrate==True:
        ratecurr = 'EUR'
        with open('creds.txt') as c: #creds.txt file contains: name.surname@stoxx.com,pass
            creds = c.read()
        creds=creds.split(',')
        auth = HTTPBasicAuth(creds[0],creds[1])
    
        url = "http://www.stoxx.com/download/customised/dowjones/eonia_rate.txt"
        r = requests.get(url, auth=auth)
    
        text = r.text
        rows = text.split('\n')[1:]
        data = [(x[:10], float(x[11:])/100) for x in rows if x!='']
    
        dfrate = pd.DataFrame(data, columns=['Date', 'rate'])
        dfrate['Date']=pd.to_datetime(dfrate['Date'], format='%d.%m.%Y', dayfirst=True)
        
        df1 = pd.merge(df1, dfrate, how='left', on='Date')
    elif rfrate==False:
        ratecurr = ''
        
    df1.fillna(method='pad', inplace=True)
    
    dto=df1.iloc[len(df1)-1,0]
    df1.reset_index(inplace=True,drop=True)
         
    dt5 = add_months(dto,-12*5)
    dt5 = df1[df1.Date>=dt5].iloc[0,0]
    dt3 = add_months(dto,-12*3)
    dt3 = df1[df1.Date>=dt3].iloc[0,0]
    dt1 = add_months(dto,-12)
    dt1 = df1[df1.Date>=dt1].iloc[0,0]
    dt0 = add_months(dto,-0)
    dt0 = df1[df1.Date>=dt0].iloc[0,0]
    
    if dt5.weekday()==5:
        dt5=dt5-dt.timedelta(days=1)
    elif dt5.weekday()==6:
        dt5=dt5-dt.timedelta(days=2)
        
    if dt3.weekday()==5:
        dt3=dt3-dt.timedelta(days=1)
    elif dt3.weekday()==6:
        dt3=dt3-dt.timedelta(days=2)
        
    if dt1.weekday()==5:
        dt1=dt1-dt.timedelta(days=1)
    elif dt1.weekday()==6:
        dt1=dt1-dt.timedelta(days=2)
        
    if dt0.weekday()==5:
        dt0=dt0-dt.timedelta(days=1)
    elif dt0.weekday()==6:
        dt0=dt0-dt.timedelta(days=2)
    
    dtlst=[dt5, dt3, dt1, dt0] #3 dates
    ct5y = len(df1[(df1.Date <= dtlst[3]) & (df1.Date > dtlst[0])]) #observation counts
    ct3y = len(df1[(df1.Date <= dtlst[3]) & (df1.Date > dtlst[1])])
    ct1y = len(df1[(df1.Date <= dtlst[3]) & (df1.Date > dtlst[2])])

    #dfvals=df1[df1.Date.isin(dtlst)==True]
    dfvals=pd.DataFrame()
    for fe in dtlst:
        dfvals=pd.concat([dfvals, pd.DataFrame(df1[df1.Date<=fe].sort('Date', ascending=False).reset_index(drop=True).iloc[0,:]).T], axis=0)
    dfvals.reset_index(inplace=True,drop=True)
    actret=np.array(dfvals.iloc[len(dfvals)-1,1:4])/np.array(dfvals.iloc[0:len(dfvals)-1,1:4])-1

    dfvals.reset_index(inplace=True,drop=True)
    actret=np.array(dfvals.iloc[len(dfvals)-1,1:4])/np.array(dfvals.iloc[0:len(dfvals)-1,1:4])-1

    yrs = []
    for x in range(len(actret)):
    	yrs.insert(0, yrstemp[x])    

    #actual returns
    dfactret=pd.DataFrame(actret)
    dfactret.columns=df1.columns[1:4]
    dfactret['years'] = yrs
    list1=np.array(dfactret)

    #annualized returns
    dfannret=dfactret
    dfannret.iloc[0,:3]=dfannret.iloc[0,:3].map(lambda x: (x+1)**(250/ct5y)-1)
    dfannret.iloc[1,:3]=dfannret.iloc[1,:3].map(lambda x: (x+1)**(250/ct3y)-1)
    dfannret.iloc[2,:3]=dfannret.iloc[2,:3].map(lambda x: (x+1)**(250/ct1y)-1)
    list2=np.array(dfannret)

    #vola
    returns=np.array(df1.iloc[1:len(df1),1:4])/np.array(df1.iloc[0:len(df1)-1,1:4])-1
    dfr=pd.DataFrame(returns)
    dfr['Date']=list(df1.loc[1:,'Date'])
    vol=[]
    vol.append(np.std(dfr[(dfr.Date <= dtlst[3]) & (dfr.Date > dtlst[0])].iloc[:,:3], ddof=1)*np.sqrt(250))
    vol.append(np.std(dfr[(dfr.Date <= dtlst[3]) & (dfr.Date > dtlst[1])].iloc[:,:3], ddof=1)*np.sqrt(250))
    vol.append(np.std(dfr[(dfr.Date <= dtlst[3]) & (dfr.Date > dtlst[2])].iloc[:,:3], ddof=1)*np.sqrt(250))
    dfvola=pd.DataFrame(vol)
    dfvola.columns=df1.columns[1:4]
    dfvola['years'] = yrs
    list3=np.array(dfvola)

    #dys
    dfdys=pd.DataFrame(list2)
    dfdys.iloc[:,1]=dfdys.iloc[:,1]-dfdys.iloc[:,0]
    dfdys.iloc[:,2]=dfdys.iloc[:,2]-dfdys.iloc[:,0]
    dfdys.iloc[:,0]=np.nan
    list4=np.array(dfdys)
    
    #sharpe ratio
    shrp=[]
    for d in range(len(dtlst[1:])):
        dft = df1[(df1.Date <= dtlst[3]) & (df1.Date > dtlst[d])]
        returns = np.array(dft.iloc[1:len(dft),1:4])/np.array(dft.iloc[0:len(dft)-1,1:4])-1
        eonia = np.array(dft.iloc[0:len(dft)-1,4])
        timedelta = np.array([(dft.iloc[i+1,0]-dft.iloc[i,0]).days for i in range(len(dft)-1)])
        drate = eonia*timedelta/365
        excessreturn = returns.T[0]-drate, returns.T[1]-drate, returns.T[2]-drate
        shrp.append([((np.mean(excessreturn[i])/np.std(excessreturn[i], ddof=1))*np.sqrt(250)) for i in range(3)])
    dft
    dfshrp = pd.DataFrame(shrp)
    dfshrp.columns=df1.columns[1:4]
    dfshrp['years']= yrs
    list5=np.array(dfshrp)
    
    #max drawdown
    mxdd = []
    for d in range(len(dtlst[1:])):
        dft = df1[(df1.Date <= dtlst[3]) & (df1.Date > dtlst[d])]
        pk = np.zeros((len(dft)+1,3))
        dd = np.zeros((len(dft)+1,4))
        h = np.array(dft.iloc[:,1:4])
        pk[0] = h[0]
        for i in range(len(h)):
            for j in range(3):
                pk[i+1,j] = h[i,j] if h[i,j] > pk[i,j] else pk[i,j]
                dd[i+1,j] = h[i,j] / pk[i+1,j] - 1 if h[i,j] < pk[i+1,j] else 0
        dd = dd[1:]
        mxdd.append((abs(dd[:,0].min()), abs(dd[:,1].min()), abs(dd[:,2].min()),yrs[d]))

    dfres = pd.concat([pd.DataFrame(list1), pd.DataFrame(list2), pd.DataFrame(list3), 
                       pd.DataFrame(list4), pd.DataFrame(list5), pd.DataFrame(mxdd)], 
                       keys=['return actual', 'return ann.', 'volatility ann.', 
                             'dividend yield ann.', 'Sharpe ratio '+ ratecurr +' ann.', 'max drawdown'])
    dfres.columns=[df1.columns[1], df1.columns[2], df1.columns[3], 'years']    
    
    return dfres


def calc_stats_sharpe_full(df1, rfrate=True): #True or nothing means use EONIA from web
    import requests
    from requests.auth import HTTPBasicAuth
    
    if rfrate==True:
        ratecurr = 'EUR'
        with open('creds.txt') as c: #creds.txt file contains: name.surname@stoxx.com,pass
            creds = c.read()
        creds=creds.split(',')
        auth = HTTPBasicAuth(creds[0],creds[1])
    
        url = "http://www.stoxx.com/download/customised/dowjones/eonia_rate.txt"
        r = requests.get(url, auth=auth)
    
        text = r.text
        rows = text.split('\n')[1:]
        data = [(x[:10], float(x[11:])/100) for x in rows if x!='']
    
        dfrate = pd.DataFrame(data, columns=['Date', 'rate'])
        dfrate['Date']=pd.to_datetime(dfrate['Date'], format='%d.%m.%Y', dayfirst=True)
        
        df1 = pd.merge(df1, dfrate, how='left', on='Date')
    elif rfrate==False:
        ratecurr = ''
        
    df1.fillna(method='pad', inplace=True)
    
    dto=df1.iloc[len(df1)-1,0]
    dfrom=df1.iloc[0,0]

    df1.reset_index(inplace=True,drop=True)
    
    dtlst=[dfrom, dto]
    ct = len(df1[(df1.Date <= dto) & (df1.Date > dfrom)]) #observation counts

    #dfvals=df1[df1.Date.isin(dtlst)==True]
    dfvals=pd.DataFrame()
    for fe in dtlst:
        dfvals=pd.concat([dfvals, pd.DataFrame(df1[df1.Date<=fe].sort('Date', ascending=False).reset_index(drop=True).iloc[0,:]).T], axis=0)
    dfvals.reset_index(inplace=True,drop=True)
    actret=np.array(dfvals.iloc[len(dfvals)-1,1:4])/np.array(dfvals.iloc[0:len(dfvals)-1,1:4])-1

    dfvals.reset_index(inplace=True,drop=True)
    actret=np.array(dfvals.iloc[len(dfvals)-1,1:4])/np.array(dfvals.iloc[0:len(dfvals)-1,1:4])-1  

    #actual returns
    dfactret=pd.DataFrame(actret)
    dfactret.columns=df1.columns[1:4]
    list1=np.array(dfactret)

    #annualized returns
    dfannret=dfactret
    dfannret.iloc[0,:3]=dfannret.iloc[0,:3].map(lambda x: (x+1)**(250/ct)-1)
    list2=np.array(dfannret)

    #vola
    vol=[]
    returns=np.array(df1.iloc[1:len(df1),1:4])/np.array(df1.iloc[0:len(df1)-1,1:4])-1
    dfr=pd.DataFrame(returns)
    dfr['Date']=list(df1.loc[1:,'Date'])
    vol.append((np.std(dfr[(dfr.Date <= dto) & (dfr.Date > dfrom)].iloc[:,:3], ddof=1))*np.sqrt(250))
    dfvola=pd.DataFrame(vol)
    dfvola.columns=df1.columns[1:4]
    list3=np.array(dfvola)

    #dys
    dfdys=pd.DataFrame(list2)
    dfdys.iloc[:,1]=dfdys.iloc[:,1]-dfdys.iloc[:,0]
    dfdys.iloc[:,2]=dfdys.iloc[:,2]-dfdys.iloc[:,0]
    dfdys.iloc[:,0]=np.nan
    list4=np.array(dfdys)
    
    #sharpe ratio
    shrp=[]
    dft = df1[(df1.Date <= dto) & (df1.Date > dfrom)]
    returns = np.array(dft.iloc[1:len(dft),1:4])/np.array(dft.iloc[0:len(dft)-1,1:4])-1
    eonia = np.array(dft.iloc[0:len(dft)-1,4])
    timedelta = np.array([(dft.iloc[i+1,0]-dft.iloc[i,0]).days for i in range(len(dft)-1)])
    drate = eonia*timedelta/365
    excessreturn = returns.T[0]-drate, returns.T[1]-drate, returns.T[2]-drate
    shrp.append([((np.mean(excessreturn[i])/np.std(excessreturn[i], ddof=1))*np.sqrt(250)) for i in range(3)])
    dfshrp = pd.DataFrame(shrp)
    dfshrp.columns=df1.columns[1:4]
    list5=np.array(dfshrp)
    
    #max drawdown
    mxdd = []
    dft = df1[(df1.Date <= dto) & (df1.Date > dfrom)]
    pk = np.zeros((len(dft)+1,3))
    dd = np.zeros((len(dft)+1,4))
    h = np.array(dft.iloc[:,1:4])
    pk[0] = h[0]
    for i in range(len(h)):
        for j in range(3):
            pk[i+1,j] = h[i,j] if h[i,j] > pk[i,j] else pk[i,j]
            dd[i+1,j] = h[i,j] / pk[i+1,j] - 1 if h[i,j] < pk[i+1,j] else 0
    dd = dd[1:]
    mxdd.append((abs(dd[:,0].min()), abs(dd[:,1].min()), abs(dd[:,2].min())))
  
    dfres = pd.concat([pd.DataFrame(list1), pd.DataFrame(list2), pd.DataFrame(list3), 
                       pd.DataFrame(list4), pd.DataFrame(list5), pd.DataFrame(mxdd)], 
                       keys=['return actual', 'return ann.', 'volatility ann.', 
                             'dividend yield ann.', 'Sharpe ratio '+ ratecurr +' ann.', 'max drawdown'])
    dfres.columns=[df1.columns[1], df1.columns[2], df1.columns[3]]    
    
    return dfres

#extended version - includes 1month, ytd and full period
def calc_stats_sharpe_ext(df1, rfrate=True): #True or nothing means use EONIA from web
    import requests
    from requests.auth import HTTPBasicAuth

    if rfrate==True:
        ratecurr = 'EUR'
        with open('creds.txt') as c: #creds.txt file contains: name.surname@stoxx.com,pass
            creds = c.read()
        creds=creds.split(',')
        auth = HTTPBasicAuth(creds[0],creds[1])

        url = "http://www.stoxx.com/download/customised/dowjones/eonia_rate.txt"
        r = requests.get(url, auth=auth)

        text = r.text
        rows = text.split('\n')[1:]
        data = [(x[:10], float(x[11:len(x)-1])/100) for x in rows if x!='']

        dfrate = pd.DataFrame(data, columns=['Date', 'rate'])
        dfrate['Date']=pd.to_datetime(dfrate['Date'], format='%d.%m.%Y', dayfirst=True)

        df1 = pd.merge(df1, dfrate, how='left', on='Date')
    elif rfrate==False:
        ratecurr = ''

    df1.fillna(method='pad', inplace=True)
    dto=df1.iloc[len(df1)-1,0]
    df1.reset_index(inplace=True,drop=True)
    
    dtfull = df1.iloc[0,0]
    strdtfull = str(dtfull)[:10]
    yrstemp = ['1m', 'YTD', '1y','3y','5y','10y', 'from ' + strdtfull]

    dtytd = dt.date(dto.year-1, 12, 31)
    dtytd = df1[df1.Date>=dtytd].iloc[0,0]

    dt1m = add_months(dto,-1)
    dt1m = df1[df1.Date>=dt1m].iloc[0,0]

    dt10 = add_months(dto,-12*10)
    dt10 = df1[df1.Date>=dt10].iloc[0,0]
    
    dt5 = add_months(dto,-12*5)
    dt5 = df1[df1.Date>=dt5].iloc[0,0]

    dt3 = add_months(dto,-12*3)
    dt3 = df1[df1.Date>=dt3].iloc[0,0]

    dt1 = add_months(dto,-12)
    dt1 = df1[df1.Date>=dt1].iloc[0,0]

    dt0 = add_months(dto,-0)
    dt0 = df1[df1.Date>=dt0].iloc[0,0]


    if dtytd.weekday()==5:
        dtytd=dtytd-dt.timedelta(days=1)
    elif dtytd.weekday()==6:
        dtytd=dtytd-dt.timedelta(days=2)

    if dt1m.weekday()==5:
        dt1m=dt1m-dt.timedelta(days=1)
    elif dt1m.weekday()==6:
        dt1m=dt1m-dt.timedelta(days=2)

    if dt10.weekday()==5:
        dt10=dt10-dt.timedelta(days=1)
    elif dt10.weekday()==6:
        dt10=dt10-dt.timedelta(days=2)
        
    if dt5.weekday()==5:
        dt5=dt5-dt.timedelta(days=1)
    elif dt5.weekday()==6:
        dt5=dt5-dt.timedelta(days=2)

    if dt3.weekday()==5:
        dt3=dt3-dt.timedelta(days=1)
    elif dt3.weekday()==6:
        dt3=dt3-dt.timedelta(days=2)

    if dt1.weekday()==5:
        dt1=dt1-dt.timedelta(days=1)
    elif dt1.weekday()==6:
        dt1=dt1-dt.timedelta(days=2)

    if dt0.weekday()==5:
        dt0=dt0-dt.timedelta(days=1)
    elif dt0.weekday()==6:
        dt0=dt0-dt.timedelta(days=2)

    dtlst=[dtfull, dt10, dt5, dt3, dt1, dtytd, dt1m, dt0] #7 dates
    ctfull = len(df1[(df1.Date <= dtlst[7]) & (df1.Date > dtlst[0])]) #observation counts
    ct10y = len(df1[(df1.Date <= dtlst[7]) & (df1.Date > dtlst[1])])
    ct5y = len(df1[(df1.Date <= dtlst[7]) & (df1.Date > dtlst[2])])
    ct3y = len(df1[(df1.Date <= dtlst[7]) & (df1.Date > dtlst[3])])
    ct1y = len(df1[(df1.Date <= dtlst[7]) & (df1.Date > dtlst[4])])
    ct1m = len(df1[(df1.Date <= dtlst[7]) & (df1.Date > dtlst[5])])
    ctytd = len(df1[(df1.Date <= dtlst[7]) & (df1.Date > dtlst[6])])

    #dfvals=df1[df1.Date.isin(dtlst)==True]
    dfvals=pd.DataFrame()
    for fe in dtlst:
        dfvals=pd.concat([dfvals, pd.DataFrame(df1[df1.Date<=fe].sort('Date', ascending=False).reset_index(drop=True).iloc[0,:]).T], axis=0)
    dfvals.reset_index(inplace=True,drop=True)
    actret=np.array(dfvals.iloc[len(dfvals)-1,1:4])/np.array(dfvals.iloc[0:len(dfvals)-1,1:4])-1

    yrs = []
    for x in range(len(actret)):
        yrs.insert(0, yrstemp[x])    

    #actual returns
    dfactret=pd.DataFrame(actret)
    dfactret.columns=df1.columns[1:4]
    dfactret['years'] = yrs
    list1=np.array(dfactret)

    #annualized returns
    dfannret=dfactret
    dfannret.iloc[0,:3]=dfannret.iloc[0,:3].map(lambda x: (x+1)**(250/ctfull)-1)
    dfannret.iloc[1,:3]=dfannret.iloc[1,:3].map(lambda x: (x+1)**(250/ct10y)-1)
    dfannret.iloc[2,:3]=dfannret.iloc[2,:3].map(lambda x: (x+1)**(250/ct5y)-1)
    dfannret.iloc[3,:3]=dfannret.iloc[3,:3].map(lambda x: (x+1)**(250/ct3y)-1)
    dfannret.iloc[4,:3]=dfannret.iloc[4,:3].map(lambda x: (x+1)**(250/ct1y)-1)
    dfannret.iloc[5,:3]=dfannret.iloc[5,:3].map(lambda x: (x+1)**(250/ct1m)-1)
    dfannret.iloc[6,:3]=dfannret.iloc[6,:3].map(lambda x: (x+1)**(250/ctytd)-1)
    list2=np.array(dfannret)
    dfannret.columns=[0,1,2,3]

    #vola
    returns=np.array(df1.iloc[1:len(df1),1:4])/np.array(df1.iloc[0:len(df1)-1,1:4])-1
    dfr=pd.DataFrame(returns)
    dfr['Date']=list(df1.loc[1:,'Date'])
    vol=[]
    vol.append(np.std(dfr[(dfr.Date <= dtlst[7]) & (dfr.Date > dtlst[0])].iloc[:,:3], ddof=1)*np.sqrt(250))
    vol.append(np.std(dfr[(dfr.Date <= dtlst[7]) & (dfr.Date > dtlst[1])].iloc[:,:3], ddof=1)*np.sqrt(250))
    vol.append(np.std(dfr[(dfr.Date <= dtlst[7]) & (dfr.Date > dtlst[2])].iloc[:,:3], ddof=1)*np.sqrt(250))
    vol.append(np.std(dfr[(dfr.Date <= dtlst[7]) & (dfr.Date > dtlst[3])].iloc[:,:3], ddof=1)*np.sqrt(250))
    vol.append(np.std(dfr[(dfr.Date <= dtlst[7]) & (dfr.Date > dtlst[4])].iloc[:,:3], ddof=1)*np.sqrt(250))
    vol.append(np.std(dfr[(dfr.Date <= dtlst[7]) & (dfr.Date > dtlst[5])].iloc[:,:3], ddof=1)*np.sqrt(250))
    vol.append(np.std(dfr[(dfr.Date <= dtlst[7]) & (dfr.Date > dtlst[6])].iloc[:,:3], ddof=1)*np.sqrt(250))
    dfvola=pd.DataFrame(vol)
    dfvola.columns=df1.columns[1:4]
    dfvola['years'] = yrs
    list3=np.array(dfvola)

    #dys
    dfdys=pd.DataFrame(list2)
    dfdys.iloc[:,1]=dfdys.iloc[:,1]-dfdys.iloc[:,0]
    dfdys.iloc[:,2]=dfdys.iloc[:,2]-dfdys.iloc[:,0]
    dfdys.iloc[:,0]=np.nan
    list4=np.array(dfdys)

    #sharpe ratio
    shrp=[]
    for d in range(len(dtlst[1:])):
        dft = df1[(df1.Date <= dtlst[7]) & (df1.Date > dtlst[d])]
        returns = np.array(dft.iloc[1:len(dft),1:4])/np.array(dft.iloc[0:len(dft)-1,1:4])-1
        eonia = np.array(dft.iloc[0:len(dft)-1,4])
        timedelta = np.array([(dft.iloc[i+1,0]-dft.iloc[i,0]).days for i in range(len(dft)-1)])
        drate = eonia*timedelta/365
        excessreturn = returns.T[0]-drate, returns.T[1]-drate, returns.T[2]-drate
        shrp.append([((np.mean(excessreturn[i])/np.std(excessreturn[i], ddof=1))*np.sqrt(250)) for i in range(3)])
    dft
    dfshrp = pd.DataFrame(shrp)
    dfshrp.columns=df1.columns[1:4]
    dfshrp['years']= yrs
    list5=np.array(dfshrp)

    #max drawdown
    mxdd = []
    for d in range(len(dtlst[1:])):
        dft = df1[(df1.Date <= dtlst[7]) & (df1.Date > dtlst[d])]
        pk = np.zeros((len(dft)+1,3))
        dd = np.zeros((len(dft)+1,4))
        h = np.array(dft.iloc[:,1:4])
        pk[0] = h[0]
        for i in range(len(h)):
            for j in range(3):
                pk[i+1,j] = h[i,j] if h[i,j] > pk[i,j] else pk[i,j]
                dd[i+1,j] = h[i,j] / pk[i+1,j] - 1 if h[i,j] < pk[i+1,j] else 0
        dd = dd[1:]
        mxdd.append((abs(dd[:,0].min()), abs(dd[:,1].min()), abs(dd[:,2].min()),yrs[d]))

    dfres = pd.concat([pd.DataFrame(list1), dfannret, pd.DataFrame(list3), 
                       pd.DataFrame(list4), pd.DataFrame(list5), pd.DataFrame(mxdd)], 
                       keys=['return actual', 'return ann.', 'volatility ann.', 
                             'dividend yield ann.', 'Sharpe ratio '+ ratecurr +' ann.', 'max drawdown'])
    dfres.columns=[df1.columns[1], df1.columns[2], df1.columns[3], 'period']  
    
    return dfres


def filterdts(dfrom,dto,df):
    dfreturn=df[df['Date']>=dfrom][df['Date']<=dto]
    return dfreturn

#receives a list of dates and outputs a list of 3rd fridays exept of the 1st month (double check behavior if dates in 1st month not complete)
def get_3rd_fridays(date_list):
    dates = pd.DatetimeIndex(date_list)
    isfriday=(dates.weekday==4)
    ismonthstart=(dates.day[0:len(dates)-1]>dates.day[1:len(dates)])
    dates1=dates[1:len(dates)]
    monthstarts=dates1[ismonthstart]
    fridays=dates[isfriday]
    fridaystokeep=fridays>=monthstarts[0]
    fridays=fridays[fridaystokeep]

    thirdfriday=[]
    for i in range(0,len(monthstarts)-1):
        isfridaysofthemonth=(fridays>=monthstarts[i]) & (fridays<monthstarts[i+1])
        fridaysofthemonth=fridays[isfridaysofthemonth]
        if len(fridaysofthemonth)>=2:
            thirdfriday.append(fridaysofthemonth[2])
    return thirdfriday

def rollCorr(a,b,window):
    rollCorr=np.zeros((len(a)))
    window=window-1
    for i in range(window,len(a)):
        rollCorr[i]=myCorr(a[i-window:i+1],b[i-window:i+1])
    return rollCorr
	
def rollTE(a,b,window):
    rollTE=np.zeros((len(a)))
    window=window-1
    for i in range(window,len(a)):
        rollTE[i] = np.std(a[i-window:i+1]-b[i-window:i+1],ddof=1)*np.sqrt(250)
    return rollTE

def rollMean(a,window):
    rollMean=np.zeros((len(a)))
    window=window-1
    for i in range(window,len(a)):
        rollMean[i]=np.mean(a[i-window:i+1])
    return rollMean

def rollSdev(a,window):
    rollSdev=np.zeros((len(a)))
    window=window-1
    for i in range(window,len(a)):
        rollSdev[i]=np.std(a[i-window:i+1], ddof=1)
    return rollSdev

def myCorr(a,b):
    l=len(a)
    cov=sum(a*b)/l-sum(a)/l*sum(b)/l  
    vara=sum(a*a)/l-sum(a)/l*sum(a)/l   
    varb=sum(b*b)/l-sum(b)/l*sum(b)/l
    return float(cov/(np.sqrt(vara)*np.sqrt(varb)))


def calccapfacs(df_comp):
    """
    received a df with -> column0:weight ;column1=cap; returns df with additional column2:capfactor; colum3:cappedwgt
    reindexes the df starting with 1
    """

    df_comp = df_comp.sort_values(df_comp.columns[0],ascending=False)
    df_comp.index = range(1,len(df_comp)+1)
    df_comp['capfactor']=1
    if sum(df_comp.iloc[:,1])<=1.:   
        df_comp['cappedwgt'] = 1. / len(df_comp) #equal weight
    else:
        df_comp['cappedwgt'] = df_comp.iloc[:,0]
        while len(df_comp[np.round(df_comp.cappedwgt, 7) > np.round(df_comp.iloc[:,1], 7)]) > 0:
            dblToCap = df_comp[df_comp.cappedwgt >= df_comp.iloc[:,1]].cap.sum()
            weightsnocap = df_comp[df_comp.cappedwgt < df_comp.iloc[:,1]].cappedwgt.sum()
            dblDistFactor = weightsnocap / (1 - dblToCap)
            for index, row in df_comp.iterrows():
                if row['cappedwgt'] >= row[1]: 
                    df_comp.loc[index,'cappedwgt'] = dblDistFactor * row[1]
            dblcappedsum = df_comp.cappedwgt.sum()
            df_comp['cappedwgt'] = df_comp['cappedwgt'] / dblcappedsum
    df_comp['capfactor']=(df_comp['cappedwgt']/df_comp.iloc[:,0])/max(df_comp['cappedwgt']/df_comp.iloc[:,0])
    return df_comp.reset_index(drop=True)


def cap_with_min_devs(dfx, met_=1):
    """
    calculates weights and capping factors minimizing the squared sum of the deviation with the intended weights
    received a df with column0:weight (used as initial guess) ;column1=cap; 
    returns df with additional column2:capfactor; colum3:cappedwgt
    method met_ =1: TNC (Truncated Newton's algorithm); met_ =2: L-BFGS-B (limited memory BFGS)
    reindexes the df starting with 1
    """

    from scipy.optimize import minimize
    dfx['capfactor']=1.
    options={1 : 'TNC',
         2 : 'L-BFGS-B'} #for methods 1 and 2
    def wgtfun(x):
        return sum(x-wt)**2
    x = np.array(dfx.iloc[:,0])
    wt = x
    b = [(0.,dfx.iloc[i,1]) for i in range(len(dfx))]
    c = ({'type':'eq', 'fun': lambda x: sum(x)-1. })  #methods TNC and L-BFGS-B cannot handle constraints
    
    res=minimize(wgtfun, x , method=options[met_], bounds=b, constraints=c)
    dfx['cappedwgt']=res.x
    dfx['capfactor']=(dfx['cappedwgt']/dfx.iloc[:,0])/max(dfx['cappedwgt']/dfx.iloc[:,0])
    return dfx

def add_months(date, months):
    import calendar
    month = int(date.month - 1 + months)
    year = int(date.year + month / 12)
    month = int(month % 12 + 1)
    day = min(date.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)

def monthend(dts):
    ismonthend=(dts.day[0:len(dts)-1]>dts.day[1:len(dts)])
    dts1=dts[:len(dts)]
    monthend=dts1[ismonthend]
    return pd.DataFrame(monthend)
	
def cur_versions(df, curlist): #needs 'Date' column in the first column
    cols = df.columns[1:]
    for cur in curlist:
        dtfrom = str(df.Date[0])[:10]
        dtto = str(df.Date[len(df)-1])[:10]
        dffx = q.get_curr_rate('EUR', cur, dtfrom, dtto)
        df = pd.merge(df, dffx.iloc[:,-1:], how='left', left_on='Date', right_index=True)

        if cur=='JPY':
            df.loc[df[df.Date=='2015-05-25'].index,'value_'] = 133.8804
        elif cur=='USD':
            df.loc[df[df.Date=='2015-05-25'].index,'value_'] = 1.10235
        elif cur=='AUD':
            df.loc[df[df.Date=='2015-05-25'].index,'value_'] = 1.40875
        elif cur=='CHF':
            df.loc[df[df.Date=='2015-05-25'].index,'value_'] = 1.03924987

        df = df.fillna(method='pad')

        for c in cols:
            df[c + '_' + cur] = df[c] * df['value_']
        del df['value_']

    df.tail()
    df.iloc[:,1:] = df.iloc[:,1:] / df.iloc[0,1:] *100
    #df.iloc[:,1:] = np.around(df.iloc[:,1:].astype(np.double), decimals=2)

    return df