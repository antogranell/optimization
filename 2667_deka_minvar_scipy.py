import pandas as pd
import numpy as np
import math
import datetime as dt
import sys
import pyodbc
import time

import qadconnect34plus as q
import myfunctions as myf
import dekafunctions as dkf

from scipy.optimize import minimize

#min var

def port_var(w):
    return np.dot(w,np.dot(covs,w))

ind_max=0.2
co_max=0.2
w_max=0.08
w_ucits = 0.38

dfd = pd.read_csv(dkf.dloc + '03_Deka_Europe_Multi_Factor_adtv_v2.csv', sep=';')
dfd['Date'] = pd.to_datetime(dfd['Date'], format='%d.%m.%Y', dayfirst=True)

dfd = dfd[-1:].copy()

for d in dfd['Date'].drop_duplicates():

    dx = str(d)[:10]
    print(dx)

    dfcomp = pd.read_csv(dkf.dloc + 'minvar/components/components_' + dx + '.csv', sep=';')

    dfpr = pd.read_csv(dkf.dloc + 'minvar/prices/prices_tr_' + dx + '.csv', sep=';')
    dfpr.fillna(method='pad', inplace=True)
    dfpr.fillna(method='bfill', inplace=True)

    returns = np.array(dfpr.iloc[1:len(dfpr), 1:]) / np.array(dfpr.iloc[0:len(dfpr) - 1, 1:])
    logreturns = [np.log((returns.T)[i]) for i in range(len(returns.T))]
    covs = np.cov(logreturns) * 252

    n = len(covs)
    b_ = [(0., w_max) for i in range(n)]

    cons = np.array([])
    cons = np.append(cons, {'type': 'eq', 'fun': lambda w: sum(w) - 1.})

    # industry constraints
    # https://stackoverflow.com/questions/45491376/scipy-optimization-not-running-when-i-try-to-set-constraints-using-a-for-loop/45493887#45493887
    industryidx = list()
    for i in np.unique(list(dfcomp.ICB_ind)):
        industryidx.append(dfcomp[dfcomp.ICB_ind == i]['isin'].index)

    for ind in range(len(industryidx)):
        con = {'type': 'ineq', 'fun': lambda w, ind=ind: ind_max - sum(w[industryidx[ind]])}
        cons = np.append(cons, con)

    # country constraint
    countryidx = list()
    for co in np.unique(list(dfcomp.country)):
        countryidx.append(dfcomp[dfcomp.country == co]['isin'].index)

    for cntry in range(len(countryidx)):
        con = {'type': 'ineq', 'fun': lambda w, cntry=cntry: co_max - sum(w[countryidx[cntry]])}
        cons = np.append(cons, con)

    #ucits constraint
    cons = np.append(cons, {'type': 'ineq', 'fun': lambda w: w_ucits-sum(w[w>0.045])})
    w0 = np.ones(n) / n

    optiwgts = minimize(port_var, w0, method='SLSQP', bounds=b_, constraints=cons)

    dfcomp['weight'] = optiwgts.x
    dfcomp.loc[dfcomp[dfcomp.weight < 1e-06].index, 'weight'] = 0


    pfvar_opti = np.dot(optiwgts.x, np.dot(covs,optiwgts.x))
    pfvar_ew = np.dot(w0, np.dot(covs,w0))
    print 'pf var - opti weight:', pfvar_opti, np.sqrt(pfvar_opti)
    print 'pf var - equal weight:', pfvar_ew, np.sqrt(pfvar_ew)
    print dfcomp.groupby('ICB_ind_name').sum()['weight'].sort_values(ascending=False)
    print dfcomp.groupby('country').sum()['weight'].sort_values(ascending=False).head()
    print dfcomp.loc[dfcomp[dfcomp.weight!=0].index, 'weight'].sort_values(ascending=False).head(10)
    print optiwgts.x[optiwgts.x>0.045].sum(), 'sum ucits'

import matplotlib.pyplot as plt
# the histogram of the data
fin_decent_wgts = list(dfcomp.loc[dfcomp[dfcomp.weight!=0].index, 'weight'])
n, bins, patches = plt.hist(fin_decent_wgts, 50, normed=1, facecolor='g', alpha=0.75)
plt.show()


print('done')
