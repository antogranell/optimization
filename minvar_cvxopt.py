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
from pandas.tseries.offsets import BDay
from cvxopt import matrix, solvers


#min var

dfd = pd.read_csv(dkf.dloc + '03_Deka_Europe_Multi_Factor_adtv_v2.csv', sep=';')
dfd['Date'] = pd.to_datetime(dfd['Date'], format='%d.%m.%Y', dayfirst=True)

dfd = dfd[-1:].copy()

for d in dfd['Date'].drop_duplicates():

    dx = str(d)[:10]
    print(dx)

    dfcomp = pd.read_csv(dkf.dloc + 'minvar/components/components_' + dx + '.csv', sep=';')

    dfpr = pd.read_csv(dkf.dloc + 'minvar/prices/prices_tr_'+ dx + '.csv', sep=';')
    dfpr.fillna(method='pad', inplace=True)

    returns = np.array(dfpr.iloc[1:len(dfpr), 1:]) / np.array(dfpr.iloc[0:len(dfpr) - 1, 1:])
    logreturns = [np.log((returns.T)[i]) for i in range(len(returns.T))]
    covs = np.cov(logreturns)*252

    ### Optimization

    # minimize    (1/2)*x'*P*x + q'*x
    # subject to  G*x <= h
    #             A*x = b

    n = len(covs)
    ni = 10 #num industries

    P = matrix(covs+np.eye(n))
    A = matrix(np.ones((1,n)))
    b = matrix(np.ones((1,1)))
    g = np.zeros((ni,n))
    q = matrix(np.zeros((n,1)))

    industryidx=list()
    for i in np.unique(list(dfcomp.ICB_ind_name)):
        industryidx.append(dfcomp[dfcomp['ICB_ind_name']==i]['isin'].index)

    for i in range(ni):
        g[i, industryidx[i]] = 1

    G = matrix(np.concatenate((g, np.eye(n)), 0))
    h = matrix(np.concatenate((0.2 * np.ones((ni, 1)), 0.045 * np.ones((n, 1))), 0))

    sol = solvers.qp(P, q, G, h, A, b)


#pd.DataFrame(returns).to_csv(dkf.dloc + 'minvar/returns.csv',sep=';')
print('done')