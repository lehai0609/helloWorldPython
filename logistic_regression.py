__author__ = 'lehai0609'
import pandas as pd
import statsmodels.api as sm
import numpy as np
import math

loansData = pd.read_csv('loansData.csv')
print(loansData)

#Create a new column FICO Score with lower range from FICO Range
score = map(lambda x: int(x.split('-')[0]), loansData['FICO.Range'][0:])
loansData['FICO.Score'] = score

#Indicating whether the interest rate is < 12%
ir_tf = []
for row in loansData['Interest.Rate']:
    if float(row[:4]) > 12:
        ir_tf.append(1)
    else:
        ir_tf.append(0)
loansData['IR_TF'] = ir_tf
print(loansData['IR_TF'])

#Add a column with an constant intercept of 1
loansData['Intercept']= 1.0
print(loansData['Intercept'])

#A list of column names of independent vars
ind_vars = loansData.columns.values
print(ind_vars)

#Logistic Regression function
independent_vars = ['Intercept', 'FICO.Score', 'Amount.Funded.By.Investors']
def logistic_function(ficoScore, loanAmount):
    logit = sm.Logit(loansData['IR_TF'], loansData[independent_vars])
    results = logit.fit()
    coeff = results.params
    p = 1/(1 + math.e**(coeff[0]+coeff[1]*ficoScore+coeff[2]*loanAmount))
    return p

print(logistic_function(720, 10000))

