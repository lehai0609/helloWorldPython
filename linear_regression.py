import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Read csv loansData.csv
loansData = pd.read_csv('loansData.csv')
print(loansData)
#Remove % from Interest Rate using lambda
rate = map(lambda x: float(x[:-1]), loansData['Interest.Rate'][0:])
loansData['Interest.Rate'] = rate
print(loansData['Interest.Rate'][0:5])

#Remove months from Loans Length
length = map(lambda x: int(x[:-7]), loansData['Loan.Length'][0:])
loansData['Loan.Length'] = length
print(loansData['Loan.Length'][0:5])

#Create a new column FICO Score with lower range from FICO Range
print(loansData['FICO.Range'][0:5])
score = map(lambda x: int(x.split('-')[0]), loansData['FICO.Range'][0:])
loansData['FICO.Score'] = score
print(loansData['FICO.Score'][0:5])

#Plot the histogram for FICO Score
plt.figure()
p = loansData['FICO.Score'].hist()
plt.show()

#Plot the scatter matrix
# plt.figure()
# matrix = pd.scatter_matrix(loansData,alpha=0.05,figsize=(5,5),diagonal='hist')
# plt.show()

#Find the coefficient of InterestRate = b + a1(FICOScore) + a2(LoanAmount)
fico = loansData['FICO.Score']
loanAmount = loansData['Amount.Requested']
interest = loansData['Interest.Rate']

y = np.matrix(interest).transpose()
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanAmount).transpose()

x = np.column_stack([x1, x2])

X = sm.add_constant(x)
model = sm.OLS(y, X)
f = model.fit()

print(f.summary())