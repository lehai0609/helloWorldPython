__author__ = 'lehai0609'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Load the Lending Club Statistics
loansData = pd.read_csv('loansData.csv')
print(loansData)
loansData.dropna(inplace=True)

#Convert Interest Rate to Float
rate = map(lambda x: float(x.rstrip('%')), loansData['Interest.Rate'])
loansData['IR_RT'] = rate

#Create Intercept constant
loansData['Intercept'] = float(1.0)
#Annual Income
loansData['Annual.Income'] = 12 * loansData['Monthly.Income']

#Use income (annual_inc) to model interest rates (int_rates)
model1 = sm.OLS(loansData['IR_RT'], loansData[['Intercept', 'Annual.Income']])
result1 = model1.fit()
print(result1.summary())


#Digitize home ownership status
ownershipStatus = []
for row in loansData['Home.Ownership']:
    if row == "MORTGAGE":
        ownershipStatus.append(1)
    elif row == "RENT":
        ownershipStatus.append(2)
    elif row == "OWN":
        ownershipStatus.append(3)
    else:
        ownershipStatus.append(4)
loansData['Ownership'] = ownershipStatus
#Add home ownership (home_ownership) to the model
model2 = sm.OLS(loansData['IR_RT'], loansData[['Intercept', 'Annual.Income', 'Ownership']])
result2 = model2.fit()
print(result2.summary())