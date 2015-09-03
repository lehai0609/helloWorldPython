__author__ = 'lehai0609'
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

loansData = pd.read_csv('loansData.csv')
print(loansData)

loansData.boxplot(column="Amount.Funded.By.Investors")
plt.show()

loansData.hist(column="Amount.Funded.By.Investors")
plt.show()

plt.figure()
graph = stats.probplot(loansData['Amount.Funded.By.Investors'],dist='norm',plot=plt)
plt.show()