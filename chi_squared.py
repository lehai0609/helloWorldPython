__author__ = 'lehai0609'
import matplotlib.pyplot as plt
import pandas as pd
import collections
from scipy import stats
# Load the reduced version of the lending data
loansData = pd.read_csv('loansData.csv')
# Clear the data
loansData.dropna(inplace=True)
print(loansData)
# Plot a new graph for Amount Requested frequencies
freg = collections.Counter(loansData['Amount.Requested'])
plt.figure()
plt.bar(freg.keys(), freg.values(), width=1)
plt.show()
# Chi-square test
chi, p = stats.chisquare(freg.values())
print(chi)
print(p)