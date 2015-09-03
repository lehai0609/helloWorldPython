import csv
import pandas as pd
from scipy import stats
data = '''Region, Alcohol, Tobacco
North, 6.47, 4.03
Yorkshire, 6.13, 3.76
Northeast, 6.19, 3.77
East Midlands, 4.89, 3.34
West Midlands, 5.63, 3.47
East Anglia, 4.52, 2.92
Southeast, 5.89, 3.20
Southwest, 4.79, 2.71
Wales, 5.27, 3.53
Scotland, 6.08, 4.51
Northern Ireland, 4.02, 4.56'''


data = data.splitlines()
data = [i.split(', ') for i in data]
rows = data[1::]
cols = data[0]
df = pd.DataFrame(rows,columns=cols)

df['Alcohol'] = df['Alcohol'].astype(float)

print("Mean-Median-Mode-Range-Variance-Standard Deviation of Alcohol is: $s, %s, %s, %s" %(df['Alcohol'].mean, df))
