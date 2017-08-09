import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import requests
import re

census = requests.get('http://people.sc.fsu.edu/~jburkardt/datasets/census/census_2010.txt').text
reps = requests.get ('http://people.sc.fsu.edu/~jburkardt/datasets/census/reps_2010.txt').text

resultc1 = re.findall('[A-Z]{2}', census)
resultr1 = re.findall('[A-Z]{2}', reps)

resultc2 = re.findall('[\d,]+', census)
resultr2 =  re.findall('[\d,]+', reps)

cendf = pd.DataFrame({"State":resultc1,"Population":resultc2})
repdf = pd.DataFrame({"State":resultr1, "Representatives":resultr2})
cendf= cendf.drop_duplicates()
repdf = repdf.drop_duplicates()

df1 = pd.merge(left = cendf, right = repdf, left_on = "State", right_on = "State")

df1["Populationstr"] = df1.Population.str.split(",")
for i in range(len(df1.Populationstr)):
    df1.Population[i]="".join(df1.Populationstr[i])
df1.Population  = pd.to_numeric(df1.Population)
df1.Representatives = pd.to_numeric(df1.Representatives)

beta, alpha = np.polyfit(df1.Population, df1.Representatives, 1)

sns.set()
_ = plt.plot(df1.Population, df1.Representatives, linestyle = "None", marker = ".")
_ = plt.xlabel("State Population")
_ = plt.ylabel("Representatives")
x_line = np.array([0,max(df1.Population)])
y_line = beta*x_line + alpha
_ = plt. plot(x_line, y_line, color = "red")

plt.show()








# print(census.head())
# print(reps)
# print(reps.head(), reps.tail())
#
#
# reps = pd.read_csv('reps_2010.txt', delimiter = " ", header = None)
#
#
# data = pd.merge(left = census, right = reps, axis =)
