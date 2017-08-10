import requests
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib

#Import Excel file
xl = pd.read_excel(io = 'http://www.bankofengland.co.uk/publications/Documents/quarterlybulletin/threecenturiesofdata.xls', sheetname = ["Real GDP","GDP(E) contributions 1830-2009"], skiprows = 1)

#Extra Data wanted as DataFrame
df1 = pd.merge(left = xl["Real GDP"][["Annual growth"]], right = xl["GDP(E) contributions 1830-2009"][["Aggregate investment"]], left_on = xl["Real GDP"]["Sources"], right_on = xl["GDP(E) contributions 1830-2009"]["Years"])

#Create more easily accessible columns + clen data
df1.columns = ["Annual_growth", "Aggregate_investment"]
df1 = df1.dropna()
df1 = df1.reset_index()

#Plot Style
sns.set()


#Create Sample Scatter Plot
_ = plt.plot(df1.Aggregate_investment, df1.Annual_growth, marker = ".", linestyle= "none", color = "blue", alpha = 0.6)
_ = plt.xlabel("Aggregate_investment (percent growth)")
_ = plt.ylabel("Annual_growth (percent)")

#Find and Plot Least Squares line
beta_hat, alpha_hat = np.polyfit(df1.Aggregate_investment, df1.Annual_growth, 1)
print(beta_hat, alpha_hat)

x_line = np.array([min(df1.Aggregate_investment), max(df1.Aggregate_investment)])
y_line_hat = beta_hat*x_line + alpha_hat
_ = plt.plot(x_line, y_line_hat, color = "red", linestyle = "-")
_ = plt.margins(0.02)


#Create function to Bootstap replicate pairs of least squares parameters
def draw_bs_pairs_replicates(x, y, func, size=1, **kwargs):
    """Create an array of Bootstrap Replicates for a given function"""
    bs1_replicates = np.empty(size)
    bs2_replicates = np.empty(size)
    inds = np.arange(len(x))
    for i in range(size):
        bs_inds = np.random.choice(inds, size = len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs1_replicates[i], bs2_replicates[i] = func(bs_x, bs_y, **kwargs)
    return bs1_replicates , bs2_replicates

#Create Bootstrap replicates
beta_bs, alpha_bs =  draw_bs_pairs_replicates(x=df1.Aggregate_investment, y=df1.Annual_growth, func = np.polyfit, size = 100, deg = 1)

#Add Bootstrap plots
for i in range(len(alpha_bs)):
    _ = plt.plot(x_line, beta_bs[i]*x_line + alpha_bs[i], linestyle = "-", alpha = 0.1, linewidth = 0.5, color = "navy" )


plt.show()
plt.clf()
#Permuatation Hypothesis test of slope:

investment = df1.Aggregate_investment
growth = df1.Annual_growth

#1. Generate data assuming no relation #2. Calculate Slope parameters
def draw_perm_pairs_reps(x, y, func, size=1, **kwargs):
    """Creates array(s) of paired data permutation replicates for a given function and outputs 2 parameter arrays"""
    inds2 = np.arange(len(x))
    replicates1 = np.empty(size)
    replicates2= np.empty(size)
    for i in range(size):
        perm_inds = np.random.permutation(inds2)
        x_perm = x[perm_inds]
        replicates1[i], replicates2[i] = func(x_perm, y, **kwargs)
    return replicates1, replicates2

beta_perm, alpha_perm = draw_perm_pairs_reps(investment, growth, func = np.polyfit, size = 10000, deg = 1)

#3. Form confidence interval and check p value
print("Confidence interval for random data",np.percentile(beta_perm, [2.5,97.5]))
print("P-Value:", np.sum(beta_perm >= beta_hat)/len(beta_perm))

#4: Print histogram
_1 = plt.hist(beta_perm, bins = 100, normed = True)
_1 = plt.xlabel("Beta assuming no relation between data")
_1 = plt.ylabel("Probability")

plt.show()










# def draw_perm_pairs_reps(x, y, func, size=1, **kwargs):
#     """Creates array(s) of paired data permutation replicates for a given function and places in 2D array"""
#     inds2 = np.arange(len(x))
#     replicates = np.empty(shape = size, dtype = tuple)
#     for i in range(size):
#         perm_inds = np.random.permutation(inds2)
#         x_perm = x[perm_inds]
#         replicates[i] = func(x_perm, y, **kwargs)
#     return replicates
#
# beta_perm = draw_perm_pairs_reps(investment, growth, func = np.polyfit, size = 100, deg = 1)
