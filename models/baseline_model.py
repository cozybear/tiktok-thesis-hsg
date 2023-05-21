import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
from sklearn import linear_model
#helper functions

def adjusted_r2(y_test, y_pred, n_predictors):
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1-(1-r2)*(len(y_test)-1)/(len(y_test)-n_predictors-1)
    return adj_r2

def mean_absolute_percentage_error(y_pred, y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




df_final = pd.read_csv('../final/df_final_channel.csv').groupby('user').sample(1)

a = df_final.sample(frac=.9, random_state=60)
b = df_final['average_engagement_impressions'].loc[df_final.index.difference(a.index)]
df = pd.DataFrame({'true': b, 'pred': a['average_engagement_impressions'].mean()})
test = df['true']
mean = df['pred']

print("Adj. R2: ", adjusted_r2(test, mean, 1))
print("MAE: ", mean_absolute_error(test, mean))
print("MAPE: ",mean_absolute_percentage_error(mean, test))
print("MSE: ",mean_squared_error(test, mean))
print("RMSE: ",mean_squared_error(test, mean, squared=False))
print("Pearson :", np.corrcoef(mean, test)[0][1])

res, p_value = stats.spearmanr(mean, test)
print("Spearman :", res, p_value)

plt.figure(figsize=(8, 8))
x = y = np.arange(0, 26, 1)
coef = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(coef)
plt.plot(poly1d_fn(x), '--k')
plt.scatter(test, mean, alpha=0.3)
plt.xlabel('Ground Truth [%]')
plt.ylabel('Predictions [%]')
plt.savefig("final/results_baseline.png", dpi=400, bbox_inches="tight")
