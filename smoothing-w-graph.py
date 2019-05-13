from statsmodels.tsa.api import SimpleExpSmoothing
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


raw_data = pd.read_excel("arda-2000.xlsx", sheet_name=0, parse_dates=['DATE'], index_col='ID')

data_open = raw_data['OPEN']
time_log = np.log(data_open)

ses = SimpleExpSmoothing(time_log).fit(smoothing_level=0.2)
ses1 = ses.forecast(len(time_log))


plt.plot(time_log)
plt.plot(ses1)
plt.legend(["True", "Prediction"])
plt.show()

mape = mean_absolute_percentage_error(time_log, ses1)
#print(str(round(100/mape, 2))+"%")
#print(1-mape)
#print((1-mape)*100)
print(str(round((1-mape)*100, 2))+"%")
