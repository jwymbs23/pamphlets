import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_pickle('data_frame.pickle')
#print(df['date'].value_counts())
dates = df['date']
print(list(df))

dates = dates[pd.to_numeric(dates, errors='coerce').notnull()]
dates = dates.dropna().astype(float)
#print(dates.value_counts())
#dates.plot.hist(bins = 200)
#plt.show()
