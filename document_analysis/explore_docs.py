import pickle
import pandas as pd
import matplotlib.pyplot as plt


doc_data = pickle.load(open('df_relevant.pkl','rb'))

print(list(doc_data))

doc_data['date'].hist(bins=list(range(int(doc_data['date'].min()), int(doc_data['date'].max()),2)))
plt.show()

doc_data['imagecount'].hist(bins=list(range(int(doc_data['imagecount'].min()), 100,4)))
plt.show()
