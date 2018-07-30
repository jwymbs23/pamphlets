# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


doc_df = pickle.load(open('df_sentiment.pkl', 'rb'))



all_cols = list(doc_df)
renaming_dict = {x : x.encode('utf-8') for x in all_cols}
print(renaming_dict)
doc_df.rename(renaming_dict, axis = 'columns')



#print(all_cols)



people_cols = [col for col in all_cols if col.find(u'sentiment_vals_w') != -1]

#print(list(doc_df))
print(people_cols)
#normalize data
for term in people_cols:
    doc_df[term]=(doc_df[term]-doc_df[term].mean())#.min())/(doc_df[term].max()-doc_df[term].min())
#doc_df[people_cols] = doc_df[people_cols].fillna(doc_df[people_cols].mean())

#term_vector = ['marat', 'robespierre']
#weighted = True
#if weighted == True:
#    sa_cols = ['sentiment_vals_w_'+TERM for TERM in term_vector]
#else:
#    sa_cols = ['sentiment_vals_unw_'+TERM for TERM in term_vector]
from sklearn import linear_model


for term in people_cols:
    if term != 'sentiment_vals_w_marat':
        sa_cols = ['sentiment_vals_w_marat', term]
        print(sa_cols)
        overlap_df = doc_df[sa_cols].dropna(thresh=2)#.loc[doc_df[sa_cols].notnull()]
        if(overlap_df.size > 0):
            l1 = np.asarray(overlap_df[sa_cols[0]].tolist())
            l2 = np.asarray(overlap_df[sa_cols[1]].tolist())
            line_1 = np.arange(min(l1),max(l1),0.001)
            #plt.hist2d(l1,l2,bins=20)
            #plt.show()
            overlap_df.plot(x=sa_cols[0], y=sa_cols[1],style='o', xlim=(-0.25,0.25), ylim=(-0.25,0.25))
            regr = linear_model.LinearRegression()

            # Train the model using the training sets
            regr.fit(l2.reshape(-1, 1) ,l1)
            pred_line_2 = regr.predict(line_1.reshape(-1,1))
            plt.plot(line_1, pred_line_2)
            plt.show()
#print(overlap_df.head(10))

#from pandas import TimeSeries
import statsmodels.api as sm


X = doc_df[[x for x in people_cols if x.find('marat') == -1]]
print(list(X))
print(X.head(4))
y = doc_df['sentiment_vals_w_marat']

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print(model.summary())
