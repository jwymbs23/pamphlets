import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

sent_df = pickle.load(open('df_sentiment.pkl', 'rb'))

date_range = (1788, 1800)

all_cols = list(sent_df)
print(all_cols)
people_cols = [col for col in all_cols if col.find(u'sentiment_vals_w') != -1 and col.find(u'voltaire') == -1 and col.find(u'rousseau') == -1]
sent_dict = {}
for person in people_cols:
    doc_count = sent_df[person].count()
    print(doc_count)
    if doc_count > 100:
        mean_by_date = sent_df[person].groupby(sent_df[u'date']).mean()
        dates = mean_by_date.keys().tolist()
        scores = mean_by_date.tolist()
        sentiment_series = [(date,score) for date, score in zip(dates,scores) if date > date_range[0] and date < date_range[1]]# if np.isnan(score) == False]
        sentiment_change = []
        for first,second in zip(sentiment_series[:-1], sentiment_series[1:]):
            if np.isnan(first[1]):
                last_valid = 0
            else:
                last_valid = first[1]
            if np.isnan(second[1]):
                next_valid = last_valid
            else:
                next_valid = second[1]
            sentiment_change.append(( (second[0] + first[0])*0.5, (next_valid - last_valid) ))
        #print(sentiment_change)
        sentiment_change = np.asarray(sentiment_change)
        #plt.plot(sentiment_change[:,0], sentiment_change[:,1])
        sent_dict[person[17:]] = sentiment_change
#    sent_df[person].groupby(sent_df[u'date']).mean().plot()
#plt.show()

def simscore(n1, n2):
    return np.sqrt(sum( (n1[:,1]-n2[:,1])**2))
    
names = list(sent_dict)
euclid_dist_arr = np.zeros( (len(names),len(names)) )
for name1 in range(len(names)):
    for name2 in range(len(names)):
        euclid_dist_arr[name1][name2] = simscore(sent_dict[names[name1]], sent_dict[names[name2]])
        
        
def cosine_sim(v1, v2):
    dot = sum([a * b for a,b in zip(v1, v2)])
    vv11 = sum([a*a for a in v1])
    vv22 = sum([a*a for a in v2])
    return dot/np.sqrt(vv11*vv22)

cos_dist = np.zeros( (len(names), len(names)) )
#cosine similarity between people:
for row1id, row1 in enumerate(euclid_dist_arr):
    for row2id,row2 in enumerate(euclid_dist_arr):
        #print(row1, row2)
        cos_dist[row1id][row2id] = cosine_sim(row1, row2)
        

g = sns.clustermap(cos_dist)
reordered_ind = g.dendrogram_row.reordered_ind
new_name_indices = [names[i] for i in g.dendrogram_row.reordered_ind]
#ax1.set_xticklabels(new_region_indices,fontsize=6,rotation=90)
#ax1.set_yticklabels(new_region_indices,fontsize=6)
plt.show()


reorder_corr = np.zeros((len(names), len(names)))
for i in range(euclid_dist_arr.shape[0]):
    for j in range(euclid_dist_arr.shape[0]):
        reorder_corr[i][j] = cos_dist[reordered_ind[i]][reordered_ind[j]]#euclid_dist_arr[reordered_ind[i]][reordered_ind[j]]
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.imshow(reorder_corr,interpolation='none')
ax1.set_xticks(range(len(names)))
ax1.set_yticks(range(len(names)))
ax1.set_xticklabels(new_name_indices,fontsize=12,rotation=90)
ax1.set_yticklabels(new_name_indices,fontsize=12)
plt.show()



from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
#from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.decomposition import PCA


n_clusters = 4
#clusterer = AgglomerativeClustering(n_clusters=n_clusters)
#clusterer = KMeans(n_clusters = n_clusters)
clusterer = AffinityPropagation(damping = 0.9)
cluster_labels = clusterer.fit_predict(cos_dist)#euclid_dist_arr)
sort_labels = {names[i]: cluster_labels[i] for i in range(len(names))}#sorted(list(zip(names, cluster_labels)) , key = lambda k: k[1])
#for i in sort_labels:
#    print(i[0], i[1])
#print(cluster_labels)

print(sort_labels)

colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'yellow']
av_sentiment_change = {}
count_list = [1 for i in range(10)]
for i in sent_dict:
    if sort_labels[i] in av_sentiment_change:
        av_sentiment_change[sort_labels[i]] += sent_dict[i]
        count_list[sort_labels[i]] += 1
    else:
        av_sentiment_change[sort_labels[i]] = sent_dict[i]
    plt.plot(sent_dict[i][:,0], sent_dict[i][:,1], c = colors[sort_labels[i]], alpha = 0.6, lw = 1)
for i in av_sentiment_change:
    plt.plot(av_sentiment_change[i][:,0]/count_list[i], av_sentiment_change[i][:,1]/count_list[i], c = colors[i], lw = 4)
plt.show()

#plt.imshow(euclid_dist_arr, interpolation='none')




#plt.show()

