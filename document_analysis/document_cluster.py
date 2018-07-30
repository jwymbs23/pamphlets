import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

sent_df = pickle.load(open('df_sentiment_people.pkl', 'rb'))

all_cols = list(sent_df)
print(all_cols)
#people_cols = [col for col in all_cols if col.find(u'sentiment_vals_w') != -1 and col.find(u'voltaire') == -1 and col.find(u'rousseau') == -1 and col.find(u'audu') == -1 and col.find(u'barry') == -1 and col.find(u'corday') == -1]
date_range = (1786, 1800)
df_date_cut = sent_df[(sent_df['date'] > date_range[0]) & (sent_df['date'] < date_range[1])]
sent_df = []
sent_dict = {}
drop_cols = df_date_cut.dropna(axis = 1, thresh=1000)

remaining_cols = list(drop_cols)

people_cols = [col for col in remaining_cols if col.find(u'sentiment_vals_w') != -1 and col.find(u'voltaire') == -1 and col.find(u'rousseau') == -1 and col.find(u'audu') == -1 and col.find(u'barry') == -1 and col.find(u'corday') == -1]
#print(people_cols)
df_sent_date_cols = drop_cols[['identifier','date'] + people_cols].dropna(thresh = 5)

#print(df_sent_date_cols.head(2))
mean_impute_df = df_sent_date_cols.fillna(df_sent_date_cols.mean())
#mean_impute_df = mean_impute_df - mean_impute_df.mean

#print(mean_impute_df[people_cols[1], people_cols[3]].head(20))
#print(mean_impute_df.shape)
#print(mean_impute_df.head(10))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
mean_impute_df[people_cols] = scaler.fit_transform(mean_impute_df[people_cols])
df_scaled = mean_impute_df

from sklearn.metrics.pairwise import cosine_similarity
cos_dist = cosine_similarity(df_scaled[people_cols])

#plt.imshow(cos_dist,interpolation='none')
#plt.savefig('cos_dist.png', dpi=600)


#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#X_r = pca.fit(cos_dist).transform(cos_dist)
#print(X_r)


#from sklearn.manifold import TSNE
#tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000, learning_rate = 300)
#X_r = tsne.fit_transform(cos_dist)

#from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
#from sklearn.cluster import AffinityPropagation
#from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.decomposition import PCA


n_clusters = 4
#clusterer = AgglomerativeClustering(n_clusters=n_clusters)
clusterer = KMeans(n_clusters = n_clusters)
#clusterer = AffinityPropagation(damping = 0.99)
cluster_obj = clusterer.fit_predict(cos_dist)#euclid_dist_arr)
labels = clusterer.labels_
df_scaled['labels'] = labels
#print(df_scaled.groupby('labels').mean()[people_cols])
#print(df_scaled.groupby('labels').mean()[people_cols].unstack())

fig, ax = plt.subplots(figsize=(15,7))
df_scaled.groupby('labels').mean()[people_cols].plot(ax=ax)
plt.show()

#n_clusters = len(clusterer.cluster_centers_indices_)
#print(labels)
#print(n_clusters)
#sort_labels = sorted(list(zip(names, cluster_labels)) , key = lambda k: k[1])
#for i in sort_labels:
#    print(i[0], i[1])
#import random
#colors = []
#for i in range(n_clusters):
#    colors.append('%06X' % random.randint(0, 0xFFFFFF))
colors = ['navy', 'turquoise', 'darkorange', 'red', 'yellow', 'blue', 'green', 'orange', 'black', 'grey']
lw = 2

for color, i in zip(colors, range(n_clusters)):
    plt.scatter(X_r[labels == i, 0], X_r[labels == i, 1], color=color, alpha=.8, lw=lw)



#plt.scatter(X_r[:,0], X_r[:,1])
plt.show()
exit(0)






import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score


range_n_clusters = [10, 11, 12, 14, 16, 18, 20, 25]

X = X_r

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    #ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

