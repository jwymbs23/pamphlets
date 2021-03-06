import pandas as pd
import matplotlib
matplotlib.use('Agg')

from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob

# sent_df = pickle.load(open('../pickles/3_df_sentiment.pkl', 'rb'))

# all_cols = list(sent_df)

# print(all_cols)

# people_cols = [col for col in all_cols if col.find(s_cols) != -1]# and col.find(u'voltaire') == -1 and col.find(u'rousseau') == -1]

# sent_dict = {}


def get_sent_dict(sent_df, terms, weight_flag = True, date_range = (1786,1800)):
    """
    Input:  sentiment dataframe, terms to plot
    Output: 
    
    """
    sent_dict = {}
    #print(list(sent_df))
    if weight_flag:
        s_cols = 'sentiment_vals_w_'
    else:
        s_cols = 'sentiment_vals_unw_'

    doc_count_cutoff = 0
    
    # docs_per_year = sent_df.groupby('date').count()
    # print(docs_per_year)
    terms = sorted(terms)
    for raw_term in terms:
        term = s_cols + raw_term
        try:
            #print(raw_term)
            #print(glob.glob('./pickles/*'))
            #print('./pickles/'+raw_term+'_df.pkl')
            term_df = pickle.load(open('./pickles/'+raw_term+'_df.pkl', 'rb'))
        except:
            print('error')
        term_df = pd.merge(term_df, sent_df[['identifier', 'date']], on='identifier')        
        doc_count = term_df[term].count()
        #print(doc_count)
        sentiment_series = []
        if doc_count > doc_count_cutoff:
            mean_by_date = term_df.groupby('date')[term].mean()
            date_count = term_df.groupby('date')[term].count().tolist()
            dates = mean_by_date.keys().tolist()
            scores = mean_by_date.tolist()
            # old way with single large dataframe                                                
            # mean_by_date = sent_df[term].groupby(sent_df[u'date']).mean()
            # date_count = sent_df[term].groupby(sent_df['date']).count().tolist()
            # dates = mean_by_date.keys().tolist()
            # scores = mean_by_date.tolist()
            # sentiment_series = [(date,score,dcount) for date, score,dcount in zip(dates,scores,date_count)
            #                      if date > date_range[0] and date <= date_range[1]]
            for cd, date in enumerate(range(date_range[0], date_range[1])):
                if date in dates:
                    date_idx = dates.index(date)
                    sentiment_series.append((dates[date_idx], scores[date_idx], date_count[date_idx]))
                elif date >= date_range[0] and date < date_range[1]:
                    sentiment_series.append((date, 0,0))
                                                
            sent_dict[term[len(s_cols):]] = sentiment_series
    return sent_dict




# print(list(sent_dict))
# person_list = list(sent_dict)[:5]
# plist = []
# gmin = 100
# gmax = -100
# for person in person_list:
#     sent_score = [i[1] for i in sent_dict[person]]
#     gmin = min(gmin, np.min(sent_score))
#     gmax = max(gmax, np.max(sent_score))

def gen_plot(sent_dict, terms, date_range):  
    """
    Input: sentiment dictionary for terms, terms
    Output: 
    """
    plist = []
    terms = list(sent_dict)
    fig, ax = plt.subplots(figsize=(8, max(5, len(terms))))
    #plt.rcParams['figure.facecolor'] = 'grey'
    terms = sorted(terms)[::-1]
    patches = []
    maxsent = 0
    minsent = 100
    for tc, term in enumerate(terms):
        t_data = sent_dict[term]
        # impute missing values
        sentiment = np.asarray([i[1] for i in t_data])
        mean = np.mean(sentiment[sentiment != 0])
        sentiment[sentiment == 0] = mean

        maxsent = max(np.nanmax(sentiment), maxsent)
        minsent = min(np.nanmin(sentiment), minsent)
        #print(sentiment, maxsent, minsent)
    for tc, term in enumerate(terms):
        #print(term)
        t_data = sent_dict[term]
        #    print([i[2] for i in t_data])
        dates = [i[0] for i in t_data]
        yval = [tc for _ in t_data]
        n_docs = [i[2] for i in t_data]
        #impute missing values, again
        sentiment = np.asarray([i[1] for i in t_data])
        mean = np.mean(sentiment[sentiment != 0])
        sentiment[sentiment == 0] = mean

        # sentiment = [i if not np.isnan(i) else min(sentiment_w_nans) for i in sentiment_w_nans]
        # sent_color = (np.asarray(sentiment) - gmin)/(gmax - gmin)
        #maxsent = np.nanmax(sentiment)
        #minsent = np.nanmin(sentiment)
        sent_color = ((np.asarray(sentiment) - minsent)/(maxsent - minsent))*(1)-0.5
        #print(sent_color)
        # print(sentiment, np.nanmin(sentiment),np.nanmax(sentiment))
        # plt.scatter(dates, yval, s=n_docs, color=plt.cm.RdYlBu(sent_color),zorder=1)
        for cd,date in enumerate(range(date_range[0], date_range[1]-1)):
            # print(date, n_docs, len(n_docs))
            # plt.hlines(tc, date-0.01, date+1.01, colors=plt.cm.seismic(1-sent_color[cd]), lw=np.sqrt(n_docs[cd]*0.9))
            x = np.asarray([date,date+1])
            y = (sent_color[cd+1] - sent_color[cd])*(x) + (sent_color[cd] - (date)*(sent_color[cd+1] - sent_color[cd])) + tc
            #print('y - ', y)
            #print('x - ', x)
            lwidths = np.sqrt((n_docs[cd+1] - n_docs[cd])*(x-date) + n_docs[cd])*0.005
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # vlines, ok but bumpy edges and overlapping vlines
            #ax.vlines(x, y-lwidths*0.5*0.02, y+lwidths*0.5*0.02, lw=0.7)
            # better but the linewidth is normal to the line direction, not the x-axis (so there are strange corners)
            #lc = LineCollection(segments, linewidths=lwidths, color='k')
            #ax.add_collection(lc)
            # polygons
            #for i in range(N):
            join_poly_inc = 1.02
            polygon = Polygon([ [date,y[0]+lwidths[0]],
                                [date+join_poly_inc,y[1]+lwidths[1]],
                                [date+join_poly_inc,y[1]-lwidths[1]],
                                [date,y[0]-lwidths[0]] ], True)
            patches.append(polygon)
            
            #colors = 100*np.random.rand(len(patches))

        plist.append(term)


    polygon = Polygon([ [1797,len(terms)-0.4+0.02],
                        [1799,len(terms)-0.4+0.07],
                        [1799,len(terms)-0.4-0.07],
                        [1797,len(terms)-0.4-0.02] ], True)
    patches.append(polygon)
        
    p = PatchCollection(patches)
    #p.set_array(np.array(colors))
    ax.add_collection(p)               
    plt.yticks(range(0,len(plist)),plist,rotation=0)

    plt.text(1800.1, len(terms) -1 + 0.25, "More \npositive", verticalalignment='center')
    plt.text(1800.1, len(terms) -1 - 0.25, "More \nnegative", verticalalignment='center')

    plt.text(1794.9, len(terms)-0.28, "Fewer mentions")#, bbox=dict(facecolor='white', edgecolor='none'))
    plt.text(1798.6, len(terms)-0.28, "More mentions" )#, bbox=dict(facecolor='white', edgecolor='none'))

    
    
    gov_types = ['Absolute \nMonarchy', 'Constitutional \nMonarchy', 'Convention', 'Directory', 'Consulat']
    gov_dates = [date_range[0], 1789 + 1/12, 1792 + 8/12, 1795 + 8/12, 1799 + 11/12, date_range[1]]
    
    g_color = ['red', 'blue', 'green', 'yellow', 'orange']
    timeline_y = -1.4#min(-0.9,-len(terms)*0.3)
    for cg, gov_type in enumerate(gov_types[:-1]):
        color = g_color[cg]
        #plt.hlines(timeline_y, gov_dates[cg], gov_dates[cg+1], colors=color, lw=15,zorder=0, alpha = 0.7)
        plt.axvline(gov_dates[cg+1], ymin = 0.00, color = 'k')
        plt.text((gov_dates[cg] + gov_dates[cg+1])/2, timeline_y+0.1, gov_type, ha='center')

    for h in range(len(terms)+1):
        plt.axhline(h-0.5, color='k', lw=0.7)
    plt.xlim((date_range[0], date_range[1]))
    plt.ylim((-1.9,len(terms)-0.3))
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(which='major', color='grey', lw=0.2)
    # ax.set_facecolor('xkcd:ecru')

    plt.savefig('tmp.png', facecolor = fig.get_facecolor(), transparent = True)

