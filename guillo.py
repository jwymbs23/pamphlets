import pandas as pd
import matplotlib
matplotlib.use('Agg')

from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

import matplotlib.pyplot as plt
import numpy as np
import pickle

import new_term as new_term

# sent_df = pickle.load(open('../pickles/3_df_sentiment.pkl', 'rb'))

# all_cols = list(sent_df)

# print(all_cols)

# people_cols = [col for col in all_cols if col.find(s_cols) != -1]# and col.find(u'voltaire') == -1 and col.find(u'rousseau') == -1]

# sent_dict = {}


def get_sent_dict(sent_df, terms, guill_year, weight_flag = True, date_range = (1786,1800)):
    """
    Input:  sentiment dataframe, terms to plot
    Output: 
    
    """
    sent_dict = {}
    # print(list(sent_df))
    if weight_flag:
        s_cols = 'sentiment_vals_w_'
    else:
        s_cols = 'sentiment_vals_unw_'
    #s_cols = 'certainty_vals_'
    
    doc_count_cutoff = 300
    
    # docs_per_year = sent_df.groupby('date').count()
    # print(docs_per_year)
    terms = sorted(terms)
    for raw_term in terms:
        term = s_cols + raw_term
        try:
            term_df = pickle.load(open('./pickles/'+raw_term+'_df.pkl', 'rb'))
        except:
            new_term.add_new_term(raw_term, 'n', 30)
            term_df = pickle.load(open('./pickles/'+raw_term+'_df.pkl', 'rb'))
        #print(list(term_df))
        term_df = pd.merge(term_df, sent_df[['identifier', 'date']], on='identifier')
        #print(list(term_df))
        doc_count = term_df[term].count()
        # print(doc_count)
        if doc_count > doc_count_cutoff:
            mean_by_date = term_df.groupby('date')[term].mean()
            date_count = term_df.groupby('date')[term].count().tolist()
            dates = mean_by_date.keys().tolist()
            scores = mean_by_date.tolist()
            sentiment_series = []
            #print(dates)
            for cd, date in enumerate(range(date_range[0], date_range[1])):
                if date in dates:
                    date_idx = dates.index(date)
                    #normalize
                    #sentiment_series.append((dates[date_idx], scores[date_idx] - np.mean(scores[date_idx]), date_count[date_idx]))
                    sentiment_series.append((dates[date_idx], scores[date_idx], date_count[date_idx]))
                elif date >= date_range[0] and date < date_range[1]:
                    sentiment_series.append((date, 0,0))
                    
            # sentiment_series = [(date,score,dcount) for date, score,dcount in zip(dates,scores,date_count)
            #                     if date >= date_range[0] and date < date_range[1]]
            #print(sentiment_series)
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



#     mean = np.zeros(date_range[1] - date_range[0])
#     
#     for dterm in sent_dict:
#         #print(sent_dict[dterm], len(sent_dict[dterm]))
#         mean += np.asarray([i[1] for i in sent_dict[dterm]])
#     print(mean/len(sent_dict))
#     
#     sent_dict['mean'] = [(year, mean[cy]/len(sent_dict), 0) for cy, year in enumerate(range(date_range[0], date_range[1]))]




def gen_plot(sent_dict, terms, guill_year, date_range):  
    """
    Input: sentiment dictionary for terms, terms
    Output: 
    """
    plist = []
    # terms = list(sent_dict)
    fig, ax = plt.subplots(figsize=(8, 5))
    #plt.rcParams['figure.facecolor'] = 'grey'
    # terms = sorted(terms)[::-1]
    patches = []
    print(terms, list(sent_dict))
    mean = {}
    mean_count = {}
    for tc, term in enumerate(terms):
        # print(term)
        if term in sent_dict:
            t_data = sent_dict[term]
            #    print([i[2] for i in t_data])
            dates = [i[0] - guill_year[tc] for i in t_data]
            yval = [tc for _ in t_data]
            n_docs = [i[2] for i in t_data]
            #sentiment = [i[1] for i in t_data]

            sentiment = np.asarray([i[1] for i in t_data])
            sent_mean = np.mean(sentiment[sentiment != 0])
            sentiment[sentiment == 0] = sent_mean
            #print(sentiment)

            # sentiment = [i if not np.isnan(i) else min(sentiment_w_nans) for i in sentiment_w_nans]
            # sent_color = (np.asarray(sentiment) - gmin)/(gmax - gmin)
            # maxsent = np.nanmax(sentiment)
            # minsent = np.nanmin(sentiment)
            # sent_color = (np.asarray(sentiment) - minsent)/(maxsent - minsent)*0.5
            sent_color = np.asarray(sentiment)
            plt.plot(dates, sent_color, lw=0.5)
            for ty, diffyear in enumerate(dates):
                print(diffyear)
                print(ty, sent_color[ty], sentiment[ty], sent_color)
                try:
                    mean[diffyear] += sent_color[ty]
                except:
                    mean[diffyear] = sent_color[ty]
                try:
                    mean_count[diffyear] += 1
                except:
                    mean_count[diffyear] = 1

    for diffyear in mean:
        mean[diffyear] /= mean_count[diffyear]
    mean_list = list(mean.items())
    mean_list = sorted(mean_list, key=lambda k: k[0])[::-1]
    print(mean_list)
    mean_dates = [i[0] for i in mean_list]
    mean_sent = [i[1] for i in mean_list]
    plt.plot(mean_dates, mean_sent, lw=5)


    plt.xlim((-6,4))#(date_range[0], date_range[1]))
    #plt.ylim((-0.06,0.06))

#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.spines['bottom'].set_visible(False)
#    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(which='major')
    ax.xaxis.grid(which='major')
    ax.set_xticks(range(-6,6))
    plt.ylabel('Sentiment Score')# Standard Deviation')
    plt.xlabel('Years until Guillotined')
    #plt.legend()
    # ax.set_facecolor('xkcd:ecru')
    plt.tight_layout()

    #fn = get_sample_data("./fut_guillotinee/guill_blade.png", asfileobj=False)
#    arr_img = plt.imread('./fut_guillotinee/guill_blade.png', format='png')
#    
#    imagebox = OffsetImage(arr_img, zoom=0.1)
#    imagebox.image.axes = ax
#    xy = [0.2, 0.08]
#    
#    ab = AnnotationBbox(imagebox, xy)
#    #                    xybox=(120., -80.),
#    #                    xycoords='data',
#    #                    boxcoords="offset points",
#    #                    pad=0.5,
#    #                    arrowprops=dict(
#    #                        arrowstyle="->",
#    #                        connectionstyle="angle,angleA=0,angleB=90,rad=3")
#    #)
#    
#    ax.add_artist(ab)
    

    
    plt.savefig('./fut_guillotinee/tmp.png', facecolor = fig.get_facecolor(), transparent = True)


def main():
    sent_df = pickle.load(open('./pickles/3_df_sentiment.pkl', 'rb'))

    guill_dict = {'babeuf': 1797,
                  'barbaroux': 1794,
                  'barnave': 1793,
                  'boucheporn': 1794,
                  'boyenval': 1795,
                  'brissot': 1793,
                  'carrier': 1794,
                  'chalier': 1793,
                  'chaumette': 1794,
                  'coffinhal': 1794,
                  'angremont': 1792,
                  'balleroy': 1794,
                  'costard': 1793,
                  'couthon': 1794,
                  'delacroix': 1794,
                  'kersaint': 1793,
                  'laborde': 1794,
                  'scellier': 1795,
                  'tessier': 1794,
                  'valz': 1794,
                  'sombreuil': 1794,
                  'broglie': 1794,
                  'woestyne': 1794,
                  'laporte': 1792,
                  'cazotte': 1792,
                  'xvi': 1793,
                  'antoinette': 1793,
                  'barry': 1793,
                  'corday': 1793,
                  'gouges': 1793,
                  'bailly': 1793,
                  'roland': 1793,
                  'lavoisier': 1794,
                  'élisabeth': 1794,
                  'danton': 1794,
                  'dillon': 1794,
                  'hébert': 1794,
                  'desmoulins': 1794,
                  'séchelles': 1794,
                  'westermann': 1794,
                  'philippeaux': 1794,
                  'robespierre': 1794,
                  'beauharnais': 1794,
                  'chénier': 1794}
                  
    dict_to_list = list(guill_dict.items())
    # print(dict_to_list)
    guillotined = [i[0] for i in dict_to_list]
    # print(guillotined)
    guill_year = [i[1] for i in dict_to_list]
    #guillotined = ['robespierre', 'xvi']
    #guill_year = [1794, 1793]
    # get list of terms to plot
    #terms = flask.request.values.get('terms')

    weight_flag = True
    date_range = (1786,1801)
    
    checked_term_dict = get_sent_dict(sent_df, guillotined, guill_year, weight_flag, date_range)
    
    
    gen_plot(checked_term_dict, guillotined, guill_year, date_range)
    plt.show()    
    

if __name__ == "__main__":
    main()
        
