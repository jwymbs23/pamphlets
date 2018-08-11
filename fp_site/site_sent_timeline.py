import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pickle


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
    print(list(sent_df))
    if weight_flag:
        s_cols = 'sentiment_vals_w_'
    else:
        s_cols = 'sentiment_vals_unw_'

    doc_count_cutoff = 0
    
    docs_per_year = sent_df.groupby('date').count()
    # print(docs_per_year)
    terms = sorted(terms)
    for raw_term in terms:
        term = s_cols + raw_term
        doc_count = sent_df[term].count()
        print(doc_count)
        if doc_count > doc_count_cutoff:
            mean_by_date = sent_df[term].groupby(sent_df[u'date']).mean()
            date_count = sent_df[term].groupby(sent_df['date']).count().tolist()
            dates = mean_by_date.keys().tolist()
            scores = mean_by_date.tolist()
            sentiment_series = [(date,score,dcount) for date, score,dcount in zip(dates,scores,date_count)
                                if date > date_range[0] and date <= date_range[1]]
            sent_dict[term[len(s_cols):]] = sentiment_series
    return sent_dict

# 
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
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_facecolor('xkcd:ecru')
    #plt.rcParams['figure.facecolor'] = 'grey'
    terms = sorted(terms)[::-1]
    for tc, term in enumerate(terms):
        print(term)
        t_data = sent_dict[term]
        #    print([i[2] for i in t_data])
        dates = [i[0] for i in t_data]
        yval = [tc for _ in t_data]
        n_docs = [i[2] for i in t_data]
        sentiment = [i[1] for i in t_data]
        # sentiment = [i if not np.isnan(i) else min(sentiment_w_nans) for i in sentiment_w_nans]
        # sent_color = (np.asarray(sentiment) - gmin)/(gmax - gmin)
        maxsent = np.nanmax(sentiment)
        minsent = np.nanmin(sentiment)
        sent_color = (np.asarray(sentiment) - minsent)/(maxsent - minsent)
        # print(sentiment, np.nanmin(sentiment),np.nanmax(sentiment))
        # plt.scatter(dates, yval, s=n_docs, color=plt.cm.RdYlBu(sent_color),zorder=1)
        for cd,date in enumerate(range(date_range[0], date_range[1])):
            #print(date, n_docs, len(n_docs))
            plt.hlines(tc, date-0.01, date+1.01, colors=plt.cm.seismic(1-sent_color[cd]), lw=np.sqrt(n_docs[cd]*0.9))
    
        plist.append(term)
    plt.yticks(range(0,len(plist)),plist,rotation=0)
    
    # timeline info
    
    # events_df = pd.DataFrame({'event_name': ['Accession of Louis XVI','Tennis Court Oath','Taking of the Bastille','Declaration of the Rights of Man','L\'Ami du Peuple appears','Women of Paris march to Versailles','Abolition of nobility and titles','Flight of the royal family', 'Massacre at Chaps-de-Mars','Constitution of 1791','War between France and Austria','Monarchy overthrown','September Massacres', 'France declared a Republic','Execution of Louis XVI','Committee of Public Safety established','Assassination of Jean-Paul Marat', 'The Terror begins', 'Execution of Marie Antionette','Execution of Georges Danton','Execution of Robespierre','End of the convention','Napoleon\'s Italian campaign begins', 'Napoleon\'s coup d\'etat'],
    #                           'date': ['1774','June 20, 1789','July 14, 1789','August 26, 1789','September 12, 1789','October 5, 1789','June 19, 1790','June 20, 1791','July 17, 1791','September 3, 1791','April 18, 1792','August 10, 1792','September 2, 1792','September 21, 1792','January 21, 1793','April 6, 1793', 'July 13, 1793', 'September 17, 1793','October 16, 1793','April 5, 1794','July 27, 1794','October 31, 1795','April 10, 1796','November 9, 1799'],
    #                           'importance': [1,1,1,1,4,1,2,1,2,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1]})
    # events_df['date'] = pd.to_datetime(events_df['date'])
    
    gov_types = ['Absolute Monarchy', 'Constitutional \nMonarchy', 'Convention', 'Directory', 'Consulat']
    gov_dates = [date_range[0], 1789 + 1/12, 1792 + 8/12, 1795 + 8/12, 1799 + 11/12, date_range[1]]
    
    g_color = ['red', 'blue', 'green', 'yellow', 'orange']
    timeline_y = -len(terms)*0.15
    for cg, gov_type in enumerate(gov_types[:-1]):
        color = g_color[cg]
        #plt.hlines(timeline_y, gov_dates[cg], gov_dates[cg+1], colors=color, lw=15,zorder=0, alpha = 0.7)
        plt.axvline(gov_dates[cg+1], ymin = 0.05, color = 'k')
        plt.text((gov_dates[cg] + gov_dates[cg+1])/2, timeline_y+0.1, gov_type, ha='center')
    plt.xlim((date_range[0], date_range[1]))
    plt.ylim((-len(terms)*0.2,len(terms)-0.8))
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #ax.set_facecolor('xkcd:ecru')
    

    plt.savefig('tmp.png', facecolor = fig.get_facecolor(), transparent = True)

