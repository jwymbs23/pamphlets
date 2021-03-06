import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
import pandas as pd
import seaborn as sns


#data from containing document metadata, cleaned to include relevant columns, identifier matches doc_name in the SA_term dataframe
#doc_data = pickle.load(open('df_relevant.pkl','rb'))
#doc_data.set_index('identifier',drop=False,inplace=True,verify_integrity=True)
#print(doc_data.head(3))

def make_sa_df(sa_data, TERM):
    # take list of term sentiment scores by document and make a dataframe
    sa_dict = {'identifier': [], 'sentiment_vals_w_'+TERM: [], 'certainty_vals_'+TERM: [], 'sentiment_vals_unw_'+TERM: []}
    #, 'total_doc_score_'+TERM: []}
    certainty_vals = []
    sentiment_vals = []
    for single_doc_sa_data in sa_data:
        # ensure that all data is present
        if len(single_doc_sa_data) == 5:
            doc_id = single_doc_sa_data[0][:-9]
            term = single_doc_sa_data[1]
            # print(term)
            # add data to dict that will become the dataframe
            sa_dict['identifier'].append(doc_id)
            
            # sa_dict['term'].append(single_doc_sa_data[1])
            sa_dict['sentiment_vals_w_'+TERM].append(float(single_doc_sa_data[2]))
            sa_dict['sentiment_vals_unw_'+TERM].append(float(single_doc_sa_data[4]))
            #sa_dict['total_doc_score_'+TERM].append(float(single_doc_sa_data[5]))
            # print(term_group, single_doc_sa_data[3])
            # this calculates the stdev of sentiment scores for a term WITHIN a document (how consistently does a document treat a subject positively or negatively)
            sa_dict['certainty_vals_'+TERM].append(np.std(single_doc_sa_data[3]))
            if len(single_doc_sa_data[3]) > 1:
                # print(single_doc_sa_data[3])
                sentiment_vals.append(single_doc_sa_data[2])
                # certainty is how consistent the sentiment values were accross all mentions of the search term in a single document
                certainty_vals.append(np.std(single_doc_sa_data[3]))
    # uncomment to plot 2d hist of within document certainty against sentiment score (see if more extreme emotions correlate with more consistent expression of emotion)
    # --inconclusive
    # norm = colors.LogNorm()
    # print(certainty_vals)
    # plt.title(term)
    # plt.hist2d(sentiment_vals, certainty_vals,bins=15)
    # plt.show()
    sentiment_df = pd.DataFrame(data = sa_dict)
    #    print(sa_dict['date'], sentiment_df)
    sentiment_df.set_index('identifier',inplace=True,drop=False)

    return sentiment_df
# doc_data.join(sentiment_df, rsuffix='_'+TERM, how = 'outer')



def plot_term_score_data(sa_df, TERM, weighted, date_range):
    if weighted == True:
        sa_col = u'sentiment_vals_w_'+TERM
    else:
        sa_col = u'sentiment_vals_unw_'+TERM
    # if total_doc == True:
    #     sa_col = u'total_doc_score_'+TERM
    print(sa_col, list(sa_df))
    print(sa_df['date'].head())
    term_df = sa_df.loc[sa_df[sa_col].notnull()]
    print(term_df)
    print(term_df['date'].head())
    # pick out relevant years (future note: some docs have more detailed publication info)
    start_date = date_range[0]
    end_date = date_range[1]
    # sa_df['date'].hist(bins = list(range(start_date,end_date)))
    # count_by_date = term_df.groupby('date').count()


    # TERM MENTION SHARE BY YEAR #
    full_dates = sa_df['date'].dropna()
    full_date_list = full_dates.tolist()
    n_years = end_date - start_date
    d_range = (start_date, end_date)
    # print(full_date_list)
    full_date_hist, full_date_bins = np.histogram(full_date_list, bins=n_years, range=d_range)
    # print(full_date_hist)
    term_dates = term_df['date'].dropna()
    # print(term_dates)
    term_date_list = term_dates.tolist()
    # plt.gcf().set_facecolor('white')
    term_date_hist, term_date_bins = np.histogram(term_date_list,
                                                  bins=end_date - start_date,
                                                  range=d_range)

    mean_by_date = term_df.groupby(u'date').mean()
    print(mean_by_date, term_date_bins)
    sent_list = mean_by_date.loc[term_date_bins[:-1]][sa_col].tolist()
    print(sent_list)
    
    plt.xlabel('Year')
    plt.ylabel('Share of Documents')
    plt.xticks(range(start_date,end_date,1))
    plt.title('Share of documents by year in which the term "%s" appears'%(TERM))
    plt.ticklabel_format(useOffset=False)
    print(term_date_bins[:-1], term_date_hist, full_date_hist)
    mention_by_year = term_date_hist.astype(float)/full_date_hist
    plt.bar(term_date_bins[:-1], mention_by_year, width=1,
            color=plt.cm.RdYlBu((np.asarray(sent_list) - min(sent_list))/(max(sent_list) - min(sent_list))))
    plt.tight_layout()
    # plt.rcParams['axes.facecolor']='white'
    plt.show()

    # SENTIMENT SCORE AND DISTRIBUTIONS BY YEAR #

    
    # std_by_date = term_df.groupby(u'date').std()
    # print(mean_by_date)
    # mean_by_date = mean_by_date.reset_index()
    # print(mean_by_date)
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.ticklabel_format(useOffset=False)

    plt.gcf().set_facecolor('white')
    
    # f, (ax1, ax2) = plt.subplots(2)
    sa_by_date = term_df.loc[(term_df['date'] >= start_date) &
                             (term_df['date'] <= end_date)]
    sa_date_mean_xlim = (start_date - 0.5, end_date + 0.5)
    sa_date_mean_ylim = (-0.5, 0.5)
    
    mean_by_date.plot(use_index=True, y=sa_col, title=TERM+' mean',
                      xlim=sa_date_mean_xlim,
                      color='red', ax=ax1)
    # , ylim=sa_date_mean_ylim,
    sns.violinplot(x='date', y=sa_col, data=sa_by_date, color='green',
                   saturation=0.4, ax=ax2, inner='box', gridsize=500)

    sns.stripplot(x='date', y=sa_col, data=sa_by_date,
                  jitter=True, color="y", alpha=.3, ax=ax2)

    # sns.swarmplot(x='date', y=sa_col,data=term_df.loc[
    # (term_df['date'] > start_date) & (term_df['date']
    # <= end_date)], color="g", alpha=.3, ax= ax2)
    ax1.grid(color='gray', linestyle='--', linewidth=1, axis='y',
             which='major', alpha=0.5)
    ax2.grid(color='gray', linestyle='--', linewidth=1, axis='y',
             which='major', alpha=0.5)
    plt.show()
    # std_by_date.plot(use_index = True,y='certainty_vals',title=TERM + \
    # ' aggregate uncertainty',xlim = (start_date,end_date))
    # plt.show()
