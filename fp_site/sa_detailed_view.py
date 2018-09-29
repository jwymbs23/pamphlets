import pandas as pd
import pickle
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("french")


def get_passages(doc_ids, text_list, SA):
    year_passages = []            
    for doc_id in doc_ids:
        doc_passages = []
        #print(len(text_list[doc_id+'_djvu.txt']))
        for text in text_list[doc_id+'_djvu.txt']:
            # print('\n\nttt', text)
            for passage in text:
                # print('\n\n passage',' '.join(passage))
                rt = []
                for word in passage:
                    stem_word = stemmer.stem(word)
                    if stem_word not in SA:
                        rt.append(word)
                    elif SA[stem_word] > 0:
                        rt.append('<font color="green">'+word+'</font>')
                    else:
                        rt.append('<font color="#FF5353">'+word+'</font>')
                doc_passages.append(' '.join(rt))
        year_passages.append(doc_passages)
    return year_passages



def display_texts(term, weight_flag=True):

    text_list = pickle.load(open('./pickles/'+term+'_words.pkl', 'rb'))
    SA = pickle.load(open('./pickles/3_sentiment_dictionary_stem_FEEL.pkl', 'rb'))
    sent_df = pickle.load(open('./pickles/3_df_sentiment.pkl','rb'))
    # print(list(sent_df))
    term_df = pickle.load(open('./pickles/'+term+'_df.pkl', 'rb'))#sent_df.loc[sent_df['identifier_'+term].notnull()]
    term_df = pd.merge(term_df, sent_df[['identifier', 'date', 'title']], on='identifier')

    #print('hi', term_df)
    # print(len(term_df))
    top_doc_dict = {}
    bot_doc_dict = {}
    years = term_df['date'].unique()
    #print(years)
    years = [int(year) for year in years if year > 1784 and year < 1801]
    if weight_flag == True:
        sent_col = 'sentiment_vals_w_'+term
    else:
        sent_col = 'sentiment_vals_unw_'+term

    for year in years:
        top_5 = term_df.loc[term_df['date'] == year].nlargest(5, sent_col)
        top_5_id = top_5['identifier'].tolist()
        top_5_score = top_5[sent_col].tolist()
        top_5_title = top_5['title'].tolist()
        #print(top_5_id)
        #print(top_5_score)
        top_doc_dict[str(year)] = [top_5_id, top_5_score, top_5_title]
        
        bot_5 = term_df.loc[term_df['date'] == year].nsmallest(5, sent_col)
        bot_5_id = bot_5['identifier'].tolist()
        bot_5_score = bot_5[sent_col].tolist()
        bot_5_title = bot_5['title'].tolist()
        bot_doc_dict[str(year)] = [bot_5_id, bot_5_score, bot_5_title]
        #print(bot_doc_dict)
        # print(doc[0], doc[1])
        # print(len(doc), len(doc[1]))
        #print(list(text_list))
        #print('llllllll', list(top_doc_dict))
        

        top_doc_dict[str(year)].append(get_passages(top_5_id, text_list, SA))
        bot_doc_dict[str(year)].append(get_passages(bot_5_id, text_list, SA))
        
    # overall most positive and negative documents
    top_5 = term_df.nlargest(5, sent_col)
    top_5_id = top_5['identifier'].tolist()
    top_5_score = top_5[sent_col].tolist()
    top_5_title = top_5['title'].tolist()
    top_doc_dict['overall'] = [top_5_id, top_5_score, top_5_title]
    
    bot_5 = term_df.nsmallest(5, sent_col)
    bot_5_id = bot_5['identifier'].tolist()
    bot_5_score = bot_5[sent_col].tolist()
    bot_5_title = bot_5['title'].tolist()
    bot_doc_dict['overall'] = [bot_5_id, bot_5_score, bot_5_title]
    
    top_doc_dict['overall'].append(get_passages(top_5_id, text_list, SA))
    bot_doc_dict['overall'].append(get_passages(bot_5_id, text_list, SA))
                            
    return top_doc_dict, bot_doc_dict
