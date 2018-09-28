import pandas as pd
import pickle
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("french")



def display_texts(term):

    text_list = pickle.load(open('./pickles/'+term+'_words.pkl', 'rb'))
    SA = pickle.load(open('./pickles/3_sentiment_dictionary_stem_FEEL.pkl', 'rb'))
    sent_df = pickle.load(open('./pickles/3_df_sentiment.pkl','rb'))
    # print(list(sent_df))
    term_df = pickle.load(open('./pickles/'+term+'_df.pkl', 'rb'))#sent_df.loc[sent_df['identifier_'+term].notnull()]
    # print(len(term_df))
    top_doc_dict = {}
    bot_doc_dict = {}
    # print(text_list)
    top_5 = term_df.nlargest(5, 'sentiment_vals_unw_'+term)['identifier'].tolist()
    bottom_5 = term_df.nsmallest(5, 'sentiment_vals_unw_'+term)['identifier'].tolist()
    #print(top_5, bottom_5)
    for doc in text_list:
        # print(doc[0], doc[1])
        # print(len(doc), len(doc[1]))
        
        # fix this so that the passages are stored in a dictionary
        if doc[0][:-9] in top_5:
            for text in doc[1]:
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
                            rt.append('<font color="red">'+word+'</font>')
                    try:
                        top_doc_dict[doc[0]].append(' '.join(rt))
                    except:
                        top_doc_dict[doc[0]] = [' '.join(rt)]
                        
                #doc_dict[doc[0]] = [' '.join(passage)  for text in doc[1] for passage in text]
            # if len(doc[1]) > 1:
            # print(doc_dict[doc[0]])
    for doc in text_list:
        # print(doc[0], doc[1])
        # print(len(doc), len(doc[1]))
        
        # fix this so that the passages are stored in a dictionary
        if doc[0][:-9] in bottom_5:
            for text in doc[1]:
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
                            rt.append('<font color="red">'+word+'</font>')
                    try:
                        bot_doc_dict[doc[0]].append(' '.join(rt))
                    except:
                        bot_doc_dict[doc[0]] = [' '.join(rt)]
                        

    return top_doc_dict, bot_doc_dict

# print(display_texts('robespierre'))
