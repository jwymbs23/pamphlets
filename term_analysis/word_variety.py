import lucene
import numpy as np
import string

from org.apache.lucene.analysis.standard import StandardAnalyzer

from java.nio.file import Paths
from org.apache.lucene.index import IndexReader, Term
from org.apache.lucene.index import \
        IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory
from java.io import File
from java.io import StringReader
from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import \
        MultiPhraseQuery, PhraseQuery, DocIdSetIterator
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import BytesRefIterator
from org.apache.lucene.search import TermQuery


import sys
from collections import Counter

import pickle
import glob
import pandas as pd
from stopwords import *

from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

import matplotlib.pyplot as plt

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("french")

SA_dict = pickle.load(open('../pickles/3_sentiment_dictionary_stem_FEEL.pkl', 'rb'))

def define_search_params(STORE_DIR, FIELD_CONTENTS, TERM):
    
    store = SimpleFSDirectory(Paths.get(STORE_DIR))
    reader = DirectoryReader.open(store)
    searcher = IndexSearcher(reader)
        
    # Get the analyzer
    analyzer = WhitespaceAnalyzer()
    # Constructs a query parser. We specify what field to search into.
    queryParser = QueryParser(FIELD_CONTENTS, analyzer)
    
    # Create the query
    query = queryParser.parse(TERM)
    return searcher, reader, query



def get_dict(stem_flag = True):
    if stem_flag == True:
        #Sentiment dictionary:
        SA_dict = pickle.load(open('./pickles/3_sentiment_dictionary_stem_FEEL.pkl', 'rb'))
    else:
        SA_dict = pickle.load(open('./pickles/3_sentiment_dictionary_FEEL.pkl', 'rb'))
    return SA_dict

def get_full_df(remake_df = True):
    return pickle.load(open('../pickles/3_df_relevant.pkl','rb'))
 

def get_docs_in_year(df, year):
    return df.loc[df['date'] == year]['identifier'].tolist()


def clean_term_dict(full_term_data):
    for year_dict in full_term_data:
        for term in list(year_dict.keys()):
            if len(term) < 3 or term in stopwords or '.' in term or year_dict[term] < 10:
                del year_dict[term]
    return full_term_data


def trending_plot(tfidf, top_words_year):
    trending_df = pd.DataFrame()
    years = [i[0] for i in top_words_year]
    n_years = len(tfidf)
    trending_df['year'] = years
    for year_data in top_words_year:
        for term in year_data[1:]:
            for year in years:
                term_series = [tfidf[yc][term] if term in tfidf[yc] else 0 for yc in range(n_years)]
            trending_df[term] = term_series
    print(trending_df)
    trending_df.to_pickle('trending_df.p')


def tfidf_method(full_term_data, date_range):
    n_years = date_range[1] - date_range[0]
    # idf
    idf = {}
    total_wc_year = []
    full_term_data = clean_term_dict(full_term_data)
    
    for year_dict in full_term_data:
        total_wc_year.append(sum(year_dict.values()))
        for term in year_dict:
            try:
                idf[term] += 1
            except:
                idf[term] = 1
                #            if term not in idf:
                #                term_idf = 0
                #                for term_year_dict in full_term_data:
                #                    if term in term_year_dict:
                #                        term_idf += 1
                #                        # print(term)
                #                idf[term] = term_idf
    print(idf['bonaparte'])
    
    # tf-idf
    tfidf = [{} for _ in range(n_years+1)]
    print(tfidf)
    for yc, year_dict in enumerate(full_term_data):
        for term in year_dict:
            tfidf[yc][term] = year_dict[term]/total_wc_year[yc] * np.log(n_years/(1 + idf[term]))

    top_words_year = []
    for yc, year_dict in enumerate(tfidf):
        top_words_year.append([date_range[0] + yc] + sorted(tfidf[yc] ,key=tfidf[yc].get, reverse=True)[:20])

    trending_plot(tfidf, top_words_year)
    return top_words_year


def variation(text_block):
    return len(Counter(text_block))/len(text_block)


def get_emotion(split_text):
    doc_sent = 0
    for word in split_text:
        stem = stemmer.stem(word)
        if stem in SA_dict:
            doc_sent += SA_dict[stem]
#    print(doc_sent)
    return doc_sent/len(split_text)
        

def get_word_variation(split_text, block_size=1000):
    max_word = len(split_text)
    doc_var = 0
    n_blocks = 0
    endpoint = block_size
    last_endpoint = 0
    while endpoint < max_word:
        doc_var += variation(split_text[last_endpoint:endpoint])
        n_blocks += 1
        last_endpoint = endpoint
        endpoint = last_endpoint + block_size#int(block_size*0.1)
    if n_blocks == 0:
        return False
    else:
        return doc_var/n_blocks



def main():
    #constants
    FIELD_CONTENTS = "vectext"
    DOC_NAME = "identifier"
    STORE_DIR = "../full_index1"

    lucene.initVM()
    store = SimpleFSDirectory(Paths.get(STORE_DIR))    
    
    ireader = DirectoryReader.open(store)#, True)
    #print(ireader.readerIndex(0))

    searcher = IndexSearcher(ireader)#self.getSearcher()
                    
    pickle_file = glob.glob('full_word_list.pkl')
    print(pickle_file)
    date_range = (1780,1805)
    full_df = get_full_df()
    full_term_data = []
    year_word_variation = []
    year_emotion = []
    year_emotion_std = []
    year_seq = []
    num_docs = []
    stored_vals = [1780,
                   0.10567632363375747,
                   1781,
                   0.10412921904969506,
                   1782,
                   0.11400923973621961,
                   1783,
                   0.11624698447126122,
                   1784,
                   0.1089380996206722,
                   1785,
                   0.10969358290259144,
                   1786,
                   0.11472516095726447,
                   1787,
                   0.09005205157404028,
                   1788,
                   0.09027335272279774,
                   1789,
                   0.08566105985096027,
                   1790,
                   0.08772220401513957,
                   1791,
                   0.0961809473686933,
                   1792,
                   0.09578126142310114,
                   1793,
                   0.09608189099292107,
                   1794,
                   0.09999430344087938,
                   1795,
                   0.10572740638033541,
                   1796,
                   0.11579715269837429,
                   1797,
                   0.10610471468124912,
                   1798,
                   0.10448169145472673,
                   1799,
                   0.11285257261130252,
                   1800,
                   0.1245771550398246,
                   1801,
                   0.12007902484159538,
                   1802,
                   0.12453526437058295,
                   1803,
                   0.11126328454887868,
                   1804,
                   0.11294224429883938]
    for year in range(date_range[0], date_range[1]):
        docs_in_year = get_docs_in_year(full_df, year)
        #print(docs_in_year)
        year_dict = Counter({})
        terms = []
        freqs = []
        print(year)
        year_average = 0
        year_count = 0
        year_sent = 0
        year_docs = []
        num_docs.append(len(docs_in_year))

        if not stored_vals:
            for cd, doc_id in enumerate(docs_in_year):
                #if not cd%100:
                #    print(cd , '--', len(docs_in_year))
                # get document (query by id)
                q = TermQuery(Term("identifier", doc_id+'_djvu.txt'))
                topDocs = searcher.search(q, 50000)
                
                #termvec = reader.getTermVector(topDocs.scoreDocs[0].doc, "all")
                one_doc = topDocs.scoreDocs[0].doc
                doc_name = searcher.doc(one_doc)
                #print(doc_name, doc_id)
                text = doc_name.get("text")
                split_text = text.split()
                #doc_word_variation = get_word_variation(split_text, block_size=300)
                #if doc_word_variation:
                #    year_average += doc_word_variation
                #    year_count += 1
                if len(split_text) > 0:
                    emotion = get_emotion(split_text)
                    year_sent += emotion
                    year_docs.append(emotion)
                    year_count += 1
                #if year_count > 1:
                #    break
            #year_word_variation.append(year_average/year_count)
            year_emotion.append(year_sent/year_count)
            year_emotion_std.append(np.std(year_docs))
            print(year_sent/year_count)
            year_seq.append(year)
        else:
            year_seq = [i for ci, i in enumerate(stored_vals) if not ci%2]
            year_emotion = [i for ci,i in enumerate(stored_vals) if ci%2]
    print(year_emotion, year_seq)
    #plt.plot(year_seq, year_word_variation)
    fig, ax = plt.subplots(figsize=(8, 5))
    

    # vlines, ok but bumpy edges and overlapping vlines
    #ax.vlines(x, y-lwidths*0.5*0.02, y+lwidths*0.5*0.02, lw=0.7)
    # better but the linewidth is normal to the line direction, not the x-axis (so there are strange corners)
    #lc = LineCollection(segments, linewidths=lwidths, color='k')
    #ax.add_collection(lc)
    # polygons
    #for i in range(N):
    patches = []
    for cd, date in enumerate(year_seq[:-1]):
        x = np.asarray([date,date+1])    
        lwidths = np.sqrt((num_docs[cd+1] - num_docs[cd])*(x-date) + num_docs[cd])*0.0001
        points = np.array([year_seq, year_emotion]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        join_poly_inc = 1.02
        y = [year_emotion[cd], year_emotion[cd+1]]
        polygon = Polygon([ [date,y[0]+lwidths[0]],
                            [date+join_poly_inc,y[1]+lwidths[1]],
                            [date+join_poly_inc,y[1]-lwidths[1]],
                            [date,y[0]-lwidths[0]] ], True)
        patches.append(polygon)
                                                                                                                                                                

    #p.set_array(np.array(colors))
    annote_y = 0.125
    plt.text(1789.2-5, annote_y, "Fewer documents -", verticalalignment='center')#, bbox=dict(facecolor='white', edgecolor='none'))
    plt.text(1795.2-5, annote_y, "- More documents" , verticalalignment='center')#, bbox=dict(facecolor='white', edgecolor='none'))
    polygon = Polygon([ [1794-5,annote_y+0.001],
                        [1795-5,annote_y+0.003],
                        [1795-5,annote_y-0.003],
                        [1794-5,annote_y-0.001] ], True)
    patches.append(polygon)
        

    p = PatchCollection(patches)    
    ax.add_collection(p)
    
    plt.plot(year_seq, year_emotion, c='r')
    #plt.plot(year_seq, year_emotion_std)
    plt.ylabel('Average Sentiment Score over All Documents')
    plt.xlabel('Year')
    plt.xticks(year_seq[::2])
    plt.grid(which='major', lw=0.5, alpha=0.6)
    plt.xlim((year_seq[0]-0.5, year_seq[-1]+0.5))
    plt.tight_layout()    
    plt.savefig('total_sent.png')#, transparent = True)    
    plt.show()
                

if __name__ == "__main__":
    main()
