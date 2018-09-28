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


import matplotlib.pyplot as plt



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
        endpoint = last_endpoint + 50#int(block_size*0.1)
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
    date_range = (1785,1805)
    full_df = get_full_df()
    full_term_data = []
    year_word_variation = []
    year_seq = []

    for year in range(date_range[0], date_range[1]):
        docs_in_year = get_docs_in_year(full_df, year)
        #print(docs_in_year)
        year_dict = Counter({})
        terms = []
        freqs = []
        print(year)
        year_average = 0
        year_count = 0
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
            doc_word_variation = get_word_variation(split_text, block_size=300)
            if doc_word_variation:
                year_average += doc_word_variation
                year_count += 1
        year_word_variation.append(year_average/year_count)
        year_seq.append(year)
    plt.plot(year_seq, year_word_variation)
    plt.show()
                

if __name__ == "__main__":
    main()
