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


def zscore_method(full_term_data, date_range):
    word_list = set()
    for year_dict in full_term_data:
        print(len(word_list))
        word_list.update(set(list(year_dict)))
    print(len(word_list))
    
    
    
    # fill in missing entries
    for cy in range(len(full_term_data)):
        for word in word_list:
            if word not in full_term_data[cy]:
                full_term_data[cy][word] = 0
            # for year_dict in full_term_data:
            # print(len(year_dict))
    
    # get word list mean and stdevs
    word_list_dict = {}
    for word in word_list:
        word_freqs = []
        for year_dict in full_term_data:
            word_freqs.append(year_dict[word])
        word_list_dict[word] = (np.mean(word_freqs), np.std(word_freqs))

    # calculate z-score for each word for each year
    zscore_dict = {}
    year_total = []
    for year_dict in full_term_data:
        total = 0
        for word in word_list:
            total += year_dict[word]
            if word == 'robespierre':
                print(word, year_dict[word], word_list_dict[word][0], word_list_dict[word][1])
            try:
                zscore_dict[word].append((year_dict[word]) / word_list_dict[word][0])# / word_list_dict[word][1])
            except:
                zscore_dict[word] = [(year_dict[word]) / word_list_dict[word][0]]# / word_list_dict[word][1]]
        year_total.append(total)
    # find biggest jumps in a given year
    top_n = 20
    top_in_year = [[0 for i in range(top_n+1)] for j in range(date_range[1] - date_range[0]+1)]
    top_words_year = [[None for i in range(top_n+1)] for j in range(date_range[1] - date_range[0]+1)]
    for year in range(date_range[0], date_range[1]-1):
        for word in word_list:
            if len(word) > 3:
                yc = year - date_range[0]
                if zscore_dict[word][yc] > 0.6:
                    zscore_diff = ((zscore_dict[word][yc + 1]) / (zscore_dict[word][yc]))#/word_list_dict[word][0]
                    top_in_year[yc][0] = zscore_diff
                    top_words_year[yc][0] = word
                    #print(zscore_diff)
                    sort_id = 0
                    #print(yc, sort_id, top_in_year)
                    # print(word)
                    while  top_in_year[yc][sort_id] > top_in_year[yc][sort_id+1]:
                        #print(top_in_year, sort_id)
    
                        tmp = top_in_year[yc][sort_id+1]
                        top_in_year[yc][sort_id+1] = top_in_year[yc][sort_id]
                        top_in_year[yc][sort_id] = tmp
                        
                        tmp_word = top_words_year[yc][sort_id+1]
                        # print(tmp_word, top_words_year, top_words_year[yc][sort_id+1])
                        top_words_year[yc][sort_id+1] = top_words_year[yc][sort_id]
                        top_words_year[yc][sort_id] = tmp_word
    
                        sort_id += 1
                        # print(sort_id, top_words_year[yc])
                        if sort_id == top_n:
                            break
                    # print(sort_id, word)
                    # print(word, sort_id)
                    # print(top_words_year[yc], top_in_year[yc])
                    #top_words_year[yc][sort_id] = word
                    # print(top_words_year[yc], top_in_year[yc])
                    # input()
        top_words_year[yc][0] = year+1
        print(year+1, top_words_year[yc], '\n')#, top_in_year[yc])
        
        # for word in top_words_year[yc]:
            # print(zscore_dict[word][yc+1] - zscore_dict[word][yc])
    #print(zscore_dict['robespierre'])
    #print(ireader)
    #terms = isearcher.terms() #Returns TermEnum
    #print(terms)
    return top_words_year

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




def main():
    #constants
    FIELD_CONTENTS = "text"
    DOC_NAME = "identifier"
    STORE_DIR = "../full_index"

    lucene.initVM()
    store = SimpleFSDirectory(Paths.get(STORE_DIR))    
    
    ireader = DirectoryReader.open(store)#, True)
    #print(ireader.readerIndex(0))

    searcher = IndexSearcher(ireader)#self.getSearcher()
                    
    pickle_file = glob.glob('full_word_list.pkl')
    print(pickle_file)
    date_range = (1785,1805)
    if not pickle_file:
        

        full_df = get_full_df()
        full_term_data = []
        for year in range(date_range[0], date_range[1]):
            docs_in_year = get_docs_in_year(full_df, year)
            #print(docs_in_year)
            year_dict = Counter({})
            terms = []
            freqs = []
            print(year)
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
                termvec = ireader.getTermVector(topDocs.scoreDocs[0].doc, "text")
                if termvec != None:
                    #termvec = reader.getTermVector(topDocs.scoreDocs[0].doc, "all")
    
                    termsEnum = termvec.iterator()
                    for term in BytesRefIterator.cast_(termsEnum):
                        terms.append(term.utf8ToString())
                        freqs.append(termsEnum.totalTermFreq())
                    #terms.sort()
            for term,freq in zip(terms, freqs):
                try:
                    year_dict[term] += freq
                except:
                    year_dict[term] = freq
            print(len(year_dict))
            for term in list(year_dict):
                if year_dict[term] < 5 and term not in stopwords:
                    year_dict.pop(term)
            full_term_data.append(year_dict)
            print(len(year_dict))
            #year_dict = year_dict + doc_dict
            #print(year_dict.most_common(1000))
            print('\n\n')
    
        pickle.dump(full_term_data, open('full_word_list.pkl', 'wb'))
    else:
        full_term_data = pickle.load(open('full_word_list.pkl', 'rb'))
        # get complete list of unique words
        # top_words_year = zscore_method(full_term_data, date_range)

        top_words_year = tfidf_method(full_term_data, date_range)
        print(top_words_year)
    pickle.dump(top_words_year, open('trending_ratio.pkl', 'wb'))

    

if __name__ == "__main__":
    main()
