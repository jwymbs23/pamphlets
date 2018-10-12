import lucene
import numpy as np
import string

from org.apache.lucene.analysis.standard import StandardAnalyzer

#from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths

from org.apache.lucene.index import \
        IndexWriter, IndexWriterConfig, DirectoryReader, IndexReader, MultiFields
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory



from java.io import File

from java.io import StringReader
from org.apache.lucene.analysis.core import WhitespaceAnalyzer

from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import \
        MultiPhraseQuery, PhraseQuery, DocIdSetIterator
from org.apache.lucene.search import IndexSearcher

import sys
import sentiment_analysis.sa_text as sa
import sentiment_analysis.analyze_sa as a_sa
from tqdm import tqdm
import pickle
from text_cleaning.stopwords import *
import glob



def define_search_params(STORE_DIR, FIELD_CONTENTS, TERM):
    
    #indexPath = File(STORE_DIR).toPath()
    #indexDir = FSDirectory.open(indexPath)
    
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

def get_doc_df(remake_df = True):
    if remake_df == True:
        return pickle.load(open('./pickles/3_df_relevant.pkl','rb'))
    else:
        return pickle.load(open('./pickles/3_df_sentiment.pkl','rb'))
 

def add_new_term(TERM, remake_df=True, window_size=30):
    #constants
    FIELD_CONTENTS = "text"
    DOC_NAME = "identifier"
    STORE_DIR = "./full_index1"

    #other options
    stem_flag = True
    spell_check_flag = False

    #get dataframe
    doc_data = get_doc_df(remake_df)

    #get dictionary
    SA_dict = get_dict(stem_flag)

    # print('Searching for: "'+TERM+'"')
    
    sa_term = []
    
    date_range = (1791, 1800)
    method = 'linear' #vs 1/x

    example_flag = False

    # if not 'sentiment_vals_w_'+TERM in list(doc_data):
    if not glob.glob('./pickles/%s_df.pkl'%TERM):
        lucene.initVM()
        searcher, reader, query = define_search_params(STORE_DIR, FIELD_CONTENTS, TERM)

        # fieldInfos = MultiFields.getMergedFieldInfos(reader)
        # print(fieldInfos)
        # for fieldInfo in fieldInfos.iterator():
            # print(fieldInfo.name)
        # Run the query and get documents that contain the term
        docs_containing_term = searcher.search(query, reader.numDocs())
        
        
        # print( 'Found '+str(len(docs_containing_term.scoreDocs))+' documents with the term "'+TERM+'".')
        # print( 'Calculating sentiment scores...')
        term_words = {}
        #hits = searcher.search(query, 1)
        for hit in tqdm(docs_containing_term.scoreDocs):

            doc = searcher.doc(hit.doc)
            
            #get the text from each document
            doc_text = doc.get("text")#doc.get("text")#.encode("utf-8")
            #single doc returns the score data for a single document, and a list of words that appear in the term windows for that document
            score_data, doc_words = sa.single_doc(TERM,doc_text,SA_dict, window_size, spell_check_flag, example_flag, stem_flag, method)
            #print(score_data)
            term_words[doc.get(DOC_NAME).split('/')[-1]] = doc_words
            sa_doc_score = [doc.get(DOC_NAME)] + score_data
            sa_term.append(sa_doc_score)
        sa_df = a_sa.make_sa_df(doc_data, sa_term,TERM)
        pickle.dump(sa_df, open('./pickles/%s_df.pkl'%TERM, 'wb'))
        pickle.dump(term_words, open('./pickles/%s_words.pkl'%TERM, 'wb'))
        # sa_df = doc_data
        
    # print(sa_df)
            
    # process dataframe for various properties (split this into specific functions later)
    # use_weighted = True
    # total_doc = False
    # a_sa.plot_term_score_data(sa_df,TERM,use_weighted, date_range)
    

