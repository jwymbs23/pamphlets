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
                topDocs = searcher.search(q, 50)
                
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
                if year_dict[term] < 5:
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
        word_list = set()
        for year_dict in full_term_data:
            print(len(word_list))
            word_list.update(set(list(year_dict)))
        print(len(word_list))
    
    
        
        # fill in missing entries
        for cy in range(len(full_term_data)):
            for word in word_list:
                if word not in full_term_data[cy]:
                    full_term_data[cy][word] = 1
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
        for year_dict in full_term_data:
            for word in word_list:
                if word == 'robespierre':
                    print(word, year_dict[word], word_list_dict[word][0], word_list_dict[word][1])
                try:
                    zscore_dict[word].append((year_dict[word]))# / word_list_dict[word][0]))
                except:
                    zscore_dict[word] = [(year_dict[word])]# / word_list_dict[word][0])]
    
        # find biggest jumps in a given year
        top_n = 20
        top_in_year = [[0 for i in range(top_n+1)] for j in range(date_range[1] - date_range[0]+1)]
        top_words_year = [[None for i in range(top_n+1)] for j in range(date_range[1] - date_range[0]+1)]
        for year in range(date_range[0], date_range[1]-1):
            for word in word_list:
                yc = year - date_range[0]
                if zscore_dict[word][yc+1] > 500:
                    zscore_diff = zscore_dict[word][yc + 1] / zscore_dict[word][yc]
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
            print(year+1, top_words_year[yc], '\n')#, top_in_year[yc])
            # for word in top_words_year[yc]:
                # print(zscore_dict[word][yc+1] - zscore_dict[word][yc])
        #print(zscore_dict['robespierre'])
        #print(ireader)
        #terms = isearcher.terms() #Returns TermEnum
        #print(terms)
    exit(0)
    
    #take search term as command line argument
    if len(sys.argv) != 4:
        print('Format should be: python search_docs.py, [term to search for], redo? y/n, window_size')
        exit(0)
    
    #parse user input
    TERM = sys.argv[1]
    remake_df = True if sys.argv[2] == 'y' else False
    window_size = int(sys.argv[3])

    #other options
    stem_flag = True
    spell_check_flag = False

    #get dataframe
    doc_data = get_doc_df(remake_df)

    #get dictionary
    SA_dict = get_dict(stem_flag)

    print('Searching for: "'+TERM+'"')
    
    sa_term = []
    
    date_range = (1791, 1800)
    method = 'linear' #vs 1/x

    example_flag = False

    if not 'sentiment_vals_w_'+TERM in list(doc_data):
        lucene.initVM()
        searcher, reader, query = define_search_params(STORE_DIR, FIELD_CONTENTS, TERM)
            # Run the query and get documents that contain the term
        docs_containing_term = searcher.search(query, reader.numDocs())
        
        
        print( 'Found '+str(len(docs_containing_term.scoreDocs))+' documents with the term "'+TERM+'".')
        print( 'Calculating sentiment scores...')
        term_words = []
        #hits = searcher.search(query, 1)
        for hit in docs_containing_term.scoreDocs:
            print(hit)
            doc = searcher.doc(hit.doc)
            #get the text from each document
            doc_text = doc.get("text")#.encode("utf-8")
            print(doc.get(DOC_NAME))
            #single doc returns the score data for a single document, and a list of words that appear in the term windows for that document
            score_data, doc_words = sa.single_doc(TERM,doc_text,SA_dict, window_size, spell_check_flag, example_flag, stem_flag, method)
            #print(score_data)
            term_words.append((doc.get(DOC_NAME).split('/')[-1], doc_words))
            sa_doc_score = [doc.get(DOC_NAME)] + score_data
            sa_term.append(sa_doc_score)
        sa_df = a_sa.make_sa_df(doc_data, sa_term,TERM)
        pickle.dump(term_words, open('./pickles/%s_words.pkl'%TERM, 'wb'))
    else:
        sa_df = doc_data
    
    print(sa_df)
    

        
    #process dataframe for various properties (split this into specific functions later)
    use_weighted = True
    total_doc = False
    a_sa.plot_term_score_data(sa_df,TERM,use_weighted, date_range)
    

if __name__ == "__main__":
    main()
