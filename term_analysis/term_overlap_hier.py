# -*- coding: utf-8 -*-
from lucene import \
            QueryParser, IndexSearcher, IndexReader, StandardAnalyzer, \
        TermPositionVector, SimpleFSDirectory, File, MoreLikeThis, \
            VERSION, initVM, Version
import sys
#from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os

DOC_NAME = "identifier"
STORE_DIR = "./full_index"
FIELD_CONTENTS = "text"

term_list = ['robespierre', 'danton', 'marat', 'mirabeau', 'fayette']#, 'xvi', 'antoinette', 'tyran']#, 'égalité'.decode('utf-8'), 'fraternité'.decode('utf-8'), 'révolution'.decode('utf-8'), 'salut', 'necker', 'napoleon', 'monarchie', 'aristocratie', 'hébert'.decode('utf-8'), 'gironde', 'jacobins', 'feuillants', 'royalistes','royaliste', 'guillotine', 'bastille', 'versailles', 'tuilleries', 'paume', 'constitution', 'etats', 'citoyen', 'democratie']

initVM()
# Get handle to index directory
directory = SimpleFSDirectory(File(STORE_DIR))
# Creates a searcher searching the provided index.
ireader  = IndexReader.open(directory, True)
# Implements search over a single IndexReader.
# Use a single instance and use it across queries
# to improve performance.
searcher = IndexSearcher(ireader)
# Get the analyzer
analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
# Constructs a query parser. We specify what field to search into.
queryParser = QueryParser(Version.LUCENE_CURRENT,FIELD_CONTENTS, analyzer)


appearance_dict = {}
for TERM in term_list:
    print 'Searching for: "'+TERM+'"'
    # Create the query
    query = queryParser.parse(TERM)
    
    # Run the query and get documents that contain the term
    docs_containing_term = searcher.search(query, ireader.numDocs())

    docs = []
    
    print 'Found '+str(len(docs_containing_term.scoreDocs))+' documents with the term "'+TERM+'".'
    #hits = searcher.search(query, 1)
    for hit in (docs_containing_term.scoreDocs):
        #print(hit.score, hit.doc, hit.toString())
        doc = searcher.doc(hit.doc)
        docid = doc.get(DOC_NAME)[11:-9]
        if docid in appearance_dict:
            appearance_dict[docid].append(TERM)#docs.append(doc.get(DOC_NAME))
        else:
            appearance_dict[docid] = [TERM]
#    appearance_dict[TERM] = set(docs)



"""
[
{ "name": "Fayette.doc1",
  "imports": ["Marat.doc1"] }
, { "name": "Fayette.doc2",
  "imports": [] }
, {"name": "Fayette.doc8",
  "imports": []}
]

"""


def make_venn_json():
    json_out = open('hier_overlap.json', 'w')
    json_out.write('[\n')
    first = True
    num_dict = {1: 'aaa', 2:'bbb', 3:'ccc',4:'ddd', 5:'eee', 6:'fff', 7:'ggg', 8:'hhh', 9:'iii', 10:'jjj'}
    for doc in appearance_dict:
        #if not first:
        #    json_out.write(', {\n')
        #first = False
        #print(doc)
        shared_terms = appearance_dict[doc]
        num_shared = len(shared_terms)
        if len(shared_terms) == 1:
            json_out.write('{\n  "name": "'+shared_terms[0]+'.'+num_dict[num_shared] + '.'+doc+'",\n  "imports": [] \n} ')
            json_out.write(',')
        else:
            for source in appearance_dict[doc]:
                json_out.write('{\n "name": "'+source+'.'+num_dict[num_shared] + '.'+doc+'",\n  "imports": [')
                count_tar = 0
                for target in appearance_dict[doc]:
                    if source != target:
                        count_tar += 1
                        if count_tar < num_shared:
                            json_out.write('"' + target+'.'+num_dict[num_shared] + '.'+doc + '"')
                        if count_tar == num_shared-1:
                            json_out.write('] \n} ,')
                        else:
                            json_out.write(',')
                        
            #for comb in itertools.permutations(appearance_dict[doc], 2):
                #print comb
            #    json_out.write('{\n "name": "'+comb[0]+'.'+num_dict[num_shared] + '.'+doc+'",\n  "imports": ["'+comb[1]+'.'+num_dict[num_shared] + '.'+doc+'"] \n} ')
            #    json_out.write(',')
            #print(appearance_dict[doc])
    json_out.seek(-1, os.SEEK_END)
    json_out.truncate()
    json_out.write(']')
        #exit(0)
        #for term in appearance_dict[doc]:
        #    json_out.write('  "name": "'+term+'.'+doc+',\n  "imports": ["'+term+'.'+doc+'"] \n} ')
    
make_venn_json()
