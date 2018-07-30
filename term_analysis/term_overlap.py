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

DOC_NAME = "identifier"
STORE_DIR = "./full_index"
FIELD_CONTENTS = "text"

term_list = ['robespierre', 'danton', 'xvi']#, 'marat', 'mirabeau', 'antoinette', 'fayette', 'tyran']#, 'égalité'.decode('utf-8'), 'fraternité'.decode('utf-8'), 'révolution'.decode('utf-8'), 'salut', 'necker', 'napoleon', 'monarchie', 'aristocratie', 'hébert'.decode('utf-8'), 'gironde', 'jacobins', 'feuillants', 'royalistes','royaliste', 'guillotine', 'bastille', 'versailles', 'tuilleries', 'paume', 'constitution', 'etats', 'citoyen', 'democratie']

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
        docs.append(doc.get(DOC_NAME))
    appearance_dict[TERM] = set(docs)



"""
var sets = [{sets : [0], label : 'SE', size : 28,},
            {sets : [1], label : 'Treat', size: 35},
            {sets : [2], label : 'Anti-CCP', size : 108},
            {sets : [3], label : 'DAS28', size:106},
            {sets : [0,1], size:1},
            {sets : [0,2], size:1},
            {sets : [0,3], size:14},
            {sets : [1,2], size:6},
            {sets : [1,3], size:0},
            {sets : [2,3], size:1},
            {sets : [0,2,3], size:1},
            {sets : [0,1,2], size:0},
            {sets : [0,1,3], size:0},
            {sets : [1,2,3], size:0},
            {sets : [0,1,2,3], size:0}
];
"""


def make_venn_jsonp():
    jsonp_out = open('overlap.jsonp', 'w')
    jsonp_out.write('var sets = [')
    stuff = list(appearance_dict)
    for L in range(0, len(stuff)+1):
        for subset in itertools.combinations(stuff, L):
            if len(subset) != 0:
                print(subset)
                jsonp_out.write('{sets : [')
                if len(subset) == 1:
                    jsonp_out.write(str(stuff.index(subset[0])))
                    jsonp_out.write('], label : \'' + subset[0] + '\', size : ' + str(len(appearance_dict[subset[0]])) + '},\n')
                elif len(subset) > 1:
                    for term in subset[:-1]:
                        jsonp_out.write(str(stuff.index(term)) + ',')
                    intersect = set(appearance_dict[subset[0]])
                    for term in subset[1:]:
                        intersect = appearance_dict[term].intersection(intersect)
                    jsonp_out.write(str(stuff.index(subset[-1])) + '], size: ' + str(len(intersect)) + '},\n')
    jsonp_out.write('];')
            #for c1, word1 in enumerate(appearance_dict):
    #    jsonp_out.write('{sets : [' + str(c1) + '], label : \'' + word1 + '\', size : ' + str(len(appearance_dict[word1])) + '},\n')
    #    for c2, word2 in enumerate(appearance_dict):
    #        if c2 > c1:
    #            jsonp_out.write('{sets : [' + str(c1) +',' + str(c2) + '], size : ' + str(len( appearance_dict[word1].intersection(appearance_dict[word2])  )) + '},\n')
    #
    #jsonp_out.write('];')

make_venn_jsonp()



def overlap_matrix():
    i=0
    n_words = len(appearance_dict)
    overlap_matrix = np.zeros((n_words,n_words))
    ind_1 = 0
    ind_2 = 0
    dict_term_list = []
    for word1 in appearance_dict:
        dict_term_list.append(word1)
        print(word1)
        for word2 in appearance_dict:
            #for a in appearance_dict[word1]:
            #    if a in appearance_dict[word2]:
            overlap_matrix[ind_1][ind_2] = float(len(appearance_dict[word1].intersection(appearance_dict[word2]))) / float(len(appearance_dict[word1]))#overlap_matrix[ind_1][ind_2] + 1/float(len(appearance_dict[word1]))
            ind_2 = ind_2 + 1
        ind_2 = 0
        ind_1 = ind_1 + 1
    
    print(overlap_matrix)
    
    fig,ax = plt.subplots(1,1)
    
    
    ax.imshow(overlap_matrix, interpolation='nearest')
    #ax.grid(False)
    ax.set_yticks(list(range(len(dict_term_list))))
    ax.set_xticks(list(range(len(dict_term_list))))
    
    plt.xticks(rotation=90)
    ax.set_xticklabels(dict_term_list)
    #ax.xaxis.set_tick_params(rotation=90)
    ax.set_yticklabels(dict_term_list)
    #fig.colorbar(overlap_matrix, cax=cbar_ax)
    #plt.margins(0.2)
    plt.subplots_adjust(bottom=0.3)
    #plt.savefig('overlap_matrix.jpg')
    plt.show()
    
    
    
    g = sns.clustermap(overlap_matrix)
    reordered_ind = g.dendrogram_row.reordered_ind
    new_term_indices = [dict_term_list[i] for i in g.dendrogram_row.reordered_ind]
    #ax1.set_xticklabels(new_region_indices,fontsize=6,rotation=90)
    #ax1.set_yticklabels(new_region_indices,fontsize=6)
    plt.show()
    
    
    reorder_overlap = np.zeros((len(term_list),len(term_list)))
    for i in range(overlap_matrix.shape[0]):
        for j in range(overlap_matrix.shape[0]):
            reorder_overlap[i][j] = overlap_matrix[reordered_ind[i]][reordered_ind[j]]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.imshow(reorder_overlap,interpolation='none')
    ax1.set_xticks(range(len(term_list)))
    ax1.set_yticks(range(len(term_list)))
    ax1.set_xticklabels(new_term_indices,fontsize=12,rotation=90)
    ax1.set_yticklabels(new_term_indices,fontsize=12)
    plt.show()
