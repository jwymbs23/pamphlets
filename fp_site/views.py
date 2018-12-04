import flask
from flask import render_template, make_response, Response, redirect, url_for, session


from fp_site import app
from fp_site.site_sent_timeline import get_sent_dict
from fp_site.site_sent_timeline import gen_plot
from fp_site.sa_detailed_view import display_texts
from fp_site.new_term import add_new_term
from fp_site.new_term import new_term_single
from fp_site.new_term import define_search_params
from fp_site.new_term import get_doc_list
#import spellcheck.spellcheck_internal as sp_ch
import sentiment_analysis.analyze_sa as a_sa



import pickle
import base64
import time
import random
import datetime
import itertools
import glob
import numpy as np


import urllib.parse
from io import BytesIO

import matplotlib
matplotlib.use('Agg')    
import matplotlib.pyplot as plt

import lucene
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




@app.before_first_request
def load_index():
    global vm, searcher, reader
    vm = lucene.initVM()

    FIELD_CONTENTS = "text"
    DOC_NAME = "identifier"
    STORE_DIR = "./full_index1"            
    store = SimpleFSDirectory(Paths.get(STORE_DIR))
    reader = DirectoryReader.open(store)
    searcher = IndexSearcher(reader)

            

@app.route('/', methods=['GET', 'POST'])
def home_page():
    sent_df = pickle.load(open('./pickles/3_df_sentiment.pkl', 'rb'))
    #cols = list(sent_df)
    
    #term_cols = [term.split('_')[-1] for term in cols if term.find('sentiment_vals_unw') == 0]
    #term_cols = sorted(term_cols)

    pickle_list = glob.glob('./pickles/*df.pkl')
    term_cols = [term[10:-7] for term in pickle_list]
    term_cols = sorted(term_cols)
    #print(term_cols)
        
    return render_template("home.html", terms=term_cols)


@app.route('/insights')
def insights():
    return render_template("insights.html")

@app.route('/methods')
def methods():
    return render_template("methods.html")

@app.route('/term_details/<year>/<term>', methods=['GET', 'POST'])
def term_details(term, year='overall'):
    top_term_dict, bot_term_dict = display_texts(term)
    #print(list(top_term_dict))
    #print(list(bot_term_dict))
    #print(year)
    #term_dict = dict(itertools.islice(term_dict.items(), 10))
    #print(term_dict[list(term_dict)[0]])
    doc_years = sorted(list(bot_term_dict))
    years = range(1785,1801)
    n_years = len(years)
    return render_template('term_details.html', doc_years=doc_years, years=years, year=year, term=term, top_term_dict=top_term_dict[year], bot_term_dict=bot_term_dict[year], n_years=n_years)


@app.route('/trending', methods = ['GET', 'POST'])
def trending_page():
    trending_data = pickle.load(open('./pickles/trending_ratio.pkl', 'rb'))
    #print(trending_data)
    years = [i[0] for i in trending_data]
    trending_terms = [i[1:] for i in trending_data]
    n_years = len(years)-1
    return render_template('trending.html', years=years, n_years=n_years, trending_terms=trending_terms)


@app.route('/trending_year/<year>', methods=['GET', 'POST'])
def trending_year(year):
    trending_data = pickle.load(open('./pickles/trending_ratio.pkl', 'rb'))
    blurb = open('./fp_site/timeline_blurbs/'+year+'.txt').readlines()[0].strip()
    years = [i[0] for i in trending_data]
    n_years = len(years)-1
    return render_template('trending_year.html', year=year, years=years, n_years=n_years, blurb=blurb)



@app.route('/output', methods = ['GET', 'POST'])
def output():
    sent_df = pickle.load(open('./pickles/3_df_sentiment.pkl', 'rb'))
    #cols = list(sent_df)
    #print(cols)
    #print('.')
    #term_cols = [term.split('_')[-1] for term in cols if term.find('sentiment_vals_unw') == 0]
    #print(term_cols)
    #term_cols = sorted(term_cols)

    pickle_list = glob.glob('./pickles/*df.pkl')
    term_cols = [term[10:-7] for term in pickle_list]
    term_cols = sorted(term_cols)

    checked_terms = flask.request.form.getlist('terms')
    # get list of terms to plot
    #terms = flask.request.values.get('terms')
    #print('terms:', checked_terms)
    weight_flag = True
    date_range = (1786,1801)
    
    checked_term_dict = get_sent_dict(sent_df, checked_terms, weight_flag, date_range)

    
    gen_plot(checked_term_dict, checked_terms, date_range)
    
    figfile_target = BytesIO()
    plt.savefig(figfile_target, format='png')
    figfile_target.seek(0)  # rewind to beginning of file
    figdata_png_target = figfile_target.getvalue()


    figdata_png_target = base64.b64encode(figdata_png_target)

    return render_template("output.html", terms=term_cols, checked_terms=checked_terms,
                           img_data_target=urllib.parse.quote(figdata_png_target))


@app.route('/new_term', methods=['GET', 'POST'])
def new_term():
    return render_template("new_term.html")



@app.route('/adding_term/<term>', methods=['GET', 'POST'])
def get_page(term):
    #print('getpage', term)
    return render_template('progress.html', term=term)


#@app.route('/progress', methods=['GET', 'POST'])
#def progress():
#    term = "antoinette"
#    SA_dict = pickle.load(open('./pickles/3_sentiment_dictionary_stem_FEEL.pkl', 'rb'))    
#    vm.attachCurrentThread()
#    doc_list = get_doc_list(term, searcher, reader)
#    sa_term = []
#    def generate(SA_dict, doc_list, term):
#        for ch, hit in enumerate(doc_list.scoreDocs):
#            x = ch/len(doc_list.scoreDocs)*100
#            sa_term.append(new_term_single(SA_dict, term, hit, searcher=searcher, reader=reader))
#            #x = 0
#            #while x < 100:
#            #print(x)
#            sse_event = 'import-progress'
#            if ch == len(doc_list.scoreDocs)-1:
#                sse_event = 'last-item'
#                with app.test_request_context():
#                    x = url_for('.new_term')
#            print(sse_event)    
#            #yield "id:{_id}\nevent:{event}\ndata:{data}\n\n".format(
#            #    _id=x, event=sse_event, data=str(x))
#            yield "data:" + str(x) + "\n" + "event:" + sse_event + "\n\n"
#    return Response(generate(SA_dict, doc_list, term), mimetype= 'text/event-stream')


@app.route('/high_word_count')
def high_word_count():
    return render_template('high_frequency_word.html')

@app.route('/low_word_count')
def low_word_count():
    return render_template('low_frequency_word.html')



@app.route('/add_new_term', methods=['GET', 'POST'])
def add_new_term_page():
    term = flask.request.form['term']
    session['term'] = term
    pickle_list = glob.glob('./pickles/*df.pkl')
    #print(pickle_list, './pickles/'+term+'_df.pkl' in pickle_list)
    if './pickles/'+term+'_df.pkl' in pickle_list:
        return redirect("new_term")
    elif len(term.split()) > 1:
        return redirect("new_term")
    else:
        vm.attachCurrentThread()
        # make sure term doesn't appear in too many documents
        doc_list = get_doc_list(term, searcher, reader)
        n_docs = len(doc_list.scoreDocs)
        #print(n_docs)
        if n_docs > 15000:
            return render_template("high_frequency_word.html")
        elif n_docs < 10:
            return render_template("high_frequency_word.html")
        else:
            return render_template("progress.html", term=term)

@app.route('/progress', methods=['GET','POST'])
def progress():
    #term='chevaliers'
    term = session['term']
    #with open('./pickles/term_data.txt', 'r') as f:
    #    term = f.readlines()[0]
    #term = flask.request.args.get('term')
    def generate(SA_dict, doc_list, term):
        sa_term = []
        term_words = {}
        for ch, hit in enumerate(doc_list.scoreDocs):
            x = ch/len(doc_list.scoreDocs)*100
            doc_name, doc_words, sa_data = new_term_single(SA_dict, term, hit, searcher=searcher, reader=reader)
            sa_term.append(sa_data)
            term_words[doc_name] = doc_words    
            #x = 0
            #while x < 100:
            #print(x)
            sse_event = 'import-progress'
            if ch == len(doc_list.scoreDocs)-1:
                #write sa_term
                sa_df = a_sa.make_sa_df(sa_term,term)
                if len(sa_df) > 10:
                    pickle.dump(sa_df, open('./pickles/%s_df.pkl'%term, 'wb'))
                    pickle.dump(term_words, open('./pickles/%s_words.pkl'%term, 'wb'))                                        
                #end conditions for js
                sse_event = 'last-item'
                with app.test_request_context():
                    x = url_for('.new_term')
            #print(sse_event)
            #yield "id:{_id}\nevent:{event}\ndata:{data}\n\n".format(
            #    _id=x, event=sse_event, data=str(x))
            yield "data:" + str(x) + "\n" + "event:" + sse_event + "\n\n"
    ####
    #print(term)

    #print(term)
    vm.attachCurrentThread()
    # make sure term doesn't appear in too many documents
    doc_list = get_doc_list(term, searcher, reader)
    n_docs = len(doc_list.scoreDocs)
    #print(n_docs)
    #if n_docs > 15000:
    #    with app.test_request_context():
    #        data = url_for('.high_word_count')
    #    return Response("data:"+data+"\nevent:too-many\n\n", mimetype= 'text/event-stream')#render_template("high_frequency_word.html")
    #elif n_docs < 10:
    #    with app.test_request_context():
    #        data = url_for('.low_word_count')
    #    return Response("data:"+data+"\nevent:too-few\n\n", mimetype= 'text/event-stream')#render_template("high_frequency_word.html")
    #
    #else:
    SA_dict = pickle.load(open('./pickles/3_sentiment_dictionary_stem_FEEL.pkl', 'rb'))

    #full_dict, modern_dict, map_chars, charlist = sp_ch.load_clean_word_list()
    
    
    
    ### replacement table
    #rep_data = pickle.load(open('./spellcheck/rep_table.pkl', 'rb'))
    #print(rep_data)
    #rep_table = rep_data['rep_table']
    #charlist = rep_data['charlist']
    #try:
    #    map_chars = rep_data['charmap']
    #except:
    #    map_chars = rep_data['map_chars']
    ####
    #top_n = 4
    #top_replacements = {}
    #for cf, from_letter in enumerate(rep_table):
    #    sort_idx = np.argsort(from_letter)[::-1]
    #    #print(from_letter)
    #    top_rep = [sort_idx[i] for i in range(top_n)]
    #    #print(top_rep)
    #    top_replacements[charlist[cf]] = [charlist[char] for char in top_rep]
                                                                                                            
    
    
    
    return Response(generate(SA_dict, doc_list, term), mimetype= 'text/event-stream')                
            
    #return render_template("add_new_term.html", term=term)
