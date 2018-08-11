from flask import render_template
from flask import make_response
from fp_site import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import pandas as pd
#import psycopg2
#from flask import request
import flask
from fp_site.site_sent_timeline import get_sent_dict
from fp_site.site_sent_timeline import gen_plot
from fp_site.sa_detailed_view import display_texts
import pickle
import base64

from sklearn.externals import joblib

import random
import urllib.parse
import datetime
from io import BytesIO

import matplotlib
matplotlib.use('Agg')    

import matplotlib.pyplot as plt

import itertools

#@app.route('/')
#def home_page():
#    return render_template("home.html")

# @app.route('/home')
# def slides():
#     return render_template("slides.html")

# @app.route('/about')
# def about():
#     return render_template("about.html")




@app.route('/', methods=['GET', 'POST'])
def home_page():
    sent_df = pickle.load(open('./pickles/3_df_sentiment.pkl', 'rb'))
    cols = list(sent_df)
    print(cols)
    print('.')
    term_cols = [term.split('_')[-1] for term in cols if term.find('sentiment_vals_unw') == 0]
    term_cols = sorted(term_cols)
    print(term_cols)
        
    return render_template("home.html", terms=term_cols)


# @app.route('/example', methods=['GET', 'POST'])
# def example_page():
#     pub_names = sorted(list(id_to_pub.values()))
#     return render_template("example.html", publications = pub_names)

@app.route('/term_details/<term>', methods=['GET', 'POST'])
def term_details(term):
    top_term_dict, bot_term_dict = display_texts(term)
    #term_dict = dict(itertools.islice(term_dict.items(), 10))
    #print(term_dict[list(term_dict)[0]])
    return render_template('term_details.html', term=term, top_term_dict=top_term_dict, bot_term_dict=bot_term_dict)



@app.route('/output', methods = ['GET', 'POST'])
def output():
    sent_df = pickle.load(open('./pickles/3_df_sentiment.pkl', 'rb'))
    cols = list(sent_df)
    print(cols)
    print('.')
    term_cols = [term.split('_')[-1] for term in cols if term.find('sentiment_vals_unw') == 0]
    print(term_cols)
    term_cols = sorted(term_cols)
    checked_terms = flask.request.form.getlist('terms')
    # get list of terms to plot
    #terms = flask.request.values.get('terms')
    print('terms:', checked_terms)
    weight_flag = True
    date_range = (1786,1800)
    
    checked_term_dict = get_sent_dict(sent_df, checked_terms, weight_flag, date_range)

    
    gen_plot(checked_term_dict, checked_terms, date_range)
    
    figfile_target = BytesIO()
    plt.savefig(figfile_target, format='png')
    figfile_target.seek(0)  # rewind to beginning of file
    figdata_png_target = figfile_target.getvalue()


    figdata_png_target = base64.b64encode(figdata_png_target)

    return render_template("output.html", terms=term_cols, img_data_target=urllib.parse.quote(figdata_png_target))
