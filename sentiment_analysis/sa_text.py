# from Tkinter import Tk, Text
import string
import re

import text_cleaning.norvig_spellcheck as spell_check
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("french")

"""
for all the files that each term appears in, locate the actual occurences of
the terms scan through words $wdist$ before and after the term appears, and
assign SA score to words weighted by their distance from the term weight by
distance because OCR text is not good enough (yet) to reliably parse full
sentences. The thinking behind this is that the closer a word is to the search
term, the more likely it is that they are connected, but I also want to account
for the general tone of the text where the search term appears.
"""


def score_func(score, dist, wdist, max_score, method):
    """
    Input:  (float) sentiment score of word
            (int)   distance between word and term
            (int)   window size
            (int)   maximum score
            (str)   how to weight distance between word and term
    Output: (float) weighted score
    Calculates a weighted word sentiment score based on it's distance from a
    target term for a given set of weighting parameters.
    """
    if method == 'inverse':
        return (float(score)/abs(dist) - max_score/wdist) * (wdist)/(wdist-1.0)
    elif method == 'linear':
        return (float(score))*(-1.0/float(wdist) * float(abs(dist)) + 1.0)
    else:
        print('error: invalid method')
        exit(0)


def find_term_indices(TERM, text_split_no_punc):
    term_indices = []
    if ' ' in TERM:
        for ci, i in enumerate(text_split_no_punc[:-1]):
            if ' '.join([i, text_split_no_punc[ci+1]]) == TERM:
                term_indices.append(ci)
    else:
        # loop through entire document
        for ci, i in enumerate(text_split_no_punc):
            # locate occurcences of TERM in the document
            if i == TERM:
                term_indices.append(ci)
    return term_indices


def window_size(term_loc, wdist, text_length):
    l_range = term_loc - wdist if term_loc - wdist > 0 else 0
    r_range = term_loc + wdist + 1 if term_loc + wdist < \
        text_length else text_length
    return l_range, r_range


def single_doc(TERM, text, SA_dict, wdist, spell_check_flag,
               example_flag, stem_flag, method):

    weighted_hist_temp = []

    # remove punctuation from text and split on spaces
    text_split_no_punc = re.sub(
        '['+string.punctuation+']', '', str(text)).split()
    # do this for python2 text.translate(None, string.punctuation).split()
    # do this for python3 text.translate(punctuation).split()

    # location of TERM in document
    # limited to search terms with at most one space character at this point
    term_indices = find_term_indices(TERM, text_split_no_punc)

    # for calculating weighted and unweighted scores for all relevant
    # passages in a document
    av_emotion_doc = 0
    weighted_av_emotion_doc = 0
    w_av = []
    # make sure that the term is actually in the document
    doc_words = []

    text_length = len(text_split_no_punc)

    if len(term_indices) > 0:
        for term_loc in term_indices:
            emotion_words = []
            passage_words = []
            # generate window range to search through
            # with hard stops at the beginning and end of a document
            l_range, r_range = window_size(term_loc, wdist, text_length)

            # for calculating single passage scores
            av_emotion_single = 0
            weighted_av_emotion_single = 0
            # scan through words in window
            for word_loc in range(l_range, r_range):
                # distance between word and TERM
                dist = word_loc - term_loc
                # perform norvig spell checking (one edit)
                word = text_split_no_punc[word_loc]
                if spell_check_flag:
                    word = spell_check.correction(word)[0]
                passage_words.append(word)
                if stem_flag:
                    word = stemmer.stem(word)

                # single word score, if word is not in dictionary,
                # say that it has a score of 0
                score = 0
                if dist != 0:
                    # see if word is in sentiment dictionary
                    if word in SA_dict:
                        score = SA_dict[word]
                        # add score to unweighted sentiment counter
                        av_emotion_single += score
                        # add weighted score to weighted sentiment counter
                        weighted_av_emotion_single += score_func(
                            score, dist, wdist, 1, method)

            # add passage scores to document score
            av_emotion_doc += av_emotion_single
            weighted_av_emotion_doc += weighted_av_emotion_single/wdist
            # keep track of individual passage score
            # (can be used to get sentiment variation within texts)
            w_av.append(weighted_av_emotion_single)
            doc_words.append([passage_words])
        # normalize document scores by the number of TERM appearances
        av_emotion_doc /= (float(len(term_indices)))
        weighted_av_emotion_doc /= float(len(term_indices))
        # aggregate data into single item
        weighted_hist_temp = [TERM, weighted_av_emotion_doc, w_av,
                              av_emotion_doc]
        # print(weighted_hist_temp)
        # exit(0)
        
        # doc_words.append({'wscore': weighted_av_emotion_doc, 'unwscore': av_emotion_doc})
    return weighted_hist_temp, doc_words
