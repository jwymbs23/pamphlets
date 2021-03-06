import pickle
import string
from Tkinter import Tk, Text
import numpy as np
import norvig_spellcheck as spell_check


#Sentiment dictionary:
SA_dict = pickle.load(open('./sentiment_dictionary_FEEL.pkl', 'rb'))

#need this for python3
#punctuation = str.maketrans({key: None for key in string.punctuation})

#for all the files that each term appears in, locate the actual occurences of the terms
#scan through words $wdist$ before and after the term appears, and assign SA score to words weighted by their distance from the term
#weight by distance because OCR text is not good enough (yet) to reliably parse full sentences.
#thinking behind this is that the closer a word is to the search term, the more likely it is that they are connected, but I also want to account
#for the general tone of the text where the search term appears.


def score_func(score, dist, wdist, max_score):
    #1/x
    #return (float(score)/abs(dist) - max_score/(wdist)) * (wdist)/((wdist)-1.0)
    #1-x
    return (float(score))*(-1.0/float(wdist) * float(abs(dist)) + 1.0)


def find_term_indices(TERM, text_split_no_punc):
    total_doc_sentiment = 0
    term_indices = []
    if ' ' in TERM:
        #term_indices = [ci for ci, i in enumerate(text_split_no_punc[:-1]) if ' '.join([i, text_split_no_punc[ci+1]]) == TERM]
        for ci, i in enumerate(text_split_no_punc[:-1]):
            if ' '.join([unicode(i, 'utf-8'), unicode(text_split_no_punc[ci+1], 'utf-8')]) == unicode(TERM, 'utf-8'):
                term_indices.append(ci)
    else:
        #loop through entire document
        for ci, i in enumerate(text_split_no_punc):
            #calculate entire document score
            #if i.decode('utf-8') in SA_dict:
            #    total_doc_sentiment += SA_dict[i.decode('utf-8')]
            #locate occurcences of TERM in the document
            if unicode(i, 'utf-8') == TERM:#unicode(TERM, 'utf-8'):
                term_indices.append(ci)
    #normalize doc score
    total_doc_sentiment /= float(len(text_split_no_punc))
    return term_indices, total_doc_sentiment



def single_doc(TERM,text, spell_check_flag, wdist, ex_flag):
    if ex_flag == True:
        root = Tk()
        displaytext = Text(root)
        displaytext.pack()
        wtag = 0
    weighted_hist_temp = []

    #remove punctuation from text and split on spaces
    text_split_no_punc = text.translate(None, string.punctuation).split()
    #do this for python3 text.translate(punctuation).split()

    #location of TERM in document    
    #limited to search terms with at most one space character at this point
    term_indices, total_doc_sentiment = find_term_indices(TERM, text_split_no_punc)

    #for calculating weighted and unweighted scores for all relevant passages in a document
    av_emotion_doc = 0
    weighted_av_emotion_doc = 0
    w_av = []
    #make sure that the term is actually in the document
    if len(term_indices) > 0:
        for term_loc in term_indices:
            emotion_words = []
            #generate window range to search through (with hard stops at the beginning and end of a document) [clip?]
            l_range = term_loc - wdist if term_loc - wdist > 0 else 0
            r_range = term_loc + wdist + 1 if term_loc + wdist < len(text_split_no_punc) else len(text_split_no_punc)
            #for calculating single passage scores
            av_emotion_single = 0
            weighted_av_emotion_single = 0
            #loop over window
            for word_loc in range(l_range, r_range):
                #distance between word and TERM
                dist = word_loc - term_loc
                #perform norvig spell checking (one edit)
                if spell_check_flag == True:
                    word = spell_check.correction(text_split_no_punc[word_loc].decode('utf-8'))[0]
                else:
                    word = text_split_no_punc[word_loc].decode('utf-8')
                #single word score, if word is not in dictionary, say that it has a score of 0
                score = 0
                if not dist == 0:
                    #see if word is in sentiment dictionary
                    if word in SA_dict:
                        score = SA_dict[word]
                        #add score to unweighted sentiment counter
                        av_emotion_single += score
                        #add weighted score to weighted sentiment counter
                        weighted_av_emotion_single += score_func(score, dist, wdist, 1)
                #print passages colored by score
                if ex_flag == True:
                    displaytext.insert('end',word+' ',str(wtag))#'insert', ' red text', 'RED')
                    if score == 1:
                        displaytext.tag_config(str(wtag), foreground='green')
                    elif score == -1:
                        displaytext.tag_config(str(wtag), foreground='red')
                    else:
                        if dist == 0:
                            color = '0'
                        else:
                            color = str( (int((100 - 10*np.floor(10*score_func(1, dist, wdist, 1)))*0.8)))
                        displaytext.tag_config(str(wtag), foreground='grey'+color)
                    wtag += 1
            #add up passage scores to document score
            av_emotion_doc += av_emotion_single
            weighted_av_emotion_doc += weighted_av_emotion_single
            #keep track of individual passage score (can be used to get sentiment variation within texts)
            w_av.append(weighted_av_emotion_single)
            #more printing 
            if ex_flag == True:
                displaytext.insert('end', '\nscore: '+str(weighted_av_emotion_single))
                displaytext.insert('insert', '\n\n\n', 'newlines')
        #normalize document scores by the number of TERM appearances
        av_emotion_doc /= (float(len(term_indices)))
        weighted_av_emotion_doc /= float(len(term_indices))
        #aggregate data into single item
        weighted_hist_temp = [TERM, weighted_av_emotion_doc, w_av, av_emotion_doc, total_doc_sentiment]
        if ex_flag == True:
            displaytext.insert(1.0, 'Total score ' + str(weighted_av_emotion_doc) + '\n\n')
            root.mainloop()
    return weighted_hist_temp


