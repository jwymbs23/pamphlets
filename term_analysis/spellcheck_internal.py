import numpy as np
import string

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



def get_docs_in_year(df, year):
    return df.loc[df['date'] == year]['identifier'].tolist()


def clean_term_dict(full_term_data):
    for year_dict in full_term_data:
        for term in list(year_dict.keys()):
            if len(term) < 3 or term in stopwords or '.' in term or year_dict[term] < 10:
                del year_dict[term]
    return full_term_data


def strip_stopwords_punc(text):
    clean_text = []
    for word in text:
        if word not in stopwords and word not in string.punctuation:
            clean_text.append(word)
    return clean_text





#from math import log

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
#words = open("words-by-frequency.txt").read().split()
#wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
#maxword = max(len(x) for x in words)




def split_check(full_dict, fix_dict, word, total_wc, top_replacements):

    def calc_split_prob(substr):
        print(substr, np.log(full_dict[substr]*np.log(len(substr)+1)))
        return np.log(full_dict[substr]*np.log(len(substr)+1))
        
    #print(word)


    def single_spell_correct(word):
        max_word = word
        check_set = smart_word_perms([word], top_replacements)
        check_set = [i for i in check_set if i in full_dict]
        check_set = smart_word_perms(check_set, top_replacements)
        check_set = [i for i in check_set if i in full_dict]
        print(check_set)
        #print(check_set)
        for word_comp in check_set:
            if full_dict[word_comp] > full_dict[max_word]:
                max_word = word_comp
        return max_word

    
    split_prob = [(calc_split_prob(single_spell_correct(a)) if a != '' else calc_split_prob(single_spell_correct(b)), calc_split_prob(single_spell_correct(b))) for a,b in [(word[:i], word[i:]) for i in range(len(word))]]
    score = [s1+s2 for s1,s2 in split_prob]
             
    splits = [(single_spell_correct(a),single_spell_correct(b)) for a,b in [(word[:i], word[i:]) for i in range(len(word))]]
    #print(split_prob)
    #print(score)
    #print(splits)
    idx_max = np.argmax(score)

    #print(np.log(score[idx_max]/score[0]))
    #print(word, splits[idx_max])
    #print(score)
    print(word, splits[idx_max])
    input()
    for split_word in splits[idx_max]:
        try:
            fix_dict[split_word] += full_dict[word]
        except:
            fix_dict[split_word] = full_dict[word]
    return fix_dict




def same_char_same_pos(word1, word2):
    scsp = 0
    diff = []
    for l1, l2 in zip(word1, word2):
        if l1 == l2:
            scsp += 1
        else:
            diff.append((l1,l2))
    return scsp, diff




def gen_word_perms(word, charset):
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    replaces   = [L + c + R[1:] for L, R in splits if R for c in charset]
    return set(replaces)


def smart_word_perms(words, top_replacements):
    replaces = []
    for word in words:
        splits =  [(word[:i], word[i:])    for i in range(len(word) + 1)]
        # change to make more that one replacement?
        replaces.extend([L + c + R[1:] for L,R in splits if R for c in top_replacements[word[len(L)]]])
        #print(replaces)
    return set(replaces)


def replacements(full_dict, word, total_wc, rep_table, map_chars, charset):
    count = 0
    test_words = gen_word_perms(word, charset)
    max_word = '+'
    for word_comp in test_words:
        if word_comp in full_dict:
            # if len(word) == len(word_comp):
            # diff: letter replacement from - to
            scsp, diff = same_char_same_pos(word, word_comp)
            if scsp >= len(word)-1 and full_dict[word] < full_dict[word_comp]:
                #print(word, word_comp, scsp, diff)
                #print(full_dict[word_comp], full_dict[max_word])
                #if full_dict[word_comp] > full_dict[max_word]:
                #    max_word = word_comp
                for rep in diff:
                    rep_table[map_chars[rep[0]]][map_chars[rep[1]]] += 1
                    count += 1
    #print(rep_table)
    #if max_word != '+':
        #full_dict[max_word] += full_dict[word]
        #print(word, max_word)
        #del full_dict[word]
    return rep_table, count, full_dict
        

def validate(pamphlet_dict, modern_dict):
    keys_a = set(pamphlet_dict.keys())
    keys_b = set(modern_dict.keys())
    intersection = keys_a & keys_b
    return len(intersection)/len(keys_a), len(intersection)


def read_modern_dict():
    word_dict = {}
    with open('fr.txt', 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            count = split_line[1]
            word_dict[word] = count
    return word_dict
        

def load_clean_word_list():
                    
    pickle_file = glob.glob('full_word_list.pkl')
    term_data = pickle.load(open('full_word_list.pkl', 'rb'))
    #print(term_data[4])
    full_dict = Counter({})
    for year_data in term_data:
        full_dict += year_data
    del term_data
    #print(full_dict)

    # read in modern french word list
    modern_dict = read_modern_dict()

    word_list = list(full_dict)
    print(len(word_list), validate(full_dict, modern_dict))
    # split on periods
    for word in word_list:
        if '.' in word:
            for sub in word.split('.'):
                try:
                    full_dict[sub] += full_dict[word]
                except:
                    full_dict[sub] = 1
            del full_dict[word]
        if ',' in word:
            del full_dict[word]
        #count digits:
        if len([i for i in word if i.isdigit()]) > 0.5*len(word):
        #    print(word)
            
            del full_dict[word]
    word_list = list(full_dict)
    print(len(word_list), validate(full_dict, modern_dict))
    
    total_wc = sum(full_dict.values())
    fixed_dict = Counter({})

    # charset and replacement matrix
    charset = set(''.join(word_list))
    #print(charset)
    map_chars = {}
    charlist = []
    for ci, i in enumerate(charset):
        charlist.append(i)
        map_chars[i] = ci

    return full_dict, modern_dict, map_chars, charlist


def gen_rep_table(full_dict, modern_dict, map_chars, charlist):
    word_list = list(full_dict)
    fixed_dict = Counter({})

    # charset and replacement matrix
    charset = set(''.join(word_list))
    print(charset)
    map_chars = {}
    charlist = []
    for ci, i in enumerate(charset):
        charlist.append(i)
        map_chars[i] = ci


    total_wc = sum(full_dict.values())
        
    rep_table = np.zeros((len(charset),len(charset)))
    print(rep_table)
    cw_valid = 1
    count = 0
    for cw, word in enumerate(word_list):
        if not cw%100000:
            print(cw, validate(full_dict, modern_dict))
            if cw > 0:
                break
        if len(word) > 5:
            rep_table, count, full_dict = replacements(full_dict, word, total_wc, rep_table, map_chars, charset)
            cw_valid += 1
    print(np.max(rep_table))

    


    pickle.dump({'total_count': count, 'rep_table': rep_table, 'charlist': charlist, 'map_chars': map_chars}, open('rep_table.pkl', 'wb'))
    
    plt.imshow(rep_table, interpolation=None)
    plt.xticks(range(len(charlist)), charlist)
    plt.yticks(range(len(charlist)), charlist)
    plt.show()

    # do splits
    #for cw, word in enumerate(word_list):
    #    split_check(full_dict, word, total_wc)
    #    if cw > 100:
    #        break
    return rep_table, count


def spell_correct(full_dict, modern_dict, top_replacements):
    #word_list = list(full_dict)
    #for cw, word in enumerate(word_list):
    max_word = word
    check_set = smart_word_perms(word, top_replacements)
    for word_comp in check_set:
        if full_dict[word_comp] > full_dict[max_word]*5:
            max_word = word_comp
    #if max_word != word:
        #print(word, max_word)
    #    full_dict[max_word] += full_dict[word]
    #    del full_dict[word]
    #if not cw%10000:
    #    print(validate(full_dict, modern_dict))
    return max_word



    
if __name__ == "__main__":
    full_dict, modern_dict, map_chars, charlist = load_clean_word_list()
    word_list = list(full_dict)



    
    
    
    #print(vectorize('le', map_chars))
    remake_rep_table = False
    if remake_rep_table == True:
        rep_table, count = gen_rep_table(full_dict, modern_dict, map_chars, charlist)
    else:
        rep_data = pickle.load(open('rep_table.pkl', 'rb'))
        
        print(rep_data)
        rep_table = rep_data['rep_table']
        charlist = rep_data['charlist']
        try:
            map_chars = rep_data['charmap']
        except:
            map_chars = rep_data['map_chars']
    ###
    top_n = 4
    top_replacements = {}
    for cf, from_letter in enumerate(rep_table):
        sort_idx = np.argsort(from_letter)[::-1]
        #print(from_letter)
        top_rep = [sort_idx[i] for i in range(top_n)]
        #print(top_rep)
        top_replacements[charlist[cf]] = [charlist[char] for char in top_rep]
    print(top_replacements)



    fix_dict = Counter({})

    for cw, word in enumerate(word_list):
        if not cw%10000 and cw > 0: 
            print(cw, validate(fix_dict, modern_dict))
        fix_dict = split_check(full_dict, fix_dict, word, len(full_dict), top_replacements)
        #print(fix_dict)
    exit(0)
    #word_list = list(full_dict)
    for cw, word in enumerate(word_list):
        spell_correct(full_dict, modern_dict, top_replacements)
        if max_word != word:
            #print(word, max_word)
            full_dict[max_word] += full_dict[word]
            del full_dict[word]
        if not cw%10000:
            print(validate(full_dict, modern_dict))
    exit(0)
    

    


    
    
    
