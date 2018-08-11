# -*- coding: utf-8 -*-
import re
#from collections import Counter

#WORDS = Counter(words(open('../corpus/corpus.txt').read()))
WORDS = {}
with open('./corpus/fr.txt', 'r') as f:
    for line in f:
        split_line = line.strip().split(' ')
        key,val = split_line[0], int(split_line[1])
        if val > 300:
            WORDS[key] = val
#print(WORDS)
N = sum(WORDS.values())
#print(len(WORDS))

def P(word):#N=sum(WORDS.values())): 
    "Probability of `word`."
    if word in WORDS:
        return WORDS[word] / N
    else:
        return 0

def correction(word): 
    "Most probable spelling correction for word."
    #print(candidates(word))
    if word in WORDS:
        return word,0
    max_candidate = max(candidates(word), key = P)
    max_prob =  P(max_candidate)
    if max_prob == 0:
        return word, -1
    else:
        return max_candidate, 1
    #else:
    #   return ''

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or [word])#known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyzéàèùâêîôûäëüç'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    #transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + replaces + inserts)# + transposes)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

