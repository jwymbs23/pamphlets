import pickle
import csv 
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("french")
#take the sentiment csv and convert to a python dictionary with the form {word: score, ...}
#pickle the dictionary

sent_dict = {}
sent_count = {}
sentiment = csv.reader(open('FEEL_sentiment.csv', 'r'), delimiter=';')

stem = True
for rc, row in enumerate(sentiment):
    if rc > 1:
        #print(row[2])
        num_cat = sum(map(int,row[3:]))
        if stem == False:
            sent_dict[row[1].decode('utf-8')] = 1 if row[2] == u'positive' else -1 #1.*num_cat/6 if row[2] == 'positive' else -1*num_cat/6.
        else:
            stem_word = stemmer.stem(row[1].decode('utf-8'))
            if stem_word in sent_dict:
                score_change =  1.0 if row[2] == u'positive' else -1.0
                sent_dict[stem_word] += score_change
                sent_count[stem_word] += 1.0
            else:
                sent_dict[stem_word] = 1.0 if row[2] == u'positive' else -1.0
                sent_count[stem_word] = 1.0

if stem == False:
    pickle.dump(sent_dict, open('sentiment_dictionary_FEEL.pkl', 'wb'))
else:
    for word in sent_dict:
        sent_dict[word] /= sent_count[word]
    pickle.dump(sent_dict, open('sentiment_dictionary_stem_FEEL.pkl','wb'))
print(sent_dict)
