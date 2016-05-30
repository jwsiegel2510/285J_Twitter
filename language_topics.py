import pickle
import numpy as np
import math

(W, H) = pickle.load(open('../LDA_WH_matrices/LDA_100_topics_WH.pkl','rb'))

topic_list = []

language_list = pickle.load(open('language_list.pkl','rb'))

language_set = set(language_list)
llist = list(language_set)
ldict = { x:llist.index(x) for x in llist}

important_tweets = []

for i in range(len(W[0])):
    important_tweets = important_tweets + list(np.argsort(W[:,i])[-30:])

important_tweets = list(set(important_tweets))

# print(important_tweets)

for i in range(len(language_list)):
    topic_list.append(np.argmax(W[i]))

Lang_Topic_Mat = np.zeros((len(H),len(llist)))
Lang_Vector = np.zeros(len(llist))

for k in important_tweets:
    Lang_Topic_Mat[topic_list[k],ldict[language_list[k]]] += 1

for k in important_tweets:
    Lang_Vector[ldict[language_list[k]]] += 1

Entropies = []

for i in range(len(H)):
    s = np.sum(Lang_Topic_Mat[i,:])
    entropy = 0.0
    for j in range(len(llist)):
        if Lang_Topic_Mat[i,j] > 0:
            p = Lang_Topic_Mat[i,j]/s
            entropy -= p*math.log(p)
    Entropies.append(entropy)

print(np.sum(Entropies)/len(Entropies))

entropy = 0
for i in range(len(llist)):
    if Lang_Vector[i] > 0:
        p = Lang_Vector[i]/len(important_tweets)
        entropy -= p*math.log(p)

print(entropy)
