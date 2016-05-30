import pickle
import numpy as np
import math

(W, H) = pickle.load(open('../LDA_WH_matrices/LDA_100_topics_WH.pkl','rb'))

topic_list = []

language_list = pickle.load(open('language_list.pkl','rb'))

language_set = set(language_list)
llist = list(language_set)
ldict = { x:llist.index(x) for x in llist}

for i in range(len(language_list)):
    topic_list.append(np.argmax(W[i]))

Lang_Topic_Mat = np.zeros((len(H),len(llist)))

for i,j in zip(topic_list,language_list):
    Lang_Topic_Mat[i,ldict[j]] += 1

Entropies = []

for i in range(len(H)):
    s = np.sum(Lang_Topic_Mat[i,:])
    entropy = 0.0
    for j in range(len(llist)):
        if Lang_Topic_Mat[i,j] > 0:
            p = Lang_Topic_Mat[i,j]/s
            entropy -= p*math.log(p)
    Entropies.append(entropy)

for i in range(len(Entropies)):
    print(Entropies[i])
