import pickle
import numpy as np

available_languages = [
    'ar',
    'ca',
    'cs',
    'de',
    'el',
    'en',
    'es',
    'fi',
    'fr',
    'hu',
    'it',
    'ja',
    'kr',
    'nl',
    'no',
    'pl',
    'pt',
    'ru',
    'sk',
    'sv',
    'tr',
    ]

Topic_Language_Mat = pickle.load(open('topic_language_mat.pkl','rb'))

(W, H) = pickle.load(open('../NMF/NMF_500_topics_WH.pkl','rb'))
names = np.array(pickle.load(open('TF_IDF_feature_names.pkl','rb')))

sorted_topics = [list(names[np.argsort(x)[-10:]][::-1]) for x in H]

sorted = [available_languages[np.argmax(x)] for x in Topic_Language_Mat]
for (i,x) in enumerate(sorted):
    print(sorted_topics[i])
    print(x)
