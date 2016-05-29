import pickle
import numpy as np
import re

(W, H) = pickle.load(open('NMF_topics_WH_lang.pkl','rb'))

raw_text = pickle.load(open('raw_text_data.pkl','rb'))

print(W.shape)

    # Make sure NaNs turn into strings
    # (We probably don't want this in the long run)
raw_text = [str(x) for x in raw_text]
print("Number of Samples:", len(raw_text))

clean_text = [" ".join([   # joins a list of words back together with spaces in between them
                                re.sub(r'[^a-zA-Z]+', '', # force alphanumeric (after doing @ and # checks)
                                word.replace('"','').lower()) # force lower case, remove double quotes
                            for word in tweet.split() # go word by word and keep them if...
                                if not word.startswith('@') and # they don't start with @, #, or http
                                not word.startswith('#') and
                                not word.startswith('http')]
                            ) # force ascii encoding, ignore weird characters just in case
                        for tweet in raw_text]

for (i,x) in enumerate(clean_text[:50]):
    print(x)
    print(np.argmax(W[i]))
    print('\n')
