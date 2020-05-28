import tensorflow as tf 
import os 
import tensorflow_datasets as tdfs
import re
import numpy as np

print(tf.__version__)
# a2.2.0
# please check that you manage cuda


SEQ_LENGTH = 14
SKIP_LENGTH= 2

# Any of you favorite source
datasetPath = os.path.join("G:","IA","Dataset","textes")
srcTxt = os.path.join(datasetPath,"lyrics_rap_fr.txt")


# STEP 0 - Getting text
with open(srcTxt,'r',encoding='utf8') as fh:
    txt = fh.read().lower()


# STEP 1 - Cleaning text

def cleanTxt(txt) :
    res = re.sub("[\(\)\[\]\',;:]","",txt)
    return res



cleaned = cleanTxt(txt)
vocab = set(cleaned.split())


#STEP 2 - Tokenizer
# any function you like as long as each word is map to a number, of a vector
# and that you have a decode

word2idx = { word:idx for idx,word in enumerate(vocab) }
idx2word = { word2idx[word]:word for word in word2idx }

txtArray = cleaned.split()
txtToken =np.array([word2idx[w] for w in txtArray])

# STEP 3 -

nbSentences = len(txtToken) // SEQ_LENGTH

xIndexer = np.arange(SEQ_LENGTH)[None, :] + SKIP_LENGTH * np.arange(nbSentences)[:, None]
yIndexer = SEQ_LENGTH + SKIP_LENGTH * np.arange(nbSentences)[:, None]

x = txtToken[xIndexer]
y = txtToken[yIndexer]

# Check you're fine
for u,v in zip( )