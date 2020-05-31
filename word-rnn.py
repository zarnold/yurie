import tensorflow as tf 
import os 
import tensorflow_datasets as tdfs
import re
import numpy as np

print(tf.__version__)
# a2.2.0
# please check that you manage CUDA


'''
STANDARD PIPELINE :
  getting -text
  clean and filter
  tokenize it
  build the Features and the Target
  Architecture yor neural net
  Define the loss
  Run it and define a validation method 
  Optimize
'''

SEQ_LENGTH = 5
SKIP_LENGTH= 1
WORD_DIM = 20

# Any of you favorite source
datasetPath = os.path.join("G:","IA","Dataset","textes")
srcTxt = os.path.join(datasetPath,"lyrics_rap_fr.txt")

srcTxt = os.path.join(".","lecture_ce2.txt")


# STEP 0 - Getting text
with open(srcTxt,'r',encoding='utf8') as fh:
    txt = fh.read().lower()


# STEP 1 - Cleaning text

def cleanTxt(txt) :
    res = re.sub("[\(\)\[\]\',;:]","",txt)
    res = res.replace('\n',' \n ')
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

# STEP 3 -- Build the dataset Feature --> Target

nbSentences = ( (len(txtToken) - SEQ_LENGTH) // SKIP_LENGTH) -2


xIndexer = np.arange(SEQ_LENGTH)[None, :] + SKIP_LENGTH * np.arange(nbSentences)[:, None]
yIndexer = SEQ_LENGTH + SKIP_LENGTH * np.arange(nbSentences)[:, None]

x = txtToken[xIndexer]
y = txtToken[yIndexer]

# Buidl a random index for shuffling
randomIdx =np.arange(len(x))
np.random.shuffle(randomIdx)

# Same shuffling for x and y !
x = x[randomIdx]
y = y[randomIdx]


# Check you're fine
for u,v in zip(x[:50],y[:50] ) : 
    input =  [idx2word[w] for w in u]
    output =  [idx2word[t] for t in v]
    print("{} ====> {} ".format(input, output))



#STEP 4 -- Arthictecturing a Neural net for our problem
# Most basic solution 


# Function that generate txt
def generateOutput(model=model) :
    randomIdx = np.random.randint(len(txtArray)-SEQ_LENGTH)
    xvalid= np.array(txtArray[randomIdx:randomIdx+SEQ_LENGTH])
    xvalidEnc = np.array([[ word2idx[w] for w in xvalid]])
    yhat = model.predict(xvalidEnc)
    candidatesWord = np.argsort(yhat, axis=1)
    selectedWords = candidatesWord[0][-3:]
    outputWord = [ idx2word[i] for i in selectedWords]
    print("{}  ======> {} ".format(xvalid, outputWord))


class genCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        if not batch%30 ==0 :
            return             
        print("="*33)
        print(" Epoch {}".format( batch))
        sample = generateOutput(model=model)
        print(sample)

        
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), WORD_DIM),
    tf.keras.layers.LSTM(10),
    tf.keras.layers.Dense(len(vocab), activation="softmax")
])



loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

history = model.fit(x, y, epochs=600, callbacks=[ genCallback()] )


#  Check your model


generateOutput()

# Debugging
inSentence = [[ idx2word[i] for i in sentence ] for sentence in x]
yhat = model.predict(x)
what = [ [idx2word[w] for w in np.argsort(sequence)][-3:] for sequence in yhat]

for u, v in zip(inSentence, what):
    print('{} ====> {}'.format(u, v))



# Inquire the weight
e = model.layers[0]
wghts = e.get_weights()[0]



#
# Compute distance between words

w1="méchant."
w2="amoureux"

def distance(w1 ,w2):
    i1 = word2idx[w1]
    i2 = word2idx[w2]
    v1 = wghts[i1]
    v2 = wghts[i2]
    dist = np.linalg.norm(v1-v2)
    return dist

# Reduce the dim
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(wghts)
lowDim = pca.transform(wghts)

#2D
fig = plt.figure()
ax = fig.gca()
for idx, wght in enumerate(wghts):
    ax.text(wght[0], wght[1],   idx2word[idx])

ax.set_xlim(wghts.min(), wghts.max())
ax.set_ylim(wghts.min(), wghts.max())

# Display
import pylab as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
for idx, v in enumerate(lowDim):
    ax.text(v[0], v[1], v[2],   idx2word[idx])



ax.set_xlim(lowDim.min(), lowDim.max())
ax.set_ylim(lowDim.min(), lowDim.max())
ax.set_zlim(lowDim.min(), lowDim.max())
plt.show()




