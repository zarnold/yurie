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

# STEP 3 -- Build the dataset Feature --> Target

nbSentences = len(txtToken) // SEQ_LENGTH

xIndexer = np.arange(SEQ_LENGTH)[None, :] + SKIP_LENGTH * np.arange(nbSentences)[:, None]
yIndexer = SEQ_LENGTH + SKIP_LENGTH * np.arange(nbSentences)[:, None]

x = txtToken[xIndexer]
y = txtToken[yIndexer]

# Check you're fine
for u,v in zip(x[:50],y[:50] ) : 
    input =  [idx2word[w] for w in u]
    output =  [idx2word[t] for t in v]
    print("{} ====> {} ".format(input, output))


X= tf.convert_to_tensor(x)

#STEP 4 -- Arthictecturing a Neural net for our problem
# Most basic solution 


model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(None,14)),
    tf.keras.layers.Dense(len(vocab), activation="softmax")
])



loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

history = model.fit(x, y, epochs=150 )


#  Check your model
def generateOutput(model=model) :
    randomIdx = np.random.randint(len(txtArray)-SEQ_LENGTH)
    xvalid= np.array(txtArray[randomIdx:randomIdx+SEQ_LENGTH])
    xvalidEnc = np.array([[ word2idx[w] for w in xvalid]])
    yhat = model.predict(xvalidEnc)
    candidatesWord = np.argsort(yhat, axis=1)
    selectedWords = candidatesWord[0][-5:]
    outputWord = [ idx2word[i] for i in selectedWords]
    print("{}  ======> {} ".format(xvalid, outputWord))


generateOutput()