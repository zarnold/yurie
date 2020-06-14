import re
import numpy as np
from os.path import join
import tensorflow as tf 

from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, Embedding, Bidirectional, TimeDistributed

DATA_FOLDER = join('.','DATA')
SRC = join(DATA_FOLDER, 'humanChronology.txt')

WORD_DIM = 32
LATENT_DIM=32
LSTM_ENCODER_SIZE=64
LSTM_DECODER_SIZE=64

SAMPLE_PRINT_PERIOD = 200

def cleanChro(filePath):
    text   = ""
    with open(filePath, 'r', encoding='utf8') as fp :
        line = fp.readline().strip()
        while line:
            line = fp.readline().strip()
            g = re.match('^[0-9].*:',line)
            if g :
                text += line
                text += ' ENDOFLINE '
    return(text)




cleaned = cleanChro(SRC)

CLEANED_SRC = join(DATA_FOLDER,'cleanedChro.txt')
with open(CLEANED_SRC,'w',encoding='utf8') as fp :
    fp.write(t)




#STEP 2 - Tokenizer
# any function you like as long as each word is map to a number, of a vector
# and that you have a decode
vocab = set(cleaned.split())



# Encoder and decoders could be anything
# a complexe Vectorization
# or else
word2idx = { word:idx for idx,word in enumerate(vocab) }
idx2word = { word2idx[word]:word for word in word2idx }



def encode(stringSeq) :
    return [ word2idx[word] if word in word2idx else len(word2idx) for word in stringSeq ]



def decode(tokensSeq) :
    return ' '.join([ idx2word[token] if token in idx2word else '<UNK>' for token in tokensSeq ])




txtArray = cleaned.split('ENDOFLINE')


tokenArray = [ encode(sequence.split()) for sequence in txtArray ]


sizes = [ len(sequence) for sequence in tokenArray]
SEQ_LEN = max(sizes)
# fill ith the unk token
train = np.zeros((len(txtArray), SEQ_LEN), dtype=int)
train.fill(len(vocab))
print(train.shape)

padAfter = False
for idx, seq in enumerate(tokenArray) :
    if padAfter : 
        wordsIdx = np.arange(len(seq))
    else :
        wordsIdx = np.arange(SEQ_LEN-len(seq), SEQ_LEN)
    train[idx][wordsIdx] = seq





x = train
y = tf.keras.utils.to_categorical(train, 1+len(vocab))



# This does not work 
encoded = tf.keras.models.Sequential([
        Embedding(1+len(vocab), WORD_DIM),
        Bidirectional(tf.keras.layers.LSTM(LSTM_ENCODER_SIZE)), 
        Dense(LATENT_DIM)     
])


decoded =tf.keras.models.Sequential([
        Input( shape=( LATENT_DIM, )),
        RepeatVector(SEQ_LEN),
        LSTM(LSTM_DECODER_SIZE, return_sequences=True),
        TimeDistributed(Dense(1 + len(vocab), activation='relu'))
])


model = tf.keras.models.Sequential([
    encoded,decoded
])


def checkEncodage():
    randomIdx = np.random.randint(len(train))
    sample = train[ randomIdx : randomIdx + 5]
    yhat = model.predict(sample)
    predictedTkens = np.argmax(yhat, axis=2)
    decodedSent = [decode(tokenSeq) for tokenSeq in predictedTkens]
    encodedSent= [decode(tokenSeq) for tokenSeq in sample]
    for u, v in zip(encodedSent, decodedSent) : 
        print("{}   ==> {}".format(encodedSent, decodedSent))



def generateFromLatent():
    randomVector = np.random.random((10, LATENT_DIM))
    r=decoded.predict(randomVector) 
    decodedSent = [decode(tokenSeq) for tokenSeq in np.argmax(r,axis=2)]
    for s in decodedSent :
        print(s)



def gen(e):
    print("=============================== Epoch {} ================================".format(e))
    checkEncodage()
    generateFromLatent()


gen(0)


# Train the whole stuff


checkpoint_path=join(".","training_checkpoints", 'fibreDestroyer')
class genCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        if not batch%SAMPLE_PRINT_PERIOD ==0 :
            return             
        print("")
        gen(batch)




model.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])



history = model.fit(x, y, callbacks=[
                     tf.keras.callbacks.TensorBoard("logs/abw-crusher"), 
                     genCallback()],
                     epochs=2000)





#  np.argmax(y,axis=2) = x