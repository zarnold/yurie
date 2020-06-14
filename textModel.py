import re
import numpy as np
from os.path import join
import tensorflow as tf 

from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector

DATA_FOLDER = join('.','DATA')
SRC = join(DATA_FOLDER, 'humanChronology.txt')

WORD_DIM = 6
LATENT_DIM=3

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


word2idx = { word:idx for idx,word in enumerate(vocab) }
idx2word = { word2idx[word]:word for word in word2idx }

def sequenceToTokens(stringSeq) :
    return [ word2idx[word] if word in word2idx else 1+len(word2idx) for word in stringSeq ]



txtArray = cleaned.split('ENDOFLINE')


tokenArray = [ sequenceToTokens(sequence.split()) for sequence in txtArray ]


sizes = [ len(sequence) for sequence in tokenArray]
SEQ_LEN = max(sizes)
train = np.zeros((len(txtArray), SEQ_LEN), dtype=int)
print(train.shape)

padAfter = False
for idx, seq in enumerate(tokenArray) :
    if padAfter : 
        wordsIdx = np.arange(len(seq))
    else :
        wordsIdx = np.arange(SEQ_LEN-len(seq), SEQ_LEN)
    print(seq)
    print(wordsIdx)
    print(words)
    train[idx][wordsIdx] = seq









# This does not work 
encoded = tf.keras.models.Sequential([
    Input(),
    LSTM(32, name='second_lstm'),
    Dense(LATENT_DIM, activation='relu')
])

y2 = encoded.predict(train)
print(y2)


decoded =tf.keras.models.Sequential([
    Input(shape=LATENT_DIM),
    RepeatVector(SEQ_LEN),
    LSTM(300, return_sequences=True, name='inter')
])


model = tf.keras.models.Sequential([
    encoded,decoded
])


#  So here is the flow
y = model.predict(x_train)

y3 = decoded.predict(y2)

tellSentence(y3[9])

ran = np.random.rand(5,LATENT_DIM) 
y3 = decoded.predict(ran)
tellSentence(y3[2])


# Train the whole stuff
model.compile(optimizer='adam',
           loss="mse",
           metrics=['accuracy'])

history = model.fit(x_train, x_train, epochs=800,
                 callbacks=get_callbacks(),
                 validation_data=(validation, validation))
