import re
import numpy as np
from os.path import join

DATA_FOLDER = join('.','DATA')
SRC = join(DATA_FOLDER, 'humanChronology.txt')



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



