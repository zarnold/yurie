import re
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
                text += '\n'
    return(text)




t = cleanChro(SRC)

CLEANED_SRC = join(DATA_FOLDER,'cleanedChro.txt')
with open(CLEANED_SRC,'w',encoding='utf8') as fp :
    fp.write(t)
