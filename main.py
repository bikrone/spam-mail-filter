import math
from stop_words import get_stop_words
import re

stop_words = set(get_stop_words('en'))

# read data
from os import listdir
from os.path import isfile, join
from collections import defaultdict

TRAIN_HAM_FOLDER = ['dataset/enron-spam/enron1/ham', 'dataset/enron-spam/enron2/ham']
TRAIN_SPAM_FOLDER = ['dataset/enron-spam/enron1/spam', 'dataset/enron-spam/enron2/spam']

TEST_HAM_FOLDER = ['dataset/enron-spam/enron2/ham']
TEST_SPAM_FOLDER = ['dataset/enron-spam/enron2/spam']

number_of_documents = 0.0

def getFilesInPath(listpath):
  result = []
  for mypath in listpath:
    result += [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
  return result

def isWord(token):
  return len(token) > 2 and token not in stop_words

def defaultVal():
  x = dict()
  x['countSpam'] = 0
  x['countHam'] = 0
  x['documentFrequency'] = 0
  return x


def IDF(token):
  return 1 + math.log(float(number_of_documents)/token_info[token]['documentFrequency'])

def getTokens(data):
  tokens = re.split('\W+', data)
  return [token for token in tokens if isWord(token)]

n_sample_tokens = 0.0
n_trained_spams = 0
n_trained_hams = 0

def isSpam(filename):
  f = open(filename)
  data = f.read()
  tokens = getTokens(data)

  # remove tokens not in database
  tokens = [token for token in tokens if token in token_info]

  spamminess = 1
  hamminess = 1

  freq = defaultdict(int)
  tf_idf = defaultdict(int)

  for token in tokens:
    freq[token] += 1

  for token in tokens:
    tf_idf[token] = (1+math.log(freq[token])) * IDF(token)

  # coefficient for spamminess and hamminess: spamminess * sammy ~ hamminess * hammy
  spammy = long(1)
  hammy  = long(1)

  for token in tokens:
    spammy *= n_trained_hams
    hammy *= n_trained_spams
    nContainToken = token_info[token]['countSpam'] +token_info[token]['countHam'] 

    # use Long (Bignum type) to avoid float-overflow (dataloss)
    spamminess *= long(1000.0*(token_info[token]['countSpam'] + 1.0)/(nContainToken + 2))
    hamminess *= long(1000.0*(token_info[token]['countHam'] + 1.0)/(nContainToken + 2))

  nToken = long(len(tokens))
  
  return (spamminess*spammy>=hamminess*hammy)

token_info = defaultdict(defaultVal)

hams = getFilesInPath(TRAIN_HAM_FOLDER)
spams = getFilesInPath(TRAIN_SPAM_FOLDER)

number_of_documents = len(hams) + len(spams)
n_trained_spams = len(spams)
n_trained_hams = len(hams)

current_document_index = 0

for filename in hams:
  current_document_index+=1
  print('Processing {}/{}'.format(current_document_index, number_of_documents))
  f = open(filename)
  data = f.read()
  tokens = getTokens(data)
  for token in tokens:
    token_info[token]['countHam'] += 1
    token_info[token]['documentFrequency'] += 1

for filename in spams:
  current_document_index+=1
  print('Processing {}/{}'.format(current_document_index, number_of_documents))
  f = open(filename)
  data = f.read()
  tokens = getTokens(data)
  for token in tokens:
    token_info[token]['countSpam'] += 1
    token_info[token]['documentFrequency'] += 1

f = open('token_data.txt', 'w')

for token in token_info:
  f.write('{} {} {} {}\n'.format(token, 
          token_info[token]['countSpam'], 
          token_info[token]['countHam'], 
          token_info[token]['documentFrequency']))

f.close()

n_sample_tokens = len(token_info)



# check HAM FOLDER
hams = getFilesInPath(TEST_HAM_FOLDER)
spams = getFilesInPath(TEST_SPAM_FOLDER)

nRight = 0
for filename in hams:
  nRight += 0 if isSpam(filename) else 1

print('Accuracy predicting HAM: {}/{} = {}%'.format(nRight, len(hams), 100.0*float(nRight)/len(hams)))


nRight = 0
for filename in spams:
  nRight += 1 if isSpam(filename) else 0

print('Accuracy predicting SPAM: {}/{} = {}%'.format(nRight, len(spams), 100.0*float(nRight)/len(spams)))
