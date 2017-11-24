import itertools
import gensim
import numpy as np

from nltk import FreqDist
from nltk.tokenize import regexp_tokenize

from keras import backend as K
from keras.layers import Layer, Embedding

START = '$_START_$'
END = '$_END_$'
unk_token = '$_UNK_$'

def process_data(sent_Q,sent_A,wordVec_model,dimx=100,dimy=100,vocab_size=10000,embedding_dim=300):
#if True:
    sent1 = []
    
    sent1.extend(sent_Q)
    sent1.extend(sent_A)
    
    sent1 = [' '.join(i) for i in sent1]

    sentence = ["%s %s %s" % (START,x,END) for x in sent1]
    tokenize_sent = [regexp_tokenize(x, 
                                     pattern = '\w+|$[\d\.]+|\S+') for x in sentence]
    
    
    freq = FreqDist(itertools.chain(*tokenize_sent))
    print 'found ',len(freq),' unique words'
    vocab = freq.most_common(vocab_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unk_token)
    
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    for i,sent in enumerate(tokenize_sent):
        tokenize_sent[i] = [w if w in word_to_index else unk_token for w in sent]
    
    len_train = len(sent_Q)
    text=[]
    for i in tokenize_sent:
        text.extend(i)
    
    sentences_x = []
    sentences_y = []
    
    for sent in tokenize_sent[0:len_train]:
        temp = [START for i in range(dimx)]
        for ind,word in enumerate(sent[0:dimx]):
            temp[ind] = word
        sentences_x.append(temp)
       
    for sent in tokenize_sent[len_train:]:
        temp = [START for i in range(dimy)]
        for ind,word in enumerate(sent[0:dimy]):
            temp[ind] = word       
        sentences_y.append(temp)
     
    X_data = []
    for i in sentences_x:
        temp = []
        for j in i:
            temp.append(word_to_index[j])
        temp = np.array(temp).T
        X_data.append(temp)
    
    y_data=[]
    for i in sentences_y:
        temp = []
        for j in i:
            temp.append(word_to_index[j])
        temp = np.array(temp).T
        y_data.append(temp)
    
    X_data = np.array(X_data)
    y_data = np.array(y_data) 

    embedding_matrix = np.zeros((len(index_to_word) + 1,embedding_dim))
    
    unk = []
    for i,j in enumerate(index_to_word):
        try:
            embedding_matrix[i] = wordVec_model[j]
        except:            
            unk.append(j)
            continue
    print 'number of unkown words: ',len(unk)
    print 'some unknown words ',unk[0:5]
    return X_data,y_data,embedding_matrix, word_to_index

class Abs(Layer):
    def __init__(self, **kwargs):
        super(Abs, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        inp1, inp2 = x[0],x[1]
        return K.abs(inp1-inp2)
    
    def get_output_shape_for(self, input_shape):
        return input_shape

def word2vec_embedding_layer(embedding_matrix):
    layer = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix])
    return layer

def process_test(sent_A, sent_B, word_to_index, dimx = 100, dimy = 100):
    X_test_A = []
    X_test_B = []
    
    unk = 0
    
    temp = [START for i in range(dimx)]
    for num, word in enumerate(sent_A.split()):
        temp[num+1] = word
    
    X_A = []
    for word in temp:
        try:
            X_A.append(word_to_index[word])
        except:
            X_A.append(unk)
    
    temp = [START for i in range(dimy)]
    for num, word in enumerate(sent_B.split()):
        temp[num+1] = word
    
    X_B = []
    for word in temp:
        try:
            X_B.append(word_to_index[word])
        except:
            X_B.append(unk)
    
    X_test_A.append(X_A)
    X_test_B.append(X_B)
    
    return np.array(X_test_A), np.array(X_test_B)


############# LOADING WORD VECTOR MODEL #############

try:
    word = wordVec_model['word']
    print 'using loaded model.....'
except:
    wordVec_model = gensim.models.KeyedVectors.load_word2vec_format("../../../GoogleNews-vectors-negative300.bin.gz",binary=True)

############# PROCESSING DATA #############

def process_train(q1_train, q2_train, dimx = 100, dimy = 100, vocab_size =  8000, embedding_dim = 300):
    q1 = []
    q2 = []
    
    for sent in q1_train:
        q1.append(sent.split())
    
    for sent in q2_train:
        q2.append(sent.split())
    
    X_train_l, X_train_r, embedding_matrix, word_to_index = process_data(q1, q2,
                                                                         wordVec_model, dimx=dimx,
                                                                         dimy=dimy, vocab_size = vocab_size,
                                                                         embedding_dim = embedding_dim)
    
    return X_train_l, X_train_r, embedding_matrix, word_to_index

############# PREDICTONG PALAGRISM #############

def predict(model, emb_q1, emb_q2):
    return 1 - model.predict([emb_q1, emb_q2])