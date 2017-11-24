# -*- coding: utf-8 -*-
import utility

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Merge, Multiply, Convolution1D, Lambda

############# DEFINING HYPERPARAMETERS #############

dimx = 50
dimy = 50
batch_size = 20
dense_neuron = 16
nb_filter = 100
vocab_size = 8000
embedding_dim = 300

############# DEVELOPING BASIC CNN MODEL #############

def train_model(X_train_l, X_train_r, label, embedding_matrix):
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
    
    x = utility.word2vec_embedding_layer(embedding_matrix)(inpx)
    y = utility.word2vec_embedding_layer(embedding_matrix)(inpy)
    
    ques = Convolution1D(nb_filter=100,
                                filter_length=4,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1)(x)
    
    ans = Convolution1D(nb_filter=100,
                        filter_length=4,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1)(y)
            
    hx = Lambda(lambda x: K.max(x, axis=1), output_shape=(nb_filter,))(ques)
    hy = Lambda(lambda x: K.max(x, axis=1), output_shape=(nb_filter,))(ans)
        
    h1 = Multiply()([hx,hy])
    h2 = utility.Abs()([hx,hy])
        
    h =  Merge(mode="concat",name='h')([h1, h2])
    
    wrap = Dense(dense_neuron, activation='relu',name='wrap')(h)
    score = Dense(1,activation='sigmoid',name='score')(wrap)
        
    model = Model([inpx, inpy],[score])
    model.compile( loss='mse',optimizer="adadelta")
    
############# TRAINING MODEL #############
    
    model.fit([X_train_l,X_train_r], 
              [label],
              nb_epoch=5,
              batch_size=batch_size,verbose=1)
    
    return model