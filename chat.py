import os
from scipy import spatial
import numpy as np
import gensim
import nltk
from keras.models import load_model
 
import theano
theano.config.optimizer="None"

model=load_model('LSTM5000.h5')
mod = gensim.models.Word2Vec.load('word2vec.bin');
while(True):
    x=raw_input("Enter the message:");
    sentend=np.ones((300,),dtype=np.float32) 

    sent=nltk.word_tokenize(x.lower())
    sentvec = [mod[w] for w in sent if w in mod.wv.vocab]

    sentvec[14:]=[]
    sentvec.append(sentend)
    if len(sentvec)<15:
        for i in range(15-len(sentvec)):
            sentvec.append(sentend) 
    sentvec=np.array([sentvec])
    
    predictions = model.predict(sentvec)
    #print predictions
    outputlist=[mod.most_similar([predictions[0][i]])[0][0] for i in range(15)]
    
    e = []
    for i in outputlist:
    		e.append(i)

    output=' '.join(e)
    print(output)
