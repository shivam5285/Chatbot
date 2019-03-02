"""
This code generates SQL database using reddit comments 
"""

#import pandas as pd
import sqlite3
#import pickle

#==============================================================================
# df = pd.read_csv('trainfrom.txt')
# print(df.head())
#==============================================================================
conn = sqlite3.connect('2009-06.db')
c = conn.cursor()


    #data = c.fetchall()
    #print(data)
        


import os
import json
import nltk
import gensim
import numpy as np
from gensim import corpora, models, similarities
import pickle

model = gensim.models.Word2Vec.load('word2vec.bin');

x=[]
y=[]
path2 = "data"

def read_from_db():
    c.execute('SELECT * FROM parent_reply WHERE parent IS NOT NULL')
    for row in c.fetchall():
        
        x.append(row[2])
        y.append(row[3])
        
read_from_db()
#print(x)

tok_x=[]
tok_y=[]
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))
    if i%5000==0:
        print(i)
        print('Done')
    #print(x[i], y[i])
    
    

sentend=np.ones((300,),dtype=np.float32) 
#print(sentend)
vec_x=[]
for sent in tok_x:
    sentvec = [model[w] for w in sent if w in model.wv.vocab]
    vec_x.append(sentvec)

vec_y=[]
for sent in tok_y:
    sentvec = [model[w] for w in sent if w in model.wv.vocab]
    vec_y.append(sentvec)           
#print("DONE: vec_y.append(sentvec)")   
    
for tok_sent in vec_x:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    
#print("DONE: tok_sent.append(sentend)")
for tok_sent in vec_x:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend) 
#print("DONE: tok_sent.append(sentend)")
            
for tok_sent in vec_y:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    
#print("DONE: tok_sent.append(sentend)")
for tok_sent in vec_y:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)                
#print("DONE: tok_sent.append(sentend)")           
with open('conversation.pickle','w') as f:
    pickle.dump([vec_x,vec_y],f)
