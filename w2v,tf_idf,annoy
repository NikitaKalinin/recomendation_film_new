import re
import sys

from functools import lru_cache
import pymorphy2
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models.word2vec import Word2Vec
import pickle as pkl
MORPH = pymorphy2.MorphAnalyzer()

data = pd.read_csv('all_maybe.csv')
data.fillna(value='', inplace=True)
data=data.applymap(str)
data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

@lru_cache(maxsize=100000)
def get_normal_form (i):
    return MORPH.normal_forms(i)[0]


def normalize_text(text):
    normalized = [ get_normal_form(word)  for word in re.findall('[a-zA-Zа-яА-Я]{3,}', text)] # [a-zA-Zа-яА-Я]{3,}|\d+
    return ' '.join([word for word in normalized])
    
data['description_parse']=pd.Series([data.decription[i]+' '+data['Название:'][i]+' '+','.join(data['Жанр:'][i].split(',')[:2])+' '+','.join(data['В ролях:'][i].split(',')[:2])+' '+data['Режиссер:'][i] for i in range(data.decription.shape[0])])
from multiprocessing import Pool
with Pool(5) as pool:
    data['description_parse'] = pool.map(normalize_text ,data['description_parse'])
sentences = data.description_parse.str.split()

model = Word2Vec(sentences, size=500, workers=8, iter=100,window=6,sg=1,min_count=1)
model.save('./w2v_products_our_films.w2v_gensim3') 

model = Word2Vec.load('./w2v_products_our_films.w2v_gensim3')

w2v = dict(zip(model.wv.index2word, model.wv.syn0))
word_to_idx = {} # для преобразования текста к индексам - скармливание модели.
counter = 0
embed_matrix = []

for i in w2v:
    word_to_idx[i] = counter
    embed_matrix.append(w2v[i])
embed_matrix = np.vstack(embed_matrix)

model.similar_by_vector(model['мать'] )

data_storage = {i[0]:i[1]['decription'] + ' ' + i[1]['Название:']+ ' ' + i[1]['Жанр:']+ ' ' + i[1]['В ролях:']+ ' ' + i[1]['Режиссер:']for i in data.iterrows()}



titles_ = (data.decription+' '+data['Название:']+' '+data['Жанр:']+' '+data['В ролях:']+' '+data['Режиссер:']).values
# [normalize_text(i) for i in list(data_storage.values())]

titles_1=np.array(titles_,dtype=titles_.dtype)
with Pool(8) as pool:
    titles_1 = pool.map(normalize_text, titles_)
    
with Pool(8) as pool:
    titles_1 = pool.map(str, titles_1)
data['description_parse'][0].replace('й','и').replace('ё','е').lower()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# our corpus
dates = data['description_parse']

cv = CountVectorizer()

# convert text data into term-frequency matrix
dates = cv.fit_transform(dates)

tfidf_transformer = TfidfTransformer()

# convert term-frequency matrix into tf-idf
tfidf_matrix = tfidf_transformer.fit_transform(dates)

# create dictionary to find a tfidf word each word
tf_idf_vocab = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))


import pickle as pkl
pkl.dump(tf_idf_vocab, open('tf_idf_vocab2', 'wb'))

tf_idf_vocab = pkl.load(open('tf_idf_vocab2', 'rb'))

data_storage_norm = {}
for i in (data_storage):
    text = normalize_text(data_storage[i])
    vec = np.zeros(500)
    for word in text.split(' '):
       
        if word in model and word in tf_idf_vocab:
            vec+=model[word] * tf_idf_vocab[word]

    data_storage_norm[i] = vec
    
from annoy import AnnoyIndex

NUM_TREES = 1000
VEC_SIZE_EMB = 500

counter = 0

index_title_emb = AnnoyIndex(VEC_SIZE_EMB)

print("Build annoy base")
for prod_hash in data_storage_norm:
    title_vec = data_storage_norm[prod_hash] # Вытаскиваем вектор
    
    index_title_emb.add_item(counter, title_vec) # Кладем в анной
    counter +=1
        
index_title_emb.build(NUM_TREES)
print("builded")

index_title_emb.save('./annoy_52')

VEC_SIZE_EMB=500
index_img_emb = AnnoyIndex(VEC_SIZE_EMB)
index_img_emb.load('./annoy_52')

import itertools

vector='сталлоне бежит из тюрьмы'
text = normalize_text(vector)

vec = np.zeros(500)
for word in text.split(' '):
    print(word)
    if word in model and word in tf_idf_vocab:
        vec+=model[word] * tf_idf_vocab[word]
#print(vec)
annoy_res = index_title_emb.get_nns_by_vector(vec, 10, include_distances=True,search_k=5000)
print(annoy_res)
pred= data.iloc[annoy_res[0]]
pred

