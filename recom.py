import re, random, math, pymorphy2
from functools import lru_cache
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models.word2vec import Word2Vec
import pickle as pkl
from scipy.sparse import csr_matrix
import scipy.sparse
from multiprocessing import Pool
from functools import partial

MORPH = pymorphy2.MorphAnalyzer()

class Recom():
    def __init__(self, films='data/films.csv', model='data/w2v_products_our_films.w2v_gensim3', VEC_SIZE_EMB=500,
                 index_img_emb='data/annoy_52'):
        self.data = pd.read_csv(films)
        self.data.fillna(value='', inplace=True)
        self.data = self.data.applymap(str)
        self.model = Word2Vec.load(model)
        self.VEC_SIZE_EMB = VEC_SIZE_EMB
        self.index_img_emb = AnnoyIndex(VEC_SIZE_EMB)
        self.index_img_emb.load(index_img_emb)
        self.tf_idf_vocab = pkl.load(open('data/tf_idf_vocab2', 'rb'))
        self.collab_matrix = scipy.sparse.load_npz('data/new5mean.npz')
        self.films_movielens = pd.read_csv('data/movies.csv')

    @lru_cache(maxsize=100000)
    def get_normal_form(self, i):
        return MORPH.normal_forms(i)[0]

    def normalize_text(self, text):
        text = ' '.join([word for word in re.findall('[a-zA-Zа-яА-Я]{3,}', text)])
        normalized = [self.get_normal_form(word) for word in text.split()]
        return normalized

    def get_recom(self, request='', num=10, search_k=50000):
        text = self.normalize_text(request)
        print(text)
        if text == []:
            return None
        vec = 0
        for word in text:
            w1 = word.replace('й', 'и').replace('ё', 'е')
            if word in self.model and w1 in self.tf_idf_vocab:
                vec = vec + self.model[word] * self.tf_idf_vocab[w1]
        if isinstance(vec, int):
            return None
        annoy_res = list(self.index_img_emb.get_nns_by_vector(vec, 10, include_distances=True,search_k=500000))
        return self.data.iloc[annoy_res[0]].values.tolist()

    def get_recom_by_films(self, test_vector):
        users_index = np.array([i for i in range(self.collab_matrix.shape[0])])
        test_vector[test_vector.nonzero()] += 5 - test_vector[test_vector.nonzero()].mean()
        slice_collab = self.collab_matrix[:, np.nonzero(test_vector)[0]]
        unique_indeces = np.unique(slice_collab.nonzero()[0])
        users_index = users_index[unique_indeces]
        non_zero_slice_collab = slice_collab[unique_indeces, :]
        knn_test = np.array([])
        with Pool(processes=4) as pool:
            knn_test = pool.map(partial(distance, y=test_vector[test_vector.nonzero()]),
                                non_zero_slice_collab.toarray())
            pool.terminate()
        similar_users = self.collab_matrix[users_index[np.argsort(knn_test)[:5]]]
        films_index = np.array([i for i in range(self.collab_matrix.shape[1])])
        unwatched_films = np.delete(films_index, test_vector.nonzero())
        recommend_rows = np.delete(similar_users.toarray(), test_vector.nonzero(), 1)
        mean_score = np.apply_along_axis(best_mean, axis=0, arr=recommend_rows)
        best_index = np.argsort(mean_score)[-10:-1]
        movieIds = unwatched_films[best_index]
        return self.films_movielens.iloc[movieIds].values.tolist()

    def get_movies_shape(self):
        return self.films_movielens.shape[0]

    def get_random_films(self):
        random_list = [random.randint(0, self.films_movielens.shape[0]) for i in range(30)]
        return self.films_movielens.iloc[random_list].values.tolist()

def best_mean(x):
    if len(x[x != 0]) != 0:
        return np.mean(x[x != 0])
    else:
        return 0

def distance(x, y):
    cos_x, cos_y = x[x != 0], y[x != 0]
    return np.mean((cos_x - cos_y)**2) + 1/len(cos_x)




