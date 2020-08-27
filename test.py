import pandas as pd
import pickle
import re
import pyisbn
import sys
import requests


def save_description_subject(start):
  chunk_size = len(no_des_isbn)//chunk
  begin, end = chunk_size*start, chunk_size*start + chunk_size
  print(begin, end)
  tmp_des = {}
  tmp_sub = {}
  for isbn in tqdm(no_des_isbn[begin:end]):
    # if isbn2description[isbn] != '':
    #   continue
    # if isbn in isbn2description:
    #   continue
    description = []
    subject = []
    try:
      result_openlib = requests.get('https://openlibrary.org/api/books?bibkeys=ISBN:{}&jscmd=details&format=json'.format(isbn)).json()
      key1 = 'ISBN:{}'.format(isbn)
      subject = result_openlib[key1]['details']['subjects']
      description.append(result_openlib[key1]['details']['description'])
    except:
      pass
      # try:
      #   result_google = requests.get('https://www.googleapis.com/books/v1/volumes?q=isbn:{}'.format(isbn)).json()
      #   if len(description) == 0:
      #     description.append(result_google['items'][0]['volumeInfo']['description'])
      #   subject += result_google['items'][0]['volumeInfo']['categories']
      # except:
      #   pass
    if len(description) > 0 and len(isbn2description[isbn]) == 0:
      tmp_des[isbn] = description[0]
    if len(subject) > 0 and len(isbn2subject[isbn]) == 0:
      tmp_sub[isbn] = subject
    # print(description)
    # print(subject)
  with open(str(start)+'_description.pickle', 'wb') as f:
    pickle.dump(tmp_des, f, protocol=pickle.HIGHEST_PROTOCOL)
  with open(str(start)+'_subject.pickle', 'wb') as f:
    pickle.dump(tmp_sub, f, protocol=pickle.HIGHEST_PROTOCOL)


# from multiprocessing import Pool
# from tqdm import tqdm
# from pprint import pprint
# import pyisbn
# try:
#   with open('isbn2description4.pickle', 'rb') as f:
#     isbn2description = pickle.load(f)
# except:
#   isbn2description = {}
#
# try:
#   with open('isbn2popular_shelves4.pickle', 'rb') as f:
#     isbn2subject = pickle.load(f)
# except:
#   isbn2subject = {}
#
# with open('isbn2title4.pickle', 'rb') as f:
#   isbn2rating = pickle.load(f)
# isbn2rating = set(isbn2rating.keys())
#
import numpy as np
# from sentence_transformers import SentenceTransformer
# # model = SentenceTransformer('bert-base-nli-stsb-mean-tokens') # 14.77 22.568
# # model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens') # 26.112 35.97
#
# # sentences = ['The prince revenge to become the king', 'the noble son of old hamlet bring the greatest punishment to his treason king']
# sentences = ['The cat has evolved to become a human, the cat ', 'the noble son of old hamlet bring the greatest punishment to his treason king']
# emb_sentence = model.encode(sentences)
# s1 = emb_sentence[0]
# s2 = emb_sentence[1]
# print(len(s1))
# dist = np.sum((s1-s2)**2)**0.5
# print(dist)

# a1 = np.array([1,2])
# a2 = np.mean([[3,4],[5,6]], axis=1)
# result = np.hstack((a1[:, np.newaxis],a2))
# print(result)
# print(pyisbn.validate('9782013974080'))
# starts = list(range(chunk))
# pool = Pool(processes=len(starts))
# pool.map(save_description_subject, starts)

def pairwise_distances(x, y=None):
  import torch
  '''
  Input: x is a Nxd matrix
         y is an optional Mxd matirx
  Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
          if y is not given then use 'y=x'.
  i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
  '''
  x_norm = (x ** 2).sum(1).view(-1, 1)
  if y is not None:
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
  else:
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

  dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
  # Ensure diagonal is zero if x=y
  # if y is None:
  #     dist = dist - torch.diag(dist.diag)
  dist[dist != dist] = 0
  return dist
  # return torch.clamp(dist, 0.0, np.inf)

from tqdm import tqdm


def pair_sparse_matrix():
  from tqdm import tqdm

  root = './archive_filter_lang/'
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)
  fs = ['book_similar_pair.pickle', 'same_author_pair.pickle'] + \
       ['same_rating_children_{}.pickle'.format(str(i)) for i in range(5)] + \
       ['same_rating_comics_graphic_{}.pickle'.format(str(i)) for i in range(6)] + ['same_rating_poetry.pickle'] + \
       ['same_rating_fantasy_paranormal_{}.pickle'.format(str(i)) for i in range(24)] + \
       ['same_rating_history_biography_{}.pickle'.format(str(i)) for i in range(24)] + \
       ['same_rating_mystery_thriller_crime_{}.pickle'.format(str(i)) for i in range(24)] + \
       ['same_rating_romance_{}.pickle'.format(str(i)) for i in range(24)] + \
       ['same_rating_young_adult_{}.pickle'.format(str(i)) for i in range(24)]
  pair_count = lil_matrix((len(isbn2ind), len(isbn2ind)))
  print(len(isbn2ind), len(isbn2ind))
  for f in fs:
    print('Processing', f)
    with open(root+f, 'rb') as r:
      pair2count = pickle.load(r)
    for pair in tqdm(pair2count):
      if len(pair) < 2:
        continue
      isbn1, isbn2 = pair
      if isbn1 not in isbn2ind or isbn2 not in isbn2ind:
        continue
      # print('element:', pair_count[isbn2ind[isbn1], isbn2ind[isbn2]])
      pair_count[isbn2ind[isbn1], isbn2ind[isbn2]] += pair2count[pair]
      pair_count[isbn2ind[isbn2], isbn2ind[isbn1]] += pair2count[pair]

  with open('similar_pair.pickle', 'wb') as f:
    pickle.dump(pair_count, f, protocol=pickle.HIGHEST_PROTOCOL)

def pairwise_sim(fs):
  from scipy.sparse import lil_matrix
  print('Processing:', fs)
  root = './archive_filter_lang/'
  pair_count = lil_matrix((len(isbn2ind), len(isbn2ind)))
  for fn in fs:
    with open(root + fn, 'rb') as r:
      pair2count = pickle.load(r)
    for pair in tqdm(pair2count):
      if len(pair) < 2:
        continue
      isbn1, isbn2 = pair
      if isbn1 not in isbn2ind or isbn2 not in isbn2ind:
        continue
      pair_count[isbn2ind[isbn1], isbn2ind[isbn2]] += pair2count[pair]
      pair_count[isbn2ind[isbn2], isbn2ind[isbn1]] += pair2count[pair]
  return pair_count

def sim():
  fns = [['same_author_pair.pickle'],
         ['same_shelf_pair.pickle'],
         ['book_similar_pair.pickle'],
         ['same_rating_children_{}.pickle'.format(str(i)) for i in range(5)],
         ['same_rating_comics_graphic_{}.pickle'.format(str(i)) for i in range(6)],
         ['same_rating_poetry.pickle'],
         ['same_rating_fantasy_paranormal_{}.pickle'.format(str(i)) for i in range(24)],
         ['same_rating_history_biography_{}.pickle'.format(str(i)) for i in range(24)],
         ['same_rating_mystery_thriller_crime_{}.pickle'.format(str(i)) for i in range(24)],
         ['same_rating_romance_{}.pickle'.format(str(i)) for i in range(24)],
         ['same_rating_young_adult_{}.pickle'.format(str(i)) for i in range(24)]]
  names = ['same_author_sim', 'same_shelf_sim', 'similar_pair_sim', 'same_rating_children_sim', 'same_rating_comics_graphic_sim', 'same_rating_poetry_sim', 'same_rating_fantasy_paranormal_sim', 'same_rating_history_biography_sim', 'same_rating_mystery_thriller_crime_sim', 'same_rating_romance_sim', 'same_rating_young_adult_sim']
  i = 0
  for n, fs in zip(names, fns):
    if i < 9:
      i += 1
      continue
    print(n)
    print(fs)
    print(i)
    matrix = pairwise_sim(fs)
    # result.append(matrix)
    with open('{}.pickle'.format(n), 'wb') as f:
      pickle.dump(matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    i += 1


# pair_sparse_matrix()

# def pairwise_dist(v):
#   return np.sum((emb - v) ** 2, axis=1) ** 0.5
#

# emb = np.load('embedding.npy')
with open('isbn2ind.pickle', 'rb') as f:
  isbn2ind = pickle.load(f)
ind2isbn = {isbn2ind[isbn]:isbn for isbn in isbn2ind}
#
# print(isbn2ind)
# sim()

root = './sim_score/'
columns = ['same_author', 'same_rating_children', 'same_rating_comics', 'same_rating_fantasy', 'same_rating_history', 'same_rating_mystery', 'same_rating_poetry', 'same_rating_romance', 'same_rating_young_adult', 'same_shelf', 'similar_pair']

import os
with open('sim_pair_scores.csv', 'w') as w:
  for i, fn in enumerate(os.listdir(root)):
    fn = root + fn
    print(fn)
    with open(fn, 'rb') as f:
      data = pickle.load(f)
    # print('mean:', (data.mean() * 97000**2)/(data.count_nonzero()))
    xs, ys = data.nonzero()
    for x, y in zip(xs, ys):
      if y > x:
        isbn1 = ind2isbn[x]
        isbn2 = ind2isbn[y]
        scores = ['0'] * len(columns)
        scores[i] = str(data[x, y])
        line = [isbn1, isbn2] + scores
        w.write(','.join(line) + '\n')




#
#
# isbn = list(isbn2ind.keys())[1]
# print('isbn:', isbn)
#
# weight_name = ['author', 'genre', 'goodread_similar', 'similar_interest']
# weight_score = [7, 7, 3, 8]
# weight_score = np.array(weight_score)/np.sum(weight_score)
#
# weight = sim(isbn)
# final_weight = np.dot(np.transpose(weight), weight_score)
# print(final_weight.shape)
# print(emb.shape)
# dist = pairwise_dist(emb[isbn2ind[isbn]])
# print(np.argsort(dist))

# for commit


'''
training contrastive

apply transformations to embedding space of the books,
takes into account the pairwise euclidena distnace between points,
similar author pairs
similar rataing pirs
simialr shelf pairs
similar book pairs


'''

