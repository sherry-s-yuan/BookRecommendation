import os
import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz


lemmatizer = WordNetLemmatizer()
stop_word = stopwords.words('english')


root = 'E:/Toolkit/Backup/book_data/'

def book2rating1(root):
  '''parse book2rating.books.csv file'''
  path = root + 'book2rating/books.csv'
  data = pd.read_csv(path)
  columns = ['isbn13', 'authors', 'title', 'language_code', 'average_rating', 'ratings_count', 'work_text_reviews_count', 'ratings_1','ratings_2','ratings_3','ratings_4','ratings_5']
  data = data[columns]
  data.dropna(subset=['isbn13'], inplace=True)
  data = data.replace(np.nan, '', regex=True)
  data.drop_duplicates(subset=['isbn13'], inplace=True)
  data = data[data['language_code'].str.contains("en") | data['language_code'].str.match('')]
  # print(data)
  isbn2rating = {isbn: rating for isbn, rating in zip(data['isbn13'], data['average_rating'])}
  with open('isbn2rating1.pickle', 'wb') as f:
    pickle.dump(isbn2rating, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(isbn2rating)
  del isbn2rating

  isbn2title = {isbn: title for isbn, title in zip(data['isbn13'], data['title'])}
  with open('isbn2title1.pickle', 'wb') as f:
    pickle.dump(isbn2title, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(isbn2title)
  del isbn2title

  isbn2ratings_count = {isbn: ratings_count for isbn, ratings_count in zip(data['isbn13'], data['ratings_count'])}
  with open('isbn2ratings_count1.pickle', 'wb') as f:
    pickle.dump(isbn2ratings_count, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(isbn2ratings_count)
  del isbn2ratings_count

  isbn2reviews_count = {isbn: reviews_count for isbn, reviews_count in zip(data['isbn13'], data['work_text_reviews_count'])}
  with open('isbn2reviews_count1.pickle', 'wb') as f:
    pickle.dump(isbn2reviews_count, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(isbn2reviews_count)
  del isbn2reviews_count


  isbn2rating_distribution = {isbn: [ratings_1, ratings_2, ratings_3, ratings_4, ratings_5] for isbn, ratings_1, ratings_2, ratings_3, ratings_4, ratings_5 in zip(data['isbn13'], data['ratings_1'],data['ratings_2'],data['ratings_3'],data['ratings_4'],data['ratings_5'])}
  with open('isbn2rating_distribution1.pickle', 'wb') as f:
    pickle.dump(isbn2rating_distribution, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(isbn2rating_distribution)
  del isbn2rating_distribution

def book2rating2(root):
  '''parse book2rating.books.csv file'''
  path = root + 'book2rating/books2.csv'
  data = pd.read_csv(path)
  columns = ['isbn13', 'authors', 'title', 'average_rating', 'ratings_count', 'text_reviews_count', '  num_pages']
  data = data[columns]
  data = data.replace(np.nan, '', regex=True)
  data.dropna(subset=['isbn13'], inplace=True)
  data.drop_duplicates(subset=['isbn13'], inplace=True)
  # data = data[data['language_code'].str.contains("en") | data['language_code'].str.match('')]
  # print(data)
  isbn2ratings_count = {isbn: ratings_count for isbn, ratings_count in zip(data['isbn13'], data['ratings_count'])}
  with open('isbn2ratings_count2.pickle', 'wb') as f:
    pickle.dump(isbn2ratings_count, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(isbn2ratings_count)
  del isbn2ratings_count

  isbn2reviews_count = {isbn: reviews_count for isbn, reviews_count in
                        zip(data['isbn13'], data['text_reviews_count'])}
  with open('isbn2reviews_count2.pickle', 'wb') as f:
    pickle.dump(isbn2reviews_count, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(isbn2reviews_count)
  del isbn2reviews_count

  isbn2rating = {isbn: rating for isbn, rating in zip(data['isbn13'], data['average_rating'])}
  with open('isbn2rating2.pickle', 'wb') as f:
    pickle.dump(isbn2rating, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(isbn2rating)
  del isbn2rating

  isbn2title = {isbn: title for isbn, title in zip(data['isbn13'], data['title'])}
  with open('isbn2title2.pickle', 'wb') as f:
    pickle.dump(isbn2title, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(isbn2title)
  del isbn2title

  isbn2num_pages = {isbn: num_pages for isbn, num_pages in zip(data['isbn13'], data['  num_pages'])}
  with open('isbn2num_pages2.pickle', 'wb') as f:
    pickle.dump(isbn2num_pages, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(isbn2num_pages)
  del isbn2num_pages


def book2rating3(root):
  '''parse book2rating.books.csv file'''
  path = root + 'book2rating/BX-Book-Ratings.csv'
  data = pd.read_csv(path, delimiter=';', encoding='latin1')
  columns = ['ISBN', 'Book-Rating']
  data = data[columns]
  data = data.replace(np.nan, '', regex=True)
  data.dropna(subset=['ISBN'], inplace=True)
  isbn2rank_distribution = {}
  for isbn, rating in zip(data['ISBN'], data['Book-Rating']):
    if isbn not in isbn2rank_distribution:
      isbn2rank_distribution[isbn] = [0]*11
    isbn2rank_distribution[isbn][int(rating)] += 1
  rating_distribution = list(isbn2rank_distribution.values())
  ratings_count_arr = np.array(rating_distribution).sum(axis=1).tolist()
  average_rating_arr = np.dot(np.array(rating_distribution), np.array([0,1,2,3,4,5,6,7,8,9,10])).tolist()
  print(np.array(rating_distribution))
  ratings_count = dict(list(zip(list(isbn2rank_distribution.keys()), ratings_count_arr)))
  average_rating = dict(list(zip(list(isbn2rank_distribution.keys()), average_rating_arr)))

  with open('isbn2rating_distribution3.pickle', 'wb') as f:
    pickle.dump(rating_distribution, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(rating_distribution)
  del rating_distribution

  with open('isbn2rating3.pickle', 'wb') as f:
    pickle.dump(average_rating, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(average_rating)
  del average_rating

  with open('isbn2ratings_count3.pickle', 'wb') as f:
    pickle.dump(ratings_count, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(ratings_count)
  del ratings_count

def book2rating4(root):
  import json
  from tqdm import tqdm
  with open('isbn2title4.pickle', 'rb') as f:
    isbn2title = pickle.load(f)
  path = root + 'book2rating/goodreads_books.json'
  columns = ['isbn13','text_reviews_count','average_rating', 'description', 'num_pages', 'ratings_count', 'title', 'book_id']
  data = []
  ebook_count = 0
  total_count = 0
  # isbn2authors = {}
  isbn2popular_shelves = {}
  # isbn2description = {}
  isbn2similar_books = {}
  with open(path, 'r') as f:
    for line in tqdm(f):
      line = json.loads(line)
      isbn13 = line['isbn13']
      # line_items = [line[item] for item in columns]
      total_count += 1
      if isbn13 != '' and isbn13 in isbn2title:
        # data.append(line_items)
        # isbn2authors[isbn13] = [item['author_id'] for item in line['authors']]
        # isbn2popular_shelves[isbn13] = [(item['name'], int(item['count'])) for item in line['popular_shelves']  if int(item['count']) > 1]
        # isbn2description[isbn13] = line['description'].lower()
        isbn2similar_books[isbn13] = line['similar_books']
        # print(isbn2similar_books[isbn13])
        # print(line.keys())
        # print(line)
      else:
        ebook_count +=1
  # print('num ebook', ebook_count)
  # print('total_count', total_count)
  # data = pd.DataFrame(data, columns=columns)
  # print(data)
  #
  # data.dropna(subset=['isbn13'], inplace=True)
  # data = data.replace(np.nan, '', regex=True)
  # data.drop_duplicates(subset=['isbn13'], inplace=True)

  # with open('isbn2popular_shelves4.pickle', 'wb') as f:
  #   pickle.dump(isbn2popular_shelves, f, protocol=pickle.HIGHEST_PROTOCOL)
  # print(list(isbn2popular_shelves.items())[:10])
  # del isbn2popular_shelves

  # with open('isbn2description4.pickle', 'wb') as f:
  #   pickle.dump(isbn2description, f, protocol=pickle.HIGHEST_PROTOCOL)
  # print(list(isbn2description.items())[:10])
  # del isbn2description


  with open('isbn2similar_books4.pickle', 'wb') as f:
    pickle.dump(isbn2similar_books, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(list(isbn2similar_books.items())[:10])
  del isbn2similar_books

  # isbn2rating = {isbn: rating for isbn, rating in zip(data['isbn13'], data['average_rating'])}
  # with open('isbn2rating4.pickle', 'wb') as f:
  #   pickle.dump(isbn2rating, f, protocol=pickle.HIGHEST_PROTOCOL)
  # print(isbn2rating)
  # del isbn2rating
  #
  # isbn2title = {isbn: title for isbn, title in zip(data['isbn13'], data['title'])}
  # with open('isbn2title4.pickle', 'wb') as f:
  #   pickle.dump(isbn2title, f, protocol=pickle.HIGHEST_PROTOCOL)
  # print(isbn2title)
  # del isbn2title
  #
  # isbn2ratings_count = {isbn: ratings_count for isbn, ratings_count in zip(data['isbn13'], data['ratings_count'])}
  # with open('isbn2ratings_count4.pickle', 'wb') as f:
  #   pickle.dump(isbn2ratings_count, f, protocol=pickle.HIGHEST_PROTOCOL)
  # print(isbn2ratings_count)
  # del isbn2ratings_count

  # isbn2reviews_count = {isbn: reviews_count for isbn, reviews_count in
  #                       zip(data['isbn13'], data['text_reviews_count'])}
  # with open('isbn2reviews_count4.pickle', 'wb') as f:
  #   pickle.dump(isbn2reviews_count, f, protocol=pickle.HIGHEST_PROTOCOL)
  # print(isbn2reviews_count)
  # del isbn2reviews_count
  #
  #
  # with open('isbn2authors4.pickle', 'wb') as f:
  #   pickle.dump(isbn2authors, f, protocol=pickle.HIGHEST_PROTOCOL)
  # print(isbn2authors)
  # del isbn2authors

  # isbn2book_id = {isbn: book_id for isbn, book_id in
  #                       zip(data['isbn13'], data['book_id'])}
  # with open('isbn2book_id.pickle', 'wb') as f:
  #   pickle.dump(isbn2book_id, f, protocol=pickle.HIGHEST_PROTOCOL)
  # print(isbn2book_id)
  # del isbn2book_id


def combine_ratings():
  import pyisbn
  with open('isbn2rating4.pickle', 'rb') as f:
    data = pickle.load(f)
  # not_valid = 0
  # not_in_region = 0
  # for isbn in data:
  #   try:
  #     if not pyisbn.validate(isbn):
  #       not_valid += 1
  #   except:
  #     not_in_region += 1
  # print(not_valid/len(data))
  # print(not_in_region/len(data))
  print(len(data))
  pop_key = []
  for isbn in data:
    if data[isbn] == '':
      pop_key.append(isbn)
  for key in pop_key:
    data.pop(key, None)
  for isbn in data:
    data[isbn] = float(data[isbn])
  print(len(data))
  with open('isbn2rating3.pickle', 'rb') as f:
    data2 = pickle.load(f)
  print(len(data))
  for isbn in data2:
    orig_isbn = isbn
    if not isinstance(isbn, str):
      isbn = int(isbn)
      isbn = str(isbn)
    if len(isbn) == 10 or len(isbn) == 13:
      if len(str(isbn)) != 13:
        try:
          isbn13 = pyisbn.convert(isbn)
        except:
          continue
      else: isbn13 = isbn
    else:
      continue
    if isbn13 not in data and len(str(data2[orig_isbn])) != 0:
      data[isbn13] = float(data2[orig_isbn])
  print(len(data))
  with open('isbn2rating4.pickle', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def combine_title():
  import pyisbn
  with open('isbn2title4.pickle', 'rb') as f:
    data = pickle.load(f)
  print(len(data))
  pop_key = []
  for isbn in data:
    if data[isbn] == '':
      pop_key.append(isbn)
  for key in pop_key:
    data.pop(key, None)
  print(len(data))
  with open('isbn2title2.pickle', 'rb') as f:
    data2 = pickle.load(f)
  print(len(data))
  for isbn in data2:
    orig_isbn = isbn
    if not isinstance(isbn, str):
      isbn = int(isbn)
      isbn = str(isbn)
    if len(isbn) == 10 or len(isbn) == 13:
      if len(str(isbn)) != 13:
        try:
          isbn13 = pyisbn.convert(isbn)
        except:
          continue
      else: isbn13 = isbn
    else:
      continue
    if isbn13 not in data and len(str(data2[orig_isbn])) != 0:
      data[isbn13] = data2[orig_isbn]
  print(len(data))
  with open('isbn2title4.pickle', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def combine_rating_count():
  import pyisbn
  with open('isbn2ratings_count4.pickle', 'rb') as f:
    data = pickle.load(f)
  print(len(data))
  pop_key = []
  for isbn in data:
    if data[isbn] == '':
      pop_key.append(isbn)
  for key in pop_key:
    data.pop(key, None)
  for isbn in data:
    data[isbn] = int(data[isbn])

  print(len(data))
  with open('isbn2ratings_count3.pickle', 'rb') as f:
    data2 = pickle.load(f)
  print(len(data))
  for isbn in data2:
    orig_isbn = isbn
    if not isinstance(isbn, str):
      isbn = int(isbn)
      isbn = str(isbn)
    if len(isbn) == 10 or len(isbn) == 13:
      if len(str(isbn)) != 13:
        try:
          isbn13 = pyisbn.convert(isbn)
        except:
          continue
      else:
        isbn13 = isbn
    else:
      continue
    if isbn13 not in data and len(str(data2[orig_isbn])) != 0:
      data[isbn13] = int(data2[orig_isbn])
  print(len(data))
  with open('isbn2ratings_count4.pickle', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def combine_review_count():
  import pyisbn
  with open('isbn2reviews_count4.pickle', 'rb') as f:
    data = pickle.load(f)
  print(len(data))
  pop_key = []
  for isbn in data:
    if data[isbn] == '':
      pop_key.append(isbn)
  for key in pop_key:
    data.pop(key, None)
  for isbn in data:
    data[isbn] = int(data[isbn])

  print(len(data))
  with open('isbn2reviews_count2.pickle', 'rb') as f:
    data2 = pickle.load(f)
  print(len(data))
  for isbn in data2:
    orig_isbn = isbn
    if not isinstance(isbn, str):
      isbn = int(isbn)
      isbn = str(isbn)
    if len(isbn) == 10 or len(isbn) == 13:
      if len(str(isbn)) != 13:
        try:
          isbn13 = pyisbn.convert(isbn)
        except:
          continue
      else:
        isbn13 = isbn
    else:
      continue
    if isbn13 not in data and len(str(data2[orig_isbn])) != 0:
      data[isbn13] = int(data2[orig_isbn])
  print(len(data))
  with open('isbn2reviews_count4.pickle', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# book2rating1(root)
# book2rating2(root)
# book2rating3(root)
# book2rating4(root)

# combine_ratings()
# combine_title()
# combine_rating_count()
# combine_review_count()

def filter_keys(f1, f2):
  with open(f1, 'rb') as f:
    isbn2title = pickle.load(f)

  with open(f2, 'rb') as f:
    isbn2rating = pickle.load(f)

  exist_isbn = set(isbn2title.keys()).intersection(set(isbn2rating.keys()))
  title_pop_key = set()
  print(len(isbn2title))
  print(len(isbn2rating))
  print(len(exist_isbn))
  for isbn in isbn2title:
    if isbn not in exist_isbn:
      title_pop_key.add(isbn)
  for isbn in isbn2rating:
    if isbn not in exist_isbn:
      title_pop_key.add(isbn)
  for isbn in title_pop_key:
    isbn2title.pop(isbn, None)
    isbn2rating.pop(isbn, None)
  print(len(isbn2title))
  print(len(isbn2rating))
  # with open(f1, 'wb') as f:
  #   pickle.dump(isbn2title, f, protocol=pickle.HIGHEST_PROTOCOL)
  # with open(f2, 'wb') as f:
  #   pickle.dump(isbn2rating, f, protocol=pickle.HIGHEST_PROTOCOL)


# filter_keys('isbn2title4.pickle', 'isbn2reviews_count4.pickle')

def get_title():
  import requests
  from tqdm import tqdm
  from pprint import pprint
  with open('isbn2title4.pickle', 'rb') as f:
    isbn2title = pickle.load(f)
  with open('isbn2rating4.pickle', 'rb') as f:
    isbn2rating = pickle.load(f)
  print(len(isbn2title))
  print(len(isbn2rating))

  for isbn in tqdm(isbn2rating):
    if isbn in isbn2title:
      continue
    try:
      # result = requests.get('https://www.googleapis.com/books/v1/volumes?q=isbn:{}'.format(isbn)).json()
      result = requests.get('https://openlibrary.org/api/books?bibkeys=ISBN:{}&jscmd=data&format=json'.format(isbn)).json()
      # pprint(result)
      if len(result) == 0:
        continue
      key1 = 'ISBN:{}'.format(isbn)
      title = result[key1]['title']
    except:
      continue
    isbn2title[isbn] = title
    print(title)
  with open('isbn2title4.pickle', 'wb') as f:
    pickle.dump(isbn2title, f, protocol=pickle.HIGHEST_PROTOCOL)

import re
import string

def text_clean(text):
  text = text.lower().strip()
  text = re.sub(r'\d +', '', text)
  text = text.translate(str.maketrans('', '', string.punctuation))
  return text

import nltk
def isbn2title_concreteness():
  from tqdm import tqdm
  with open('word2concreteness.pickle', 'rb') as f:
    word2concreteness = pickle.load(f)

  # try:
  #   with open('isbn2title_concreteness.pickle', 'rb') as f:
  #     isbn2title_c = pickle.load(f)
  # except:
  #   import traceback
  #   traceback.print_exc()
  #   isbn2title_c = {}
  isbn2title_c = {}

  with open('isbn2description4.pickle', 'rb') as f:
    isbn2title = pickle.load(f)

  count = 0
  print(len(isbn2title))

  # vocab_emb = np.array([model[w] if w in model else np.zeros_like(model['nice']) for w in word2concreteness])
  # isbn = [isbn for isbn in isbn2title if isbn not in isbn2title_c]
  for isbn in tqdm(isbn2title):
    # if isbn in isbn2title_c:
    #   continue
    title = isbn2title[isbn]
    if len(title) == 0:
      continue
    # print(title)
    # print(isbn2title_c[isbn])
    words = [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text_clean(title)) if w not in stop_word]
    # word_emb = np.array([model[w] for w in words if w in model])
    # if len(word_emb) == 0:
    #   count += 1
    #   continue
    # words = np.array(list(word2concreteness.keys()))[np.argmin(euclidean_distances(word_emb, vocab_emb), axis=1)].tolist()
    c = [word2concreteness[w] for w in words if w in word2concreteness]
    if len(c) == 0:
      count += 1
      continue
    isbn2title_c[isbn] = sum(c)/len(c)
  print('num without title concreteness:', count)
  with open('isbn2description_concreteness.pickle', 'wb') as f:
    pickle.dump(isbn2title_c, f, protocol=pickle.HIGHEST_PROTOCOL)

import gensim
import gensim.corpora as corpora
from gensim import models
from gensim.utils import simple_preprocess

def sent_to_words(sentences):
  for sentence in tqdm(sentences):
    yield (simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
  return [[word for word in simple_preprocess(str(doc)) if word not in stop_word] for doc in tqdm(texts)]

def filter_pos(sentences, allowed_postags=['NN', 'VB', 'RB', 'JJ']):
  """https://spacy.io/api/annotation"""
  from tqdm import tqdm
  texts_out = []
  print(len(sentences))
  for sent in tqdm(sentences):
    pos_tagged = nltk.pos_tag(sent)
    texts_out.append([lemmatizer.lemmatize(word.lower()) for word, tag in pos_tagged if sum([pos in tag for pos in allowed_postags]) > 0])
  return texts_out

count = 0
def fasttext_emb_description(descriptions, title):
  global count
  with open('fasttext.pickle', 'rb') as f:
    model = pickle.load(f)
  for d, t in tqdm(zip(descriptions, title)):
    emb = [model[w] for w in d+t if w in model]
    if len(emb) > 0:
      yield np.mean(emb, axis=0)
    else:
      yield np.zeros((300))
      count += 1



def isbn2sbert_embedding():
  from gensim.utils import simple_preprocess
  # from sentence_transformers import SentenceTransformer
  # model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
  with open('isbn2description4.pickle', 'rb') as f:
    isbn2description = pickle.load(f)
  with open('isbn2title4.pickle', 'rb') as f:
    isbn2title = pickle.load(f)
  isbns = [isbn for isbn in isbn2description if isbn2description[isbn] != '']
  # chunk = 50
  # chunk_size = len(isbns) // chunk
  # start = 0
  # isbns = isbns[chunk_size*start: chunk_size*start + chunk_size]
  # isbns = isbns[:200]
  descriptions = [text_clean(isbn2description[isbn]) for isbn in tqdm(isbns)]
  titles = [text_clean(isbn2title[isbn]) for isbn in tqdm(isbns)]
  del isbn2description
  del isbn2title
  print(len(isbns))

  print('start')
  # descriptions_emb = model.encode(descriptions)
  try:
    with open('clean_description.pickle', 'rb') as f:
      descriptions = pickle.load(f)
    print(descriptions[:10])
  except:
    descriptions = list(sent_to_words(descriptions))
    descriptions = remove_stopwords(descriptions)
    descriptions = filter_pos(descriptions)
    with open('clean_description.pickle', 'wb') as f:
      pickle.dump(descriptions, f, protocol=pickle.HIGHEST_PROTOCOL)

  try:
    with open('clean_title.pickle', 'rb') as f:
      titles = pickle.load(f)
    print(titles[:10])
  except:
    titles = list(sent_to_words(titles))
    titles = remove_stopwords(titles)
    titles = filter_pos(titles)
    with open('clean_title.pickle', 'wb') as f:
      pickle.dump(titles, f, protocol=pickle.HIGHEST_PROTOCOL)

  descriptions_emb = list(fasttext_emb_description(descriptions, titles))
  descriptions_emb = np.array(descriptions_emb)
  global count
  print('num with out embedding', count)

  isbns = np.array(isbns)
  print(isbns.shape)
  print(len(descriptions_emb))
  np.savez_compressed('isbn2embedding', isbn=isbns, embedding=descriptions_emb)
  # result = np.hstack((isbns[:, np.newaxis], descriptions_emb))
  # print(result.shape)
  del descriptions
  del titles
  # data = pd.DataFrame(result)
  # print('end')
  # data.to_csv('isbn2embedding.csv', index=False, header=False)
  # data.to_csv('isbn2embedding_{}.csv'.format(str(start)), header=False)

# isbn2sbert_embedding()

def isbn2emb_compress():
  from sklearn.decomposition import PCA
  data = np.load('isbn2embedding.npz')
  X = data['embedding']
  # isbn2embedding = pd.read_csv('isbn2embedding.csv', header=None)
  # X = isbn2embedding.iloc[:, 1:].to_numpy()
  pca = PCA(n_components=2)
  result = pca.fit_transform(X)
  del X
  isbns = data['isbn']
  mapping = dict(zip(isbns, result))
  print(mapping)
  with open('isbn2emb_compress.pickle', 'wb') as f:
    pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
  #
  # # isbns = isbn2embedding.iloc[:, 1].to_numpy()
  # data = np.hstack((isbns[:, np.newaxis], result))
  # data = pd.DataFrame(data, columns=['isbn', 'x', 'y'])
  # data.to_csv('isbn2emb_compress.csv', header=False)

# isbn2emb_compress()
# isbn2title_concreteness()

def data2csv():
  with open('isbn2title4.pickle', 'rb') as f:
    isbn2title = pickle.load(f)
  with open('isbn2rating4.pickle', 'rb') as f:
    isbn2rating = pickle.load(f)
  with open('isbn2ratings_count4.pickle', 'rb') as f:
    isbn2ratings_count = pickle.load(f)
  with open('isbn2reviews_count4.pickle', 'rb') as f:
    isbn2reviews_count = pickle.load(f)
  with open('isbn2description4.pickle', 'rb') as f:
    isbn2description = pickle.load(f)
  with open('isbn2emb_compress.pickle', 'rb') as f:
    isbn2emb = pickle.load(f)
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)


  column = ['isbn', 'title', 'rating', 'num_raing', 'num_review', 'x', 'y', 'description']
  isbns = set(list(isbn2ind.keys()))
  isbns = isbns.intersection(set(list(isbn2rating.keys())))
  # isbns = isbns.intersection(set(list(isbn2title_c.keys())))
  isbns = isbns.intersection(set(list(isbn2ratings_count.keys())))
  isbns = isbns.intersection(set(list(isbn2reviews_count.keys())))
  isbns = isbns.intersection(set(list(isbn2title.keys())))
  # isbns = isbns.intersection(set(list(isbn2description_concreteness.keys())))
  isbns = isbns.intersection(set(list(isbn2emb.keys())))

  print('num emb', len(list(isbn2emb.keys())))

  print('num observation:', len(isbns))
  # isbns = sorted(list(isbns))
  # isbn2ind = {isbn:i for i,isbn in enumerate(isbns)}
  # print(isbn2ind)
  #
  # np.save('isbn', isbns)
  #
  # with open('isbn2ind.pickle', 'wb') as f:
  #   pickle.dump(isbn2ind, f, protocol=pickle.HIGHEST_PROTOCOL)
  #
  data = []
  for isbn in isbns:
    data.append([isbn, isbn2title[isbn], isbn2rating[isbn], isbn2ratings_count[isbn], isbn2reviews_count[isbn]] + isbn2emb[isbn].tolist() + [isbn2description[isbn]])

  data = pd.DataFrame(data, columns=column)
  data.to_csv('book_stats.csv', index=False)


def EDA():
  from scipy.stats import pearsonr
  data = pd.read_csv('book_stats.csv')
  # r, p = pearsonr(data['num_review'].tolist(), data['description_concreteness'].tolist())
  # print(r, p)
  plt.hist(data['num_review'].tolist(), bins=np.arange(0, 3000, 10))
  plt.show()

# data2csv()
#prototype contrastive learning: https://blog.einstein.ai/prototypical-contrastive-learning-pushing-the-frontiers-of-unsupervised-learning/



# EDA()
# get_title()

def clean_emb():
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)
  isbns2 = list(isbn2ind.keys())
  print(isbns2)
  data = np.load('isbn2embedding.npz')
  X = data['embedding']
  print(X.shape)
  isbns = data['isbn'].tolist()
  isbn2ind = {isbn:i for i,isbn in enumerate(isbns)}
  X2 = []
  for i in tqdm(range(len(isbns2))):
    isbn = isbns2[i]
    X2.append(X[isbn2ind[isbn]])
  X2 = np.array(X2)
  print(X2.shape)
  print(X2[100])
  print(X[isbn2ind[isbns2[100]]])
  np.save('embedding', X2)

# clean_emb()

def clean_rating():
  with open('isbn2reviews_count4.pickle', 'rb') as f:
    isbn2rating = pickle.load(f)
  isbns2 = np.load('isbn.npy')
  ratings = []
  for isbn in isbns2:
    ratings.append(isbn2rating[isbn])
  print(ratings)
  np.save('review_count', np.array(ratings))

def similar_book_pair():
  with open('isbn2book_id.pickle', 'rb') as f:
    isbn2book_id = pickle.load(f)
  with open('isbn2similar_books4.pickle', 'rb') as f:
    isbn2similar_books = pickle.load(f)
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)
  isbns = np.load('isbn.npy')
  similar_book_pairs = {}
  bookid2isbn = {book_id:isbn for isbn, book_id in isbn2book_id.items()}
  # isbn2similar_books = {isbn:[bookid2isbn[book_id] for book_id in isbn2similar_books[isbn] if
  #  book_id in bookid2isbn] for isbn in tqdm(isbn2similar_books)}
  # print(isbn2similar_books)
  # with open('isbn2similar_books4.pickle', 'wb') as f:
  #   pickle.dump(isbn2similar_books, f, protocol=pickle.HIGHEST_PROTOCOL)

  for i in tqdm(range(len(isbns))):
    isbn1 = isbns[i]
    sim_b = isbn2similar_books[isbn1]
    # print(sim_b)
    for isbn2 in sim_b:
      if isbn2 in isbn2ind:
        pair = frozenset([isbn1, isbn2])
        if pair not in similar_book_pairs:
          similar_book_pairs[pair] = 0
        similar_book_pairs[pair] += 1
        # print(pair, similar_book_pairs[pair])
        # similar_book_pairs[i][isbn2ind[isbn2]] += 1
  print(list(similar_book_pairs.items())[10:])
  with open('book_similar_pair.pickle', 'wb') as f:
    pickle.dump(similar_book_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)


def same_author_pair():
  import itertools
  with open('isbn2authors4.pickle', 'rb') as f:
    isbn2authors = pickle.load(f)
  similar_book_pairs = {}
  author2isbn = {}
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)

  for isbn, anthors in tqdm(isbn2authors.items()):
    for author, isbn in itertools.product(anthors, [isbn]):
      if author not in author2isbn:
        author2isbn[author] = []
      if isbn in isbn2ind:
        author2isbn[author].append(isbn)
  print(list(author2isbn.items())[:10])
  for isbns in tqdm(list(author2isbn.values())):
    if len(isbns) > 40:
      continue
    for isbn1, isbn2 in itertools.combinations(isbns, 2):
      pair = frozenset([isbn1, isbn2])
      if pair not in similar_book_pairs:
        similar_book_pairs[pair] = 0
      similar_book_pairs[pair] += 1
  print(list(similar_book_pairs.items())[:10])
  with open('same_author_pair.pickle', 'wb') as f:
    pickle.dump(similar_book_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)


def same_shelf_pair():
  import itertools
  from scipy.sparse import lil_matrix
  with open('isbn2popular_shelves4.pickle', 'rb') as f:
    isbn2popular_shelves = pickle.load(f)
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)
  with open('isbn2ratings_count4.pickle', 'rb') as f:
    isbn2ratings_count = pickle.load(f)

  isbns = np.load('isbn.npy')
  similar_book_pairs = {}
  # similar_book_pairs = lil_matrix((len(isbns), len(isbns)))
  print(similar_book_pairs)
  shelves2isbn = {}
  for isbn in tqdm(isbns):
    if isbn not in isbn2ind or isbn2ratings_count[isbn] < 3000:
      continue
    # print('rating:', isbn2ratings_count[isbn])
    shelves = [shelf for shelf, count in isbn2popular_shelves[isbn][1:11]]
    for shelf in shelves:
      if shelf not in shelves2isbn:
        shelves2isbn[shelf] = []
      shelves2isbn[shelf].append(isbn)
  del isbn2ratings_count
  del isbn2ind
  del isbn2popular_shelves

  for shelf, isbns in tqdm(list(shelves2isbn.items())):
    # if len(isbns) < 200:
    #   print(shelf, len(isbns))
    if len(isbns) < 150 or len(isbns) > 3000:
      continue
    for isbn1, isbn2 in itertools.combinations(isbns, 2):
      # similar_book_pairs[isbn2ind[isbn1], isbn2ind[isbn2]] += 1
      # similar_book_pairs[isbn2ind[isbn2], isbn2ind[isbn1]] += 1
      pair = frozenset([isbn1, isbn2])
      if pair not in similar_book_pairs:
        similar_book_pairs[pair] = 0
      similar_book_pairs[pair] += 1

  # print(list(similar_book_pairs.items())[:10])
  with open('same_shelf_pair.pickle', 'wb') as f:
    pickle.dump(similar_book_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

def parse_json_gz(path):
  import gzip
  import json
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l.decode("utf-8"))

def prepare_similar_rating_pair():
  from tqdm import tqdm
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)
  with open('isbn2book_id.pickle', 'rb') as f:
    isbn2book_id = pickle.load(f)
  bookid2isbn = {book_id:isbn for isbn, book_id in isbn2book_id.items()}
  # book_type = 'romance'
  book_type = 'poetry'
  # path = root + 'user_book_interaction/goodreads_interactions_children.json.gz'
  path = root + 'user_book_interaction/goodreads_interactions_{}.json.gz'.format(book_type)
  user2rating_isbn = {}
  for line in tqdm(parse_json_gz(path)):
    if not line['is_read']:
      continue
    # print(line)
    rating = line['rating']
    if rating == 0:
      continue
    user_id = line['user_id']
    book_id = line['book_id']
    if book_id not in bookid2isbn:
      continue
    isbn = bookid2isbn[book_id]
    if isbn not in isbn2ind:
      continue
    if user_id not in user2rating_isbn:
      user2rating_isbn[user_id] = {}
    if rating not in user2rating_isbn[user_id]:
      user2rating_isbn[user_id][rating] = []
    user2rating_isbn[user_id][rating].append(isbn)

  user_rating_pair = []
  for user in tqdm(user2rating_isbn):
    for rating in user2rating_isbn[user]:
      if len(user2rating_isbn[user][rating]) < 2:
        user_rating_pair.append((user, rating))
  for user, rating in user_rating_pair:
    user2rating_isbn[user].pop(rating, None)
  del_user = []
  for user in tqdm(user2rating_isbn):
    if len(list(user2rating_isbn[user].keys())) == 0:
      del_user.append(user)
  print(user2rating_isbn)

  for user in del_user:
    user2rating_isbn.pop(user, None)
  print(user2rating_isbn)

  with open('user2rating_isbn_{}.pickle'.format(book_type), 'wb') as f:
    pickle.dump(user2rating_isbn, f, protocol=pickle.HIGHEST_PROTOCOL)


def similar_rating_pair(chunk, start, book_type):
  import itertools

  with open('user2rating_isbn_{}.pickle'.format(book_type), 'rb') as f:
    user2rating_isbn = pickle.load(f)
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)

  similar_book_pairs = {}
  users = list(user2rating_isbn.keys())
  chunk_size = len(users)//chunk
  for user in tqdm(users[start*chunk_size: start*chunk_size+chunk_size]):
    for rating in user2rating_isbn[user]:
      if rating == 3:
        continue
      books = [isbn for isbn in user2rating_isbn[user][rating] if isbn in isbn2ind]
      if len(books) > 2000:
        continue
      for isbn1, isbn2 in itertools.combinations(books, 2):
        pair = frozenset([isbn1, isbn2])
        if pair not in similar_book_pairs:
          similar_book_pairs[pair] = 0
        similar_book_pairs[pair] += 1
  with open('same_rating_{}_{}.pickle'.format(book_type, str(start)), 'wb') as f:
    pickle.dump(similar_book_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

# chunk = 24
# # books = ['children', 'fantasy_paranormal', 'history_biography', 'mystery_thriller_crime', 'romance', 'young_adult']
# books = ['children']
# for book_type in books:
#   for start in range(0, chunk):
#     similar_rating_pair(chunk, start, book_type = book_type)

def isbn2language():
  from langdetect import detect
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)
  with open('isbn2description4.pickle', 'rb') as f:
    isbn2description = pickle.load(f)
  isbn2lang = {}
  count = 0
  num_eng = 0
  for isbn in tqdm(isbn2ind):
    try:
        lang = detect(isbn2description[isbn])
    except:
      count += 1
      continue
    # print('Title', title)
    # print('Description:', description)
    # print(lang)
    isbn2lang[isbn] = lang
  print('Num skipped:', count)
  print('Num eng:', num_eng)
  print('Total:', len(list(isbn2ind.keys())))
  with open('isbn2lang.pickle', 'wb') as f:
    pickle.dump(isbn2lang, f, protocol=pickle.HIGHEST_PROTOCOL)

def filter_language():
  import pyisbn
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)
  with open('isbn2lang.pickle', 'rb') as f:
    isbn2lang = pickle.load(f)
  with open('isbn2ratings_count4.pickle', 'rb') as f:
    isbn2ratings_count = pickle.load(f)
  with open('isbn2rating4.pickle', 'rb') as f:
    isbn2ratings = pickle.load(f)
  with open('isbn2title4.pickle', 'rb') as f:
    isbn2title = pickle.load(f)
  with open('isbn2authors4.pickle', 'rb') as f:
    isbn2authors = pickle.load(f)


  isbns = []

  # rating_counts = list(isbn2ratings.values()) # list(isbn2ratings_count.values())
  # print('max rating', max(rating_counts))
  # plt.hist(rating_counts, bins=np.arange(0, 20, 1))
  # plt.show()

  for isbn in tqdm(isbn2ind):
    lang = isbn2lang[isbn]
    rating_count = isbn2ratings_count[isbn]
    rating = isbn2ratings[isbn]
    title = isbn2title[isbn]
    # print(lang)
    if isbn in ['ordernumber85'] or 'http:' in isbn:
      continue
    try:
      if not pyisbn.validate(isbn):
        continue
    except:
      continue
    if rating_count > 230 and (lang == 'en') and 3.8 <= rating <= 20 and len(title) > 1:
      isbns.append(isbn)

  print('num_left', len(isbns))
  titles = set()
  new_isbns = []

  for isbn in tqdm(isbns):
    title = isbn2title[isbn].lower()
    # if a replace ind is None because rating is smaller
    # than similar book, then add is false
    if title in titles:
      continue
    else:
      titles.add(title)
      new_isbns.append(isbn)

  print(titles)

  # for isbn in tqdm(isbns):
  #   title = isbn2title[isbn].lower()
  #   rating_count = isbn2ratings_count[isbn]
  #   author = isbn2authors[isbn]
  #   replace_ind = None
  #   # if a replace ind is None because rating is smaller
  #   # than similar book, then add is false
  #   add = True
  #   for i, isbn1 in enumerate(new_isbns):
  #     title1 = isbn2title[isbn1].lower()
  #     author1 = isbn2authors[isbn1]
  #     if author1 != author:
  #       continue
  #     if fuzz.ratio(title, title1) > 93:
  #       rating_count1 = isbn2ratings_count[isbn1]
  #       if rating_count > rating_count1:
  #         replace_ind = i
  #         add = False
  #         # print(title1, ',',  title)
  #         break
  #   if replace_ind is not None:
  #     new_isbns[replace_ind] = isbn
  #   elif add:
  #     new_isbns.append(isbn)

  isbns = sorted(new_isbns)
  # isbns = sorted(isbns)
  isbn2ind = {isbn:ind for ind, isbn in enumerate(isbns)}
  # print(isbn2ind.keys())
  print('num_left', len(isbns))
  with open('isbn2ind.pickle', 'wb') as f:
    pickle.dump(isbn2ind, f, protocol=pickle.HIGHEST_PROTOCOL)

# filter_language()



def filter_similar_book_pair():
  with open('isbn2ind.pickle', 'rb') as f:
    isbn2ind = pickle.load(f)
  fname = 'same_shelf_pair.pickle'
  with open(fname, 'rb') as f:
    book_similar_pair = pickle.load(f)
  del_pair = []
  for pair in tqdm(book_similar_pair):
    # print(pair)
    if len(pair) < 2:
      continue
    isbn1, isbn2 = pair
    if isbn1 not in isbn2ind or isbn2 not in isbn2ind:
      del_pair.append(pair)
  for pair in tqdm(del_pair):
    book_similar_pair.pop(pair, None)
  # print(list(book_similar_pair.items())[10:])
  with open(fname, 'wb') as f:
    pickle.dump(book_similar_pair, f, protocol=pickle.HIGHEST_PROTOCOL)


# filter_similar_book_pair()
# clean_emb()



# prepare_similar_rating_pair()
# similar_rating_pair()
# isbn2language()
# clean_emb()
# clean_rating()
# similar_book_pair()
# same_author_pair()
# same_shelf_pair()


