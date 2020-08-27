import numpy as np
embedding_fn = 'book_embedding_fasttext.pt'
batch_size = 512
epochs = 80
eval_every_n_epochs = 1
fine_tune_from = None
log_every_n_steps = 50
weight_decay = 10e-6
out_dim = 256
s = 1
input_shape = (96,96,3)
num_workers = 0
valid_size = 0.05
temperature = 0.5
use_cosine_similarity = True
metric_weight_name = ['same_author', 'same_rating_children', 'same_rating_comics', 'same_rating_fantasy', 'same_rating_history', 'same_rating_mystery', 'same_rating_poetry', 'same_rating_romance', 'same_rating_young_adult', 'same_shelf', 'similar_pair']
metric_weight = [3,1,1,7,1,8,1,2,2,2,4]
metric_weight = (np.array(metric_weight) / np.sum(metric_weight)).tolist()
# for commit
