from torch import nn
from torch.autograd import Variable
import torch
from torch.nn import Linear, ReLU, Tanh, Sigmoid, Module, Parameter, BCELoss, MSELoss, LeakyReLU
from torch.optim import Adam, SGD
from torch.nn.init import xavier_uniform_
import numpy as np
import hyperparam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pickle
import pandas as pd

device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

with open('isbn2ind.pickle', 'rb') as f:
  isbn2ind = pickle.load(f)


def create_emb_layer(fn='book_embedding_fasttext.pt', non_trainable=False):
  emb_layer = torch.load(fn)
  num_embeddings, embedding_dim = emb_layer.weight.size()
  # emb_layer = nn.Embedding(num_embeddings, embedding_dim)
  # emb_layer.load_state_dict({'weight': weights_matrix})
  if non_trainable:
    emb_layer.weight.requires_grad = False
  return emb_layer, num_embeddings, embedding_dim


class FastTextSimCLR(nn.Module):
  def __init__(self, out_dim):
    super(FastTextSimCLR, self).__init__()
    # Feature
    self.embedding, num_embeddings, embedding_dim = create_emb_layer(fn=hyperparam.embedding_fn)
    if device != 'cpu':
      self.embedding.cuda()
    # Projector
    self.projector1 = nn.Linear(embedding_dim, embedding_dim)
    xavier_uniform_(self.projector1.weight)
    self.projector2 = nn.Linear(embedding_dim, out_dim)
    xavier_uniform_(self.projector2.weight)

  def encode(self, x):
    print('input:', x)
    emb = self.embedding(x)
    print('encoder:', emb)
    return emb

  def projector(self, x):
    return self.projector2(self.projector1(x))

  def forward(self, x):
    h = self.encode(x)
    z = self.projector(h)
    print('projection:', z)
    return z


class NTXentLoss(torch.nn.Module):
  def __init__(self, device, batch_size, temperature, use_cosine_similarity):
    super(NTXentLoss, self).__init__()
    self.batch_size = batch_size
    self.temperature = temperature
    self.device = device
    self.softmax = torch.nn.Softmax(dim=-1)
    self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
    self.similarity_function = self._get_similarity_function(use_cosine_similarity)
    self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

  def _get_similarity_function(self, use_cosine_similarity):
    if use_cosine_similarity:
      self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
      return self._cosine_simililarity
    else:
      return self._dot_simililarity


  def _get_correlated_mask(self):
    diag = np.eye(2 * self.batch_size)
    l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
    l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
    mask = torch.from_numpy((diag + l1 + l2))
    mask = (1 - mask).type(torch.bool)
    return mask.to(self.device)

  @staticmethod
  def _dot_simililarity(x, y):
    v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v

  def _cosine_simililarity(self, x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
    return v

  def forward(self, zis, zjs):
    representations = torch.cat([zjs, zis], dim=0)
    print('Representations:', representations)
    print('Representations shape:', representations.shape)

    similarity_matrix = self.similarity_function(representations, representations)
    print('similarity_matrix shape:', similarity_matrix.shape)
    # filter out the scores from the positive samples
    l_pos = torch.diag(similarity_matrix, self.batch_size)
    r_pos = torch.diag(similarity_matrix, -self.batch_size)
    positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

    negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

    logits = torch.cat((positives, negatives), dim=1)
    logits /= self.temperature
    print('logit shape:', logits.shape)


    labels = torch.zeros(2 * self.batch_size).to(self.device).long()
    loss = self.criterion(logits, labels)
    print('NTX loss:', loss)

    return loss / (2 * self.batch_size)


class SimCLR:
  def __init__(self, device):
    print('Read Dataset:')
    self.train_loader, self.test_loader = get_train_test_data(hyperparam.metric_weight)
    self.device = device
    print('Got device:', self.device)
    self.nt_xent_criterion = NTXentLoss(self.device, hyperparam.batch_size, hyperparam.temperature, hyperparam.use_cosine_similarity)

  def _step(self, encode_proj_model, x1, x2, weight):
    z1 = encode_proj_model(x1)
    z2 = encode_proj_model(x2)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    loss = self.nt_xent_criterion(z1, z2)
    print('weight', weight)
    print('weight shape:', weight.shape)
    return loss*weight

  def train(self):
    encode_proj_model = FastTextSimCLR(hyperparam.out_dim)

    optimizer = torch.optim.Adam(encode_proj_model.parameters(), 3e-4, weight_decay=hyperparam.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.train_loader), eta_min=0,
                                                           last_epoch=-1)
    n_iter = 0
    best_test_loss = np.inf
    for epoch in range(hyperparam.epochs):
      print('=========EPOCH {}==========='.format(epoch))
      for x1, x2, weight in self.train_loader:
        optimizer.zero_grad()
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        # print(x1)
        # x1 = x1.long()
        # x2 = x2.long()
        loss = self._step(encode_proj_model, x1, x2, weight)
        if n_iter % hyperparam.log_every_n_steps == 0:
          print('Train loss at step {}: {}'.format(n_iter, loss))
        loss.backward()
        n_iter += 1
      if epoch % hyperparam.eval_every_n_epochs == 0:
        test_loss = self._test(encode_proj_model)
        print('Test loss at step {}: {}'.format(n_iter, loss))
        if test_loss < best_test_loss:
          best_test_loss = test_loss
          torch.save(encode_proj_model, 'best_model.pt')
      if epoch >= 10:
        scheduler.step()
      print('cosine_lr_decay:', scheduler.get_lr()[0])

  def _test(self, model):
    with torch.no_grad():
      model.eval()
      test_loss = 0
      counter = 0
      for x1, x2, weight in self.test_loader:
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        loss = self._step(model, x1, x2, weight)
        test_loss += loss.item()
        counter += 1
      test_loss /= counter
    model.train()
    return test_loss


def prepare_data(weight):
  import random
  import os
  from tqdm import tqdm
  root = './sim_score/'
  print('Pandas Load')
  # df = pd.read_csv('sim_pair_scores.csv', header=None)
  pair2w = {}
  for i, fn in enumerate(os.listdir(root)):
    fn = root + fn
    print(fn)
    with open(fn, 'rb') as f:
      data = pickle.load(f)
    # print('mean:', (data.mean() * 97000**2)/(data.count_nonzero()))
    xs, ys = data.nonzero()
    xys = [(x,y) for x,y in list(zip(xs, ys)) if y > x]
    random.shuffle(xys)
    for x, y in xys[:10000]:
      if y > x:
        pair = (x, y)
        w = hyperparam.metric_weight[i]
        pair2w[pair] = pair2w.get(pair, 0) + w
    print(len(pair2w))
    break
  data = [(pair[0], pair[1], pair2w[pair]) for pair in pair2w]
  print('Train test split')
  random.shuffle(data)
  train = data[:int(len(data)*0.7)]
  test = data[int(len(data)*0.7):]
  return train, test


class BookSimData(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    # prepare input
    x1, x2, w = self.data[idx]
    print(x1, x2)
    return torch.tensor(x1).long(), torch.tensor(x2).long(), w

def get_train_test_data(weight):
  train, test = prepare_data(weight)
  train_d, test_d = BookSimData(train), BookSimData(test)
  train_dl = DataLoader(train_d, batch_size=hyperparam.batch_size, shuffle=True, num_workers=0)
  test_dl = DataLoader(test_d, batch_size=len(test_d), shuffle=True, num_workers=0)
  return train_dl, test_dl

simclr = SimCLR(device)
print('...Training...')
simclr.train()
# for commit


