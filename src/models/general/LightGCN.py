# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import pandas as pd

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

class LightGCNBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		parser.add_argument('--use_kg', type=int, default=0,
                            help='Whether to use KG embeddings (0: No, 1: Yes).')
		return parser
	
	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()

		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()

		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		adj_mat = adj_mat.todok()

		def normalized_adj_single(adj):
			# D^-1/2 * A * D^-1/2
			rowsum = np.array(adj.sum(1)) + 1e-10

			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()

		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)

		return norm_adj_mat.tocsr()

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.use_kg = args.use_kg
		# Load precomputed KG embeddings and skill levels
		self._load_kg_embeddings()
		self._load_user_skill_level()
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self._base_define_params()
		self.apply(self.init_weights)

	def _load_kg_embeddings(self):
		"""Load KG embeddings from precomputed file."""
		kg_path = 'C:/Users/User/Desktop/kg_embeddings.txt'
		self.kg_embeddings = pd.read_csv(kg_path, sep='\t', header=None)
		self.kg_embeddings.columns = ['entity_id'] + [f'emb_{i}' for i in range(self.emb_size)]
		self.kg_embeddings = self.kg_embeddings.set_index('entity_id').to_dict(orient='index')

	def _load_user_skill_level(self):
		"""Load user skill level mapping."""
		skill_level_path = 'C:/Users/User/ReChorus/data/k_fold_his_fav_train_int_dev_test/user_skill_level.csv'
		skill_df = pd.read_csv(skill_level_path, sep='\t')
		self.user_skill_mapping = skill_df.set_index('user_id')['skill_level'].to_dict()
	
	def _base_define_params(self):	
		self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers, self.use_kg, self.kg_embeddings)

	def preprocess_feed_dict(self, feed_dict):
		"""Modify feed_dict to include KG and skill level embeddings."""
		feed_dict['skill_level'] = torch.tensor([
			self.user_skill_mapping.get(user_id, 0) for user_id in feed_dict['user_id']
		], dtype=torch.long, device=feed_dict['user_id'].device)

		return feed_dict
	
	def forward(self, feed_dict):
		self.check_list = []
		feed_dict = self.preprocess_feed_dict(feed_dict) 
		user, items = feed_dict['user_id'], feed_dict['item_id']
		u_embed, i_embed = self.encoder(user, items, feed_dict['skill_level'])

		prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
		u_v = u_embed.repeat(1,items.shape[1]).view(items.shape[0],items.shape[1],-1)
		i_v = i_embed
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

class LightGCN(GeneralModel, LightGCNBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = LightGCNBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}
	
class LightGCNImpression(ImpressionModel, LightGCNBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return LightGCNBase.forward(self, feed_dict)

class LGCNEncoder(nn.Module):
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3, use_kg=False, kg_embeddings=None):
		super(LGCNEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj
		self.use_kg = use_kg

		self.embedding_dict = self._init_model()
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

		self.kg_embeddings = kg_embeddings

		
		if self.use_kg:
			kg_embedding_list = [
				np.array(list(self.kg_embeddings[item_id].values()), dtype=np.float32) if item_id in self.kg_embeddings else np.zeros(self.emb_size, dtype=np.float32)
				for item_id in range(item_count)
			]

			kg_embedding_tensor = torch.tensor(kg_embedding_list, dtype=torch.float32)

			self.kg_embedding_layer = nn.Embedding.from_pretrained(kg_embedding_tensor, freeze=False)

        # **Only for users**
		self.skill_level_transform = nn.Linear(1, self.emb_size)  # Convert skill levels to embeddings

	def _init_model(self):
		initializer = nn.init.xavier_uniform_
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
		})
		return embedding_dict

	@staticmethod
	def _convert_sp_mat_to_sp_tensor(X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)

	def forward(self, users, items, skill_levels):
		"""Forward pass with KG embeddings for items and skill levels for users."""
		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]

		for k in range(len(self.layers)):
			ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]

		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1)

		user_all_embeddings = all_embeddings[:self.user_count, :]
		item_all_embeddings = all_embeddings[self.user_count:, :]

		user_embeddings = user_all_embeddings[users, :]
		item_embeddings = item_all_embeddings[items, :]

		# ✅ Apply **skill level embeddings only for users**
		skill_levels = skill_levels.float().unsqueeze(-1)  # Convert to float tensor
		skill_level_emb = self.skill_level_transform(skill_levels)
		user_embeddings += skill_level_emb

		# ✅ Apply **KG embeddings only for items**
		if self.use_kg:
			# Retrieve KG embeddings only for items in the current batch
			batch_item_ids = items.cpu().numpy().flatten()  # Extract item IDs in batch
			item_kg_list = [
				np.array(list(self.kg_embeddings[item_id].values()), dtype=np.float32).reshape(self.emb_size) 
				if item_id in self.kg_embeddings else np.zeros(self.emb_size, dtype=np.float32)
				for item_id in batch_item_ids
			]

			# Convert to tensor with correct shape [batch_size, num_items_per_batch, emb_size]
			item_kg_embeddings = torch.tensor(item_kg_list, dtype=torch.float32, device=items.device)
			item_kg_embeddings = item_kg_embeddings.view(items.shape[0], items.shape[1], -1)  # Reshape to match item_embeddings

			# Ensure the shapes match before addition
			if item_embeddings.shape != item_kg_embeddings.shape:
				raise ValueError(f"Shape mismatch: item_embeddings {item_embeddings.shape}, item_kg_embeddings {item_kg_embeddings.shape}")

			# Add KG embeddings to item embeddings
			item_embeddings += item_kg_embeddings

		return user_embeddings, item_embeddings
