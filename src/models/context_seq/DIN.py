# -*- coding: UTF-8 -*-
# @Author : Jiayu Li 
# @Email  : jy-li20@mails.tsinghua.edu.cn

""" 
Reference:
	'Deep interest network for click-through rate prediction', Zhou et al., SIGKDD2018.
Implementation reference:  
	RecBole: https://github.com/RUCAIBox/RecBole
	DIN pytorch repo: https://github.com/fanoping/DIN-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextSeqModel, ContextSeqCTRModel
from utils.layers import MLP_Block

class DINBase(object):
	@staticmethod
	def parse_model_args_din(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--att_layers', type=str, default='[64]',
							help="Size of each layer in the attention module.")
		parser.add_argument('--dnn_layers', type=str, default='[64]',
							help="Size of each layer in the MLP module.")
		parser.add_argument('--use_kg', type=int, default=0,
                        	help='Whether to include knowledge graph embeddings (0: No, 1: Yes).')
		return parser

	def _define_init(self, args,corpus):
		self.user_context = ['user_id']+corpus.user_feature_names
		self.item_context = ['item_id']+corpus.item_feature_names
		self.situation_context = corpus.situation_feature_names
		self.item_feature_num = len(self.item_context)
		self.user_feature_num = len(self.user_context)
		self.situation_feature_num = len(corpus.situation_feature_names) if self.add_historical_situations else 0
  
		self.vec_size = args.emb_size
		self.att_layers = eval(args.att_layers)
		self.dnn_layers = eval(args.dnn_layers)
		self.use_kg = args.use_kg

		self._load_user_skill_level()

		if self.use_kg:
			self._load_kg_embeddings()

		self._define_params_DIN()
		self.apply(self.init_weights)

	def _load_user_skill_level(self):
		"""Load user skill level mapping."""
		skill_level_path = 'C:/Users/User/ReChorus/data/k_fold_his_fav_train_int_dev_test/user_skill_level.csv'
		skill_df = pd.read_csv(skill_level_path, sep='\t')
		self.user_skill_mapping = skill_df.set_index('user_id').to_dict(orient='index')

	def _load_kg_embeddings(self):
		"""Load KG embeddings from precomputed file."""
		kg_path = 'C:/Users/User/Desktop/kg_embeddings.txt'
		self.kg_embeddings = pd.read_csv(kg_path, sep='\t', header=None)
		self.kg_embeddings.columns = ['entity_id'] + [f'emb_{i}' for i in range(self.vec_size)]
		self.kg_embeddings = self.kg_embeddings.set_index('entity_id').to_dict(orient='index')

	def preprocess_feed_dict(self, feed_dict):
		"""Add skill_level and best_sport to feed_dict."""
		feed_dict['skill_level'] = [
			self.user_skill_mapping[user_id]['skill_level'] if user_id in self.user_skill_mapping else 0
			for user_id in feed_dict['user_id']
		]
		feed_dict['best_sport'] = [
			self.user_skill_mapping[user_id]['best_sport_id'] if user_id in self.user_skill_mapping else 0
			for user_id in feed_dict['user_id']
		]
		return feed_dict

	def _define_params_DIN(self):
		self.embedding_dict = nn.ModuleDict()
		for f in self.user_context + self.item_context + self.situation_context:
			self.embedding_dict[f] = (
				nn.Embedding(self.feature_max[f], self.vec_size)
				if f.endswith('_c') or f.endswith('_id') or f in ['skill_level', 'best_sport']
				else nn.Linear(1, self.vec_size, bias=False)
			)

		# Attention MLP Layer
		pre_size = 4 * (self.item_feature_num + self.situation_feature_num) * self.vec_size
		self.att_mlp_layers = MLP_Block(
			input_dim=pre_size,
			hidden_units=self.att_layers,
			output_dim=1,
			hidden_activations='Sigmoid',
			dropout_rates=self.dropout,
			batch_norm=False
		)

		if self.use_kg:
			# DNN MLP Layer (Corrected pre_size)
			pre_size = (
				1856 +  # user_his_emb2d
				1856 +  # user_his_emb2d * current_emb2d
				7424    # all_context2d
			)
		else:
			pre_size = (2*(self.item_feature_num+self.situation_feature_num)+self.item_feature_num
              +len(self.situation_context) + self.user_feature_num) * self.vec_size

		self.dnn_mlp_layers = MLP_Block(
			input_dim=pre_size,
			hidden_units=self.dnn_layers,
			output_dim=1,
			hidden_activations='Dice',
			dropout_rates=self.dropout,
			batch_norm=True,
			norm_before_activation=True
		)

	def attention(self, queries, keys, keys_length, mask_mat,softmax_stag=False, return_seq_weight=False):
		'''Reference:
			RecBole layers: SequenceAttLayer, https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py#L294
			queries: batch * (if*vecsize)
		'''
		embedding_size = queries.shape[-1]  # H
		hist_len = keys.shape[1]  # T
		queries = queries.repeat(1, hist_len)
		queries = queries.view(-1, hist_len, embedding_size)
		# MLP Layer
		input_tensor = torch.cat(
			[queries, keys, queries - keys, queries * keys], dim=-1
		)
		output = torch.transpose(self.att_mlp_layers(input_tensor), -1, -2)
		# get mask
		output = output.squeeze(1)
		mask = mask_mat.repeat(output.size(0), 1)
		mask = mask >= keys_length.unsqueeze(1)
		# mask
		if softmax_stag:
			mask_value = -np.inf
		else:
			mask_value = 0.0
		output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
		output = output.unsqueeze(1)
		output = output / (embedding_size**0.5)
		# get the weight of each user's history list about the target item
		if softmax_stag:
			output = fn.softmax(output, dim=2)  # [B, 1, T]
		if not return_seq_weight:
			output = torch.matmul(output, keys)  # [B, 1, H]
		torch.cuda.empty_cache()
		return output.squeeze(dim=1)

	def get_all_embedding(self, feed_dict, merge_all=True):
		# item embedding
		item_feats_emb = torch.stack([self.embedding_dict[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict[f](feed_dict[f].float().unsqueeze(-1))
					for f in self.item_context],dim=-2) # batch * feature num * emb size
		# historical item embedding
		history_item_emb = torch.stack([self.embedding_dict[f](feed_dict['history_'+f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict['history_'+f](feed_dict[f].float().unsqueeze(-1))
					for f in self.item_context],dim=-2) # batch * feature num * emb size
		# Add KG Embeddings
		if self.use_kg:
			kg_emb = torch.stack([
				torch.tensor(self.kg_embeddings[item_id], dtype=torch.float32)
				if item_id in self.kg_embeddings else torch.zeros(self.vec_size)
				for item_id in feed_dict['item_id']
			]).to(self.device)  # (batch, emb_size)

			# Expand and align KG embeddings with item_feats_emb
			kg_emb = kg_emb.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, emb_size)
			kg_emb = kg_emb.repeat(1, item_feats_emb.shape[1], item_feats_emb.shape[2], 1)

			# Concatenate KG embeddings along the last dimension
			item_feats_emb = torch.cat([item_feats_emb, kg_emb], dim=-1)  # (batch, feature_num, emb_size + kg_emb_size)

		# user embedding
		user_feats_emb = torch.stack([self.embedding_dict[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id') or f in ['skill_level', 'best_sport']
					else self.embedding_dict[f](feed_dict[f].float().unsqueeze(-1))
					for f in self.user_context],dim=-2) # batch * feature num * emb size
		# situation embedding
		if len(self.situation_context):
			situ_feats_emb = torch.stack([self.embedding_dict[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict[f](feed_dict[f].float().unsqueeze(-1))
					for f in self.situation_context],dim=-2) # batch * feature num * emb size
		else:
			situ_feats_emb = []
		# historical situation embedding
		if self.add_historical_situations:
			history_situ_emb = torch.stack([self.embedding_dict[f](feed_dict['history_'+f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict[f](feed_dict['history_'+f].float().unsqueeze(-1))
					for f in self.situation_context],dim=-2) # batch * feature num * emb size
			history_emb = torch.cat([history_item_emb,history_situ_emb],dim=-2).flatten(start_dim=-2)
			item_num = item_feats_emb.shape[1]
			current_emb = torch.cat([item_feats_emb, situ_feats_emb.unsqueeze(1).repeat(1,item_num,1,1)],dim=-2).flatten(start_dim=-2)
		else:
			history_emb = history_item_emb.flatten(start_dim=-2)
			current_emb = item_feats_emb.flatten(start_dim=-2)

		if merge_all:
			item_num = item_feats_emb.shape[1]  # Number of items
			if self.use_kg:
				feature_dim = item_feats_emb.shape[2]  # Number of features per item
				emb_size = item_feats_emb.shape[-1]  # Embedding size (includes KG if use_kg)

				# ✅ Properly expand user_feats_emb to match item_num
				user_feats_emb = user_feats_emb.unsqueeze(1).repeat(1, item_num, 1, 1)  # (batch, item_num, user_feature_num, emb_size)

				# ✅ Flatten user_feats_emb across user_feature_num
				user_feats_emb = user_feats_emb.flatten(start_dim=2)  # (batch, item_num, user_feature_num * emb_size)

				# ✅ Add a feature dimension to match item_feats_emb
				user_feats_emb = user_feats_emb.unsqueeze(2).repeat(1, 1, feature_dim, 1)  # (batch, item_num, feature_dim, emb_size)

				# ✅ Ensure embedding sizes align
				if user_feats_emb.shape[-1] > emb_size:
					user_feats_emb = user_feats_emb[..., :emb_size]  # Trim excess embeddings
				elif user_feats_emb.shape[-1] < emb_size:
					padding = emb_size - user_feats_emb.shape[-1]
					user_feats_emb = torch.cat([
						user_feats_emb,
						torch.zeros_like(user_feats_emb[..., :padding])
					], dim=-1)

				# ✅ Concatenate embeddings
				all_context = torch.cat([
					item_feats_emb,
					user_feats_emb
				], dim=-1)  # (batch, item_num, feature_dim, combined_emb_size)

				# ✅ Flatten the feature dimension
				all_context = all_context.flatten(start_dim=2)  # (batch, item_num, combined_feature_dim)

			else:
				if len(situ_feats_emb):
					all_context = torch.cat([item_feats_emb, user_feats_emb.unsqueeze(1).repeat(1,item_num,1,1),
								situ_feats_emb.unsqueeze(1).repeat(1,item_num,1,1)],dim=-2).flatten(start_dim=-2)
				else:
					all_context = torch.cat([item_feats_emb, user_feats_emb.unsqueeze(1).repeat(1,item_num,1,1),
								],dim=-2).flatten(start_dim=-2)
			
			return history_emb, current_emb, all_context
		else:
			return history_emb, current_emb, user_feats_emb, situ_feats_emb

	def forward(self, feed_dict):
		# Dynamically add skill_level and best_sport to feed_dict
		feed_dict = self.preprocess_feed_dict(feed_dict)

		hislens = feed_dict['lengths']
		history_emb, current_emb, all_context = self.get_all_embedding(feed_dict)
		if self.use_kg:
			# Ensure history_emb and current_emb match expected dimensions
			batch_size, item_num, current_emb_dim = current_emb.shape
			_, max_len, history_emb_dim = history_emb.shape

			# Flatten current_emb for att_dnn compatibility
			current_emb2d = current_emb.view(batch_size * item_num, current_emb_dim)
			history_emb2d = history_emb.unsqueeze(1).repeat(1, item_num, 1, 1).view(batch_size * item_num, max_len, history_emb_dim)
			hislens2d = hislens.unsqueeze(1).repeat(1, item_num).view(-1)

			# Ensure all_context matches flattened dimensions
			all_context2d = all_context.view(batch_size * item_num, -1)

			predictions = self.att_dnn_kg(current_emb2d, history_emb2d, all_context2d, hislens2d)
			return {'prediction': predictions.view(batch_size, item_num)}
		else:
			predictions = self.att_dnn(current_emb,history_emb, all_context, hislens)
			return {'prediction':predictions}

	def att_dnn_kg(self, current_emb2d, history_emb2d, all_context2d, history_lengths):
		mask_mat = (torch.arange(history_emb2d.shape[1]).view(1, -1)).to(self.device)

		# Ensure dimensions match for attention
		batch_item_num, feats_emb = current_emb2d.shape
		_, max_len, his_emb = history_emb2d.shape

		if feats_emb != his_emb:
			# Project current_emb2d to match history_emb2d's last dimension
			projection_layer = nn.Linear(feats_emb, his_emb).to(self.device)
			current_emb2d = projection_layer(current_emb2d)

		# Perform attention
		user_his_emb2d = self.attention(
			current_emb2d,
			history_emb2d,
			history_lengths,
			mask_mat,
			softmax_stag=False
		)

		# Concatenate embeddings for final prediction
		din_output = torch.cat([
			user_his_emb2d,                       # Attention output
			user_his_emb2d * current_emb2d,       # Element-wise interaction
			all_context2d                         # Flattened combined embeddings
		], dim=-1)

		# Pass through MLP
		din_output = self.dnn_mlp_layers(din_output)

		# Reshape back to (batch_size, item_num)
		batch_size = history_lengths.shape[0]
		item_num = batch_item_num // batch_size
		return din_output.squeeze(dim=-1).view(batch_size, item_num)

	def att_dnn(self, current_emb, history_emb, all_context, history_lengths):
		mask_mat = (torch.arange(history_emb.shape[1]).view(1,-1)).to(self.device)
  
		batch_size, item_num, feats_emb = current_emb.shape
		_, max_len, his_emb = history_emb.shape
		current_emb2d = current_emb.view(-1, feats_emb) # transfer 3d (batch * candidate * emb) to 2d ((batch*candidate)*emb) 
		history_emb2d = history_emb.unsqueeze(1).repeat(1,item_num,1,1).view(-1,max_len,his_emb)
		hislens2d = history_lengths.unsqueeze(1).repeat(1,item_num).view(-1)
		user_his_emb2d = self.attention(current_emb2d, history_emb2d, hislens2d,mask_mat,softmax_stag=False)
  
		din_output = torch.cat([user_his_emb2d, user_his_emb2d*current_emb2d, all_context.view(batch_size*item_num,-1) ],dim=-1)
		din_output = self.dnn_mlp_layers(din_output)
		return din_output.squeeze(dim=-1).view(batch_size, item_num)

class DINCTR(ContextSeqCTRModel, DINBase):
	reader = 'ContextSeqReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size','att_layers','add_historical_situations']
	
	@staticmethod
	def parse_model_args(parser):
		parser = DINBase.parse_model_args_din(parser)
		return ContextSeqCTRModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = DINBase.forward(self,feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class DINTopK(ContextSeqModel, DINBase):
	reader = 'ContextSeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size','att_layers','add_historical_situations']
	
	@staticmethod
	def parse_model_args(parser):
		parser = DINBase.parse_model_args_din(parser)
		return ContextSeqModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return DINBase.forward(self, feed_dict)
