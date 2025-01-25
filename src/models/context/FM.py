# -*- coding: UTF-8 -*-
# @Author : Jiayu Li 
# @Email  : jy-li20@mails.tsinghua.edu.cn

""" FM
Reference:
	'Factorization Machines', Steffen Rendle, 2010 IEEE International conference on data mining.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel

class FMBase(object):
    @staticmethod
    def parse_model_args_FM(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--use_kg', type=int, default=0,
                            help='Whether to include KG embeddings (0: No, 1: Yes).')
        parser.add_argument('--normalize_embeddings', type=int, default=1,
                            help='Whether to normalize embeddings (0: No, 1: Yes).')
        return parser

    def _define_init_params(self, args, corpus):
        self.vec_size = args.emb_size
        self.use_kg = args.use_kg
        self.normalize_embeddings = args.normalize_embeddings

        # Load user skill level mapping
        self._load_user_skill_level()

        if self.use_kg:
            self._load_and_initialize_kg_embeddings()

        self._define_params_FM()
        self.apply(self.init_weights)

    def _define_params_FM(self):
        self.context_embedding = nn.ModuleDict()
        self.linear_embedding = nn.ModuleDict()
        for f in self.context_features:
            self.context_embedding[f] = (
                nn.Embedding(self.feature_max[f], self.vec_size) 
                if f.endswith('_c') or f.endswith('_id') 
                else nn.Linear(1, self.vec_size, bias=False)
            )
            self.linear_embedding[f] = (
                nn.Embedding(self.feature_max[f], 1) 
                if f.endswith('_c') or f.endswith('_id') 
                else nn.Linear(1, 1, bias=False)
            )

        # Skill level as a continuous feature
        self.skill_level_transform = nn.Linear(1, self.vec_size)
        self.skill_level_linear = nn.Linear(1, 1)
        self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)

    def _load_user_skill_level(self):
        """Load user skill level mapping."""
        skill_level_path = 'C:/Users/User/ReChorus/data/k_fold_his_fav_train_int_dev_test/user_skill_level.csv'
        skill_df = pd.read_csv(skill_level_path, sep='\t')
        self.user_skill_mapping = skill_df.set_index('user_id').to_dict(orient='index')

    def _load_and_initialize_kg_embeddings(self):
        """Load and initialize KG embeddings as trainable parameters."""
        kg_path = 'C:/Users/User/Desktop/kg_embeddings.txt'
        kg_df = pd.read_csv(kg_path, sep='\t', header=None)
        kg_df.columns = ['entity_id'] + [f'emb_{i}' for i in range(self.vec_size)]
        kg_dict = kg_df.set_index('entity_id').to_dict(orient='index')

        # Convert KG embeddings to a trainable embedding layer
        self.kg_embeddings = nn.Embedding(len(kg_dict), self.vec_size, padding_idx=0)
        kg_tensor = torch.zeros(len(kg_dict), self.vec_size)
        for i, emb in enumerate(kg_dict.values()):
            kg_tensor[i] = torch.tensor(list(emb.values()), dtype=torch.float32)
        self.kg_embeddings.weight.data.copy_(kg_tensor)
        self.kg_embeddings.weight.requires_grad = True

    def _normalize(self, tensor):
        """Normalize tensor along its last dimension."""
        return tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-8)

    def _get_embeddings_FM(self, feed_dict):
        item_ids = feed_dict['item_id']
        batch_size, item_num = item_ids.shape

        fm_vectors = []

        # Process standard context features
        for f in self.context_features:
            if f.endswith('_c') or f.endswith('_id'):
                feature_input = feed_dict[f].long() if isinstance(feed_dict[f], torch.Tensor) else torch.tensor(feed_dict[f], dtype=torch.long)
                feature_emb = self.context_embedding[f](feature_input)
            else:
                feature_emb = self.context_embedding[f](feed_dict[f].float().unsqueeze(-1))

            if self.normalize_embeddings:
                feature_emb = self._normalize(feature_emb)

            # Ensure consistent dimensions: [batch_size, item_num, feature_dim]
            if len(feature_emb.shape) == 2:
                feature_emb = feature_emb.unsqueeze(1).repeat(1, item_num, 1)
            fm_vectors.append(feature_emb)

        # Include KG embeddings if use_kg is True
        if self.use_kg:
            kg_emb = feed_dict['kg_embeddings']  # Already retrieved in preprocess_feed_dict
            if self.normalize_embeddings:
                kg_emb = self._normalize(kg_emb)
            kg_emb = kg_emb.view(batch_size, item_num, -1)
            fm_vectors.append(kg_emb)

        # Include skill_level as a continuous feature
        if 'skill_level' in feed_dict:
            skill_level_input = feed_dict['skill_level'].float().unsqueeze(-1)
            skill_level_emb = self.skill_level_transform(skill_level_input)
            if self.normalize_embeddings:
                skill_level_emb = self._normalize(skill_level_emb)
            skill_level_emb = skill_level_emb.unsqueeze(1).repeat(1, item_num, 1)
            fm_vectors.append(skill_level_emb)

        # Stack embeddings
        fm_vectors = torch.stack(fm_vectors, dim=-2)

        # Linear embedding values
        linear_value = []
        for f in self.context_features:
            if f.endswith('_c') or f.endswith('_id'):
                feature_input = feed_dict[f].long() if isinstance(feed_dict[f], torch.Tensor) else torch.tensor(feed_dict[f], dtype=torch.long)
                feature_value = self.linear_embedding[f](feature_input)
            else:
                feature_value = self.linear_embedding[f](feed_dict[f].float().unsqueeze(-1))

            if len(feature_value.shape) == 2:
                feature_value = feature_value.unsqueeze(1).repeat(1, item_num, 1)
            linear_value.append(feature_value)

        if self.use_kg:
            kg_linear_value = kg_emb.sum(dim=-1, keepdim=True)
            linear_value.append(kg_linear_value)

        if 'skill_level' in feed_dict:
            skill_level_linear_value = self.skill_level_linear(skill_level_input)
            skill_level_linear_value = skill_level_linear_value.unsqueeze(1).repeat(1, item_num, 1)
            linear_value.append(skill_level_linear_value)

        linear_value = torch.cat(linear_value, dim=-1)
        linear_value = self.overall_bias + linear_value.sum(dim=-1)

        return fm_vectors, linear_value

    def forward(self, feed_dict):
        fm_vectors, linear_value = self._get_embeddings_FM(feed_dict)
        fm_vectors = 0.5 * (fm_vectors.sum(dim=-2).pow(2) - fm_vectors.pow(2).sum(dim=-2))
        predictions = linear_value + fm_vectors.sum(dim=-1)
        return {'prediction': predictions}

class FMCTR(ContextCTRModel, FMBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = FMBase.parse_model_args_FM(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init_params(args,corpus)

	def forward(self, feed_dict):
		out_dict = FMBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class FMTopK(ContextModel, FMBase):
    reader, runner = 'ContextReader', 'BaseRunner'
    extra_log_args = ['emb_size', 'loss_n']

    @staticmethod
    def parse_model_args(parser):
        parser = FMBase.parse_model_args_FM(parser)
        return ContextModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ContextModel.__init__(self, args, corpus)
        self._define_init_params(args, corpus)

    def forward(self, feed_dict):
        # Preprocess feed_dict to add skill_level and KG features
        feed_dict = self.preprocess_feed_dict(feed_dict)
        return FMBase.forward(self, feed_dict)

    def preprocess_feed_dict(self, feed_dict):
        """Dynamically add skill_level and KG embeddings to feed_dict."""
        # Add skill level
        feed_dict['skill_level'] = torch.tensor(
            [self.user_skill_mapping[user_id]['skill_level'] if user_id in self.user_skill_mapping else 0
            for user_id in feed_dict['user_id']],
            dtype=torch.long,
            device=self.device
        )

        # Handle KG embeddings
        if self.use_kg:
            # Ensure item IDs are valid indices for the embedding layer
            valid_item_ids = feed_dict['item_id'].long()
            kg_embeddings = self.kg_embeddings(valid_item_ids)
            feed_dict['kg_embeddings'] = kg_embeddings

        return feed_dict
