# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" NeuMF
Reference:
    "Neural Collaborative Filtering"
    Xiangnan He et al., WWW'2017.
Reference code:
    The authors' tensorflow implementation https://github.com/hexiangnan/neural_collaborative_filtering
CMD example:
    python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 \
    --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseModel import GeneralModel


class NeuMF(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'layers']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of each layer.")
        parser.add_argument('--use_kg', type=int, default=0,
                           help='Whether to use KG embeddings (0: No, 1: Yes).')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.layers = eval(args.layers)
        self.use_kg = args.use_kg

        # Load KG embeddings and skill levels
        if self.use_kg:
            self._load_kg_embeddings()
        self._load_user_skill_level()

        self._define_params()
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

    def _define_params(self):
        # User and item embeddings for MF and MLP
        self.mf_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mf_i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.mlp_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mlp_i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        # KG embeddings for items
        if self.use_kg:
            kg_embedding_list = [
                np.array(list(self.kg_embeddings[item_id].values()), dtype=np.float32) if item_id in self.kg_embeddings else np.zeros(self.emb_size, dtype=np.float32)
                for item_id in range(self.item_num)
            ]
            kg_embedding_tensor = torch.tensor(kg_embedding_list, dtype=torch.float32)
            self.kg_embedding_layer = nn.Embedding.from_pretrained(kg_embedding_tensor, freeze=False)

        # Skill level transformation for users
        self.skill_level_transform = nn.Linear(1, self.emb_size)

        # MLP layers
        self.mlp = nn.ModuleList([])
        pre_size = 2 * self.emb_size  # Input size for MLP (user + item embeddings)
        for layer_size in self.layers:
            self.mlp.append(nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Prediction layer
        self.prediction = nn.Linear(pre_size + self.emb_size, 1, bias=False)

    def preprocess_feed_dict(self, feed_dict):
        """Add skill levels to feed_dict."""
        feed_dict['skill_level'] = torch.tensor([
            self.user_skill_mapping.get(user_id, 0) for user_id in feed_dict['user_id']
        ], dtype=torch.long, device=feed_dict['user_id'].device)
        return feed_dict

    def forward(self, feed_dict):
        self.check_list = []
        feed_dict = self.preprocess_feed_dict(feed_dict)

        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        skill_levels = feed_dict['skill_level'].float().unsqueeze(-1)  # [batch_size, 1]

        # Repeat user IDs to match item IDs
        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        # MF embeddings
        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)

        # MLP embeddings
        mlp_u_vectors = self.mlp_u_embeddings(u_ids)
        mlp_i_vectors = self.mlp_i_embeddings(i_ids)

        # Add skill level embeddings to user embeddings (MLP branch)
        skill_level_emb = self.skill_level_transform(skill_levels)  # [batch_size, emb_size]
        mlp_u_vectors += skill_level_emb.unsqueeze(1)  # Broadcast to match shape

        # Add KG embeddings to item embeddings (both MF and MLP branches)
        if self.use_kg:
            kg_i_vectors = self.kg_embedding_layer(i_ids)  # [batch_size, -1, emb_size]
            mf_i_vectors += kg_i_vectors
            mlp_i_vectors += kg_i_vectors

        # MF interaction
        mf_vector = mf_u_vectors * mf_i_vectors  # Element-wise product

        # MLP interaction
        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)  # Concatenate user and item embeddings
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        # Concatenate MF and MLP outputs
        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)  # Final prediction

        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
