# -*- coding: UTF-8 -*-

import torch
import logging
import numpy as np
import torch.nn as nn
from utils import utils
from torch.nn.utils.rnn import pad_sequence
from typing import List
from models.BaseModel import BaseModel
from utils.layers import MLP_Block

class RankSkillModel(BaseModel):
    reader, runner = 'RankSkillReader', 'RankSkillRunner'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_neg', type=int, default=1, help='Number of negative samples during training.')
        parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
        parser.add_argument('--loss_n',type=str,default='BPR',
							help='Type of loss functions.')
        parser.add_argument('--layers', type=str, default='[64]',
							help="Size of each layer.")
        parser.add_argument('--dropout', type=float, default=0, help='Dropout probability for each deep layer.')
        parser.add_argument('--test_all', type=int, default=0, help='Whether testing on all items.')
        parser.add_argument('--skill_level_pred', type=int, default=1, help='Enable skill level prediction (1) or disable (0).')
        parser.add_argument('--lambda1', type=float, default=1.0, help='Weight for ranking loss.')
        parser.add_argument('--lambda2', type=float, default=1.0, help='Weight for skill level loss.')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.args = args
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.skill_level_num = 4  # Number of skill levels
        self.num_neg = args.num_neg
        self.dropout = args.dropout
        self.loss_n = args.loss_n
        self.test_all = args.test_all

        # Embedding layers for users and items
        self.user_embeddings = nn.Embedding(self.user_num, args.emb_size)
        self.item_embeddings = nn.Embedding(self.item_num, args.emb_size)
        self.skill_embeddings = nn.Embedding(self.skill_level_num + 1, args.emb_size)  # Skill level embeddings
        self.user_projection = nn.Linear(args.emb_size * 2, args.emb_size)
        self.skill_projection = nn.Linear(args.emb_size * 2, args.emb_size)

        # Task-specific heads
        self.ranking_head = nn.Linear(args.emb_size, 1)  # Task 1: Ranking
        self.skill_head = nn.Linear(args.emb_size, self.skill_level_num)  # Task 2: Skill level

        self.init_weights(self)

    def forward(self, feed_dict):
        # User and item IDs
        u_ids = feed_dict['user_id']
        i_ids = feed_dict['item_id']
        user_skill_level = feed_dict['skill_pred']
        skill_truth = feed_dict['skill_truth']

        # Shared embeddings
        user_emb = self.user_embeddings(u_ids)
        item_emb = self.item_embeddings(i_ids)
        skill_emb = self.skill_embeddings(user_skill_level)

        # Project user and skill embeddings to a common space
        combined_user_emb = torch.cat([user_emb, skill_emb], dim=-1)  # Concatenate embeddings
        combined_user_emb = self.user_projection(combined_user_emb)  # Learnable projection layer

        interaction_emb = torch.bmm(item_emb, combined_user_emb.unsqueeze(-1)).squeeze(-1)

        ranking_score = interaction_emb
        skill_input = torch.cat([user_emb, skill_emb], dim=-1)  # Concatenate embeddings
        skill_input = self.skill_projection(skill_input)  # Project to expected size: [batch_size, emb_size]
        skill_pred = self.skill_head(skill_input)  # Shape: [batch_size, num_skill_levels]

        out_dict = {
            'prediction': ranking_score,  # Task 1 output
            'skill_pred': skill_pred,  # Task 2 output
            'skill_truth': skill_truth
        }
        return out_dict
    
    def loss(self, out_dict: dict):
        """
        Combined BPR loss for ranking and CrossEntropy loss for skill-level prediction.

        BPR loss is used for ranking tasks, and CrossEntropy loss is used for skill-level prediction.
        """

        # Ranking loss: BPR or BCE, determined by self.loss_n
        if self.loss_n == 'BPR':
            # BPR Loss
            pos_scores = out_dict['prediction'][:, 0]  # First column: positive scores
            neg_scores = out_dict['prediction'][:, 1:]  # Remaining columns: negative scores
            ranking_loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores)).mean()
        elif self.loss_n == 'BCE':
            # BCE Loss
            predictions = out_dict['prediction'].sigmoid()
            pos_pred = predictions[:, 0]  # Positive predictions
            neg_pred = predictions[:, 1:]  # Negative predictions
            ranking_loss = - (pos_pred.log() + (1 - neg_pred).log().sum(dim=1)).mean()
        else:
            raise ValueError(f'Undefined loss function: {self.loss_n}')

        return ranking_loss

    class Dataset(BaseModel.Dataset):
        def _get_feed_dict(self, index):
            user_id = self.data['user_id'][index]
            target_item = self.data['item_id'][index]
            
            if self.phase != 'train' and self.model.test_all:
                neg_items = np.arange(1, self.corpus.n_items)
            else:
                neg_items = self.data['neg_items'][index]
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)

            # Retrieve user skill level from user_meta
            # Perform an explicit lookup using .loc
            skill_truth = (self.corpus.user_meta_df.loc[self.corpus.user_meta_df['user_id'] == user_id, 'u_skill_level_c'].values[0] - 1)
            
            # Dynamically compute predicted skill level using the RankSkillRunner logic
            duration, total_hours = utils.process_user_data(self.corpus.user_meta_df, user_id)
            skill_pred = utils.determine_skill_level(duration, total_hours)

            # Feed dict for ranking and skill prediction
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids,
                'skill_truth': skill_truth,  # Ground truth for skill level
                'skill_pred': skill_pred,  # Dynamically computed
            }
            return feed_dict

        def actions_before_epoch(self):
            # Sample negative items for ranking
            neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
            for i, u in enumerate(self.data['user_id']):
                clicked_set = self.corpus.train_clicked_set[u]  # Neg items may appear in dev/test
                for j in range(self.model.num_neg):
                    while neg_items[i][j] in clicked_set:
                        neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
            self.data['neg_items'] = neg_items

        def collate_batch(self, feed_dicts: List[dict]) -> dict:
            """
            Collate a batch of feed dictionaries into a single batch dictionary.
            
            Args:
                feed_dicts: List of dictionaries where each dictionary corresponds to a single sample.

            Returns:
                A single dictionary containing batched tensors.
            """
            feed_dict = dict()
            for key in feed_dicts[0]:
                if isinstance(feed_dicts[0][key], np.ndarray):
                    tmp_list = [len(d[key]) for d in feed_dicts]
                    if any([tmp_list[0] != l for l in tmp_list]):
                        stack_val = np.array([d[key] for d in feed_dicts], dtype=object)
                    else:
                        stack_val = np.array([d[key] for d in feed_dicts])
                else:
                    stack_val = np.array([d[key] for d in feed_dicts])
                if stack_val.dtype == object:  # Handle sequences with inconsistent lengths
                    feed_dict[key] = pad_sequence([torch.tensor(x) for x in stack_val], batch_first=True)
                else:
                    feed_dict[key] = torch.tensor(stack_val)

            # Special handling for skill_truth, if present
            if 'skill_truth' in feed_dict:
                feed_dict['skill_truth'] = feed_dict['skill_truth'].clone().detach().long()

            # Add metadata to the batch
            feed_dict['batch_size'] = len(feed_dicts)
            feed_dict['phase'] = self.phase
            return feed_dict

class RankSkillModelRankSkillTopK(RankSkillModel):
    """
    Specific implementation for the 'RankSkillTopK' mode.
    If no additional logic is needed, this can be an alias for RankSkillModel.
    """
    pass
