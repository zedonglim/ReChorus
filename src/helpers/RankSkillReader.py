# -*- coding: UTF-8 -*-
import logging
import ast
import numpy as np
import pandas as pd
import os

from helpers.BaseReader import BaseReader

class RankSkillReader(BaseReader):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--include_item_features', type=int, default=0,
                            help='Whether to include item context features (0 or 1).')
        parser.add_argument('--include_user_features', type=int, default=0,
                            help='Whether to include user context features (0 or 1).')
        parser.add_argument('--include_unseen_items', type=int, default=0,
                            help='Whether to include unseen items in the recommendation process (0 or 1).')
        return BaseReader.parse_data_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.include_item_features = args.include_item_features
        self.include_user_features = args.include_user_features
        self.include_unseen_items = args.include_unseen_items

        # Load metadata
        self._load_ui_metadata()
        self._collect_context()
        self._precompute_item_embeddings()

    def _load_ui_metadata(self):
        """
        Load user and item metadata.
        """
        item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
        user_meta_path = os.path.join(self.prefix, self.dataset, 'user_meta.csv')
        unseen_items_path = os.path.join(self.prefix, self.dataset, 'unseen_items.csv')

        # Load metadata conditionally
        self.item_meta_df = pd.read_csv(item_meta_path, sep=self.sep) if os.path.exists(item_meta_path) and self.include_item_features else None
        self.user_meta_df = pd.read_csv(user_meta_path, sep=self.sep) if os.path.exists(user_meta_path) and self.include_user_features else None
        self.unseen_items_df = pd.read_csv(unseen_items_path, sep=self.sep) if os.path.exists(unseen_items_path) and self.include_unseen_items else None

        # Extract feature names
        self.item_feature_names = sorted([c for c in self.item_meta_df.columns if c[:2] == 'i_']) if self.item_meta_df is not None else []
        self.user_feature_names = sorted([c for c in self.user_meta_df.columns if c[:2] == 'u_']) if self.user_meta_df is not None else []
        self.unseen_item_feature_names = sorted([c for c in self.unseen_items_df.columns if c[:2] == 'i_']) if self.unseen_items_df is not None else []

    def _collect_context(self):
        """
        Collect and process context features for user, item, and unseen item metadata.
        """
        logging.info('Collecting context features...')
        id_columns = ['user_id', 'item_id']
        self.item_features, self.user_features, self.unseen_item_features = None, None, None

        # Determine feature ranges for embeddings
        self.feature_max = dict()
        for key in ['train', 'dev', 'test']:
            logging.info(f'Loading context for {key} set...')
            ids_df = self.data_df[key][id_columns]
            for f in id_columns:
                self.feature_max[f] = max(self.feature_max.get(f, 0), ids_df[f].max() + 1)

        # Include item features
        if self.item_meta_df is not None and self.include_item_features:
            item_df = self.item_meta_df[['item_id'] + self.item_feature_names]
            self.item_features = item_df.set_index('item_id').to_dict(orient='index')
            for f in self.item_feature_names:
                self.feature_max[f] = max(self.feature_max.get(f, 0), self.item_meta_df[f].max() + 1)
            logging.info('# Item Features: %d'%(item_df.shape[1]))
        # Include user features
        if self.user_meta_df is not None and self.include_user_features:
            user_df = self.user_meta_df[['user_id'] + self.user_feature_names].set_index('user_id')
            self.user_features = user_df.to_dict(orient='index')
            for f in self.user_feature_names:
                self.feature_max[f] = max(self.feature_max.get(f, 0), self.user_meta_df[f].max() + 1)
            logging.info('# User Features: %d'%(user_df.shape[1]))
        # Include unseen items
        if self.unseen_items_df is not None and self.include_unseen_items:
            unseen_item_df = self.unseen_items_df[['item_id'] + self.unseen_item_feature_names]
            self.unseen_item_features = unseen_item_df.set_index('item_id').to_dict(orient='index')
            for f in self.unseen_item_feature_names:
                self.feature_max[f] = max(self.feature_max.get(f, 0), unseen_item_df[f].max() + 1)
            logging.info(f'# Unseen Items: {len(self.unseen_items_df)}')

    def _precompute_item_embeddings(self):
        """
        Precompute normalized embeddings for unseen items.
        """
        if self.unseen_items_df is None or 'bert_embeddings' not in self.unseen_items_df:
            logging.info("No valid bert_embeddings found in unseen items. Skipping precomputation.")
            return

        # Normalize bert_embeddings
        def preprocess_embeddings(embedding):
            embedding = np.array(ast.literal_eval(embedding)) if isinstance(embedding, str) else embedding
            return embedding / (np.linalg.norm(embedding) + 1e-9) if isinstance(embedding, np.ndarray) else None

        self.unseen_items_df['normalized_embeddings'] = self.unseen_items_df['bert_embeddings'].apply(preprocess_embeddings)

        # Filter out invalid embeddings
        valid_embeddings_mask = self.unseen_items_df['normalized_embeddings'].apply(
            lambda x: isinstance(x, np.ndarray) and x.shape == (768,)
        )
        self.unseen_items_df = self.unseen_items_df[valid_embeddings_mask]

        logging.info(f"Precomputed embeddings for {len(self.unseen_items_df)} unseen items.")

    def get_item_features(self, item_id):
        """
        Retrieve item features by item_id, prioritizing unseen items if applicable.
        """
        if item_id in (self.unseen_item_features or {}):
            return self.unseen_item_features[item_id]
        return (self.item_features or {}).get(item_id, {})
