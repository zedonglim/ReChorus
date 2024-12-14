import os
import gc
import torch
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity

from utils import utils
from models.multi_task.RankSkillModel import RankSkillModel


class RankSkillRunner:
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=200,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=10,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='Optimizer: SGD, Adam, Adagrad, Adadelta.')
        parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of processors for DataLoader.')
        parser.add_argument('--pin_memory', type=int, default=0,
                            help='Use pinned memory in DataLoader.')
        parser.add_argument('--topk', type=str, default='5,10,20,50',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='NDCG,HR',
                            help='Metrics: NDCG, HR.')
        parser.add_argument('--main_metric', type=str, default='',
                            help='Main metric to determine the best model.')
        return parser

    def __init__(self, args):
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = [int(x) for x in args.topk.split(',')]
        self.metrics = [m.strip().upper() for m in args.metric.split(',')]
        self.main_metric = f'{self.metrics[0]}@{self.topk[0]}' if not args.main_metric else args.main_metric
        self.main_topk = int(self.main_metric.split("@")[1])
        self.time = None

        self.log_path = os.path.dirname(args.log_file) # path to save predictions
        self.save_appendix = args.log_file.split("/")[-1].split(".")[0] # appendix for prediction saving

    def train(self, data_dict: Dict[str, RankSkillModel.Dataset]):
        model = data_dict['train'].model
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)

        try:
            for epoch in range(self.epoch):
                # Training step
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                if np.isnan(loss):
                    logging.info(f"Loss is NaN. Stop training at epoch {epoch + 1}.")
                    break
                training_time = self._check_time()

                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                # Evaluation step
                dev_result = self.evaluate(data_dict['dev'], [self.main_topk], self.metrics)
                dev_results.append(dev_result)
                main_metric_results.append(dev_result[self.main_metric])
                logging_str = f'Epoch {epoch + 1:<5} loss={loss:<.4f} [{training_time:<3.1f} s] dev=({utils.format_metric(dev_result)})'

                # Test step during training
                if self.test_epoch > 0 and epoch % self.test_epoch == 0:
                    test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
                    logging_str += f' test=({utils.format_metric(test_result)})'
                testing_time = self._check_time()
                logging_str += f' [{testing_time:<.1f} s]'

                # Save model and apply early stopping
                if max(main_metric_results) == main_metric_results[-1]:
                    model.save_model()
                    logging_str += ' *'
                logging.info(logging_str)

                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info(f"Early stop at epoch {epoch + 1} based on dev result.")
                    break

        except KeyboardInterrupt:
            logging.info("Early stop manually.")
            exit_here = input("Exit completely without evaluation? (y/n) (default n): ")
            if exit_here.lower().startswith('y'):
                logging.info("Exit program.")
                exit(1)

        # Find the best dev result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(os.linesep + f"Best Iter(dev)={best_epoch + 1:>5}\t dev=({utils.format_metric(dev_results[best_epoch])}) [{self.time[1] - self.time[0]:<.1f} s]")
        model.load_model()

    def fit(self, dataset: RankSkillModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        dataset.actions_before_epoch()

        model.train()
        loss_list = list()
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc=f'Epoch {epoch:<3}', ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)

            # Randomly shuffle the items
            item_ids = batch['item_id']
            indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)
            batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]

            model.optimizer.zero_grad()
            out_dict = model(batch)

            # Restore predictions to the original order
            prediction = out_dict['prediction']
            if len(prediction.shape) == 2:  # Only for ranking tasks
                restored_prediction = torch.zeros(*prediction.shape).to(prediction.device)
                restored_prediction[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices] = prediction
                out_dict['prediction'] = restored_prediction

            loss = model.loss(out_dict)
            loss.backward()
            model.optimizer.step()
            loss_list.append(loss.detach().cpu().data.numpy())
        return np.mean(loss_list).item()

    def evaluate(self, dataset: RankSkillModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate ranking predictions and compute metrics for the given dataset.
        """
        predictions, _, _, _, _ = self.predict(dataset)
        return self.evaluate_ranking(predictions, topks, metrics)

    def evaluate_ranking(self, predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
        evaluations = dict()

        # Reshape predictions to ensure proper dimensions
        if predictions.ndim == 3 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)  # Remove the singleton dimension

        if predictions.ndim != 2:
            raise ValueError(f"Expected predictions to be 2D (num_users, num_items), but got shape {predictions.shape}.")

        # Evaluate on filtered predictions
        gt_rank = (predictions >= predictions[:, 0].reshape(-1, 1)).sum(axis=-1)
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = f'{metric}@{k}'
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                else:
                    raise ValueError(f'Undefined evaluation metric: {metric}')
        return evaluations

    # Pre-Scoring (Filter & Adjust Scores)
    @staticmethod
    def filter_and_adjust_scores(user_profile, unseen_items):
        """
        Filter and adjust scores for unseen items based on user profile.

        Args:
        - user_profile: dict containing user features.
        - unseen_items: DataFrame of unseen items.

        Returns:
        - filtered_items: DataFrame with adjusted scores.
        """
        u_prefer_category = user_profile['u_prefer_exercises_sports_c']  # 1: Both, 2: Exercises, 3: Sports
        unseen_items['adjusted_score'] = 1.0  # Initialize scores

        # Apply category preference adjustment
        for idx, row in unseen_items.iterrows():
            i_category = row['i_category_c']
            if u_prefer_category != 1 and u_prefer_category != i_category:
                unseen_items.at[idx, 'adjusted_score'] *= 0.8  # Slightly reduce score for mismatched categories

        return unseen_items

    # Base Profile Scoring
    @staticmethod
    def profile_based_scoring(unseen_items, user_skill_level):
        """
        Score unseen items based on user profile.

        Args:
        - unseen_items: DataFrame of unseen items.
        - user_skill_level: determined skill level.

        Returns:
        - unseen_items: DataFrame with profile-based scores added.
        """
        # Apply scoring logic
        for idx, row in unseen_items.iterrows():
            # Assume new sports/exercises start slightly below or at the user skill level
            unseen_items.at[idx, 'profile_score'] = max(1.0, user_skill_level - 1)

        return unseen_items

    # Embedding-Based Scoring
    @staticmethod
    def embedding_based_scoring(user_profile, unseen_items):
        """
        Compute embedding-based scores for unseen items.

        Args:
        - user_profile: dict containing user features (fitness goals, training types).
        - unseen_items: DataFrame of unseen items with `bert_embeddings`.

        Returns:
        - unseen_items: DataFrame with embedding-based scores added.
        """
        # Construct user preference vector (fitness goals + training types)
        user_preference_vector = np.array([
            user_profile.get(f'u_fitness_goals_{i}_c', 0) for i in range(1, 17)
        ] + [
            user_profile.get(f'u_preferred_training_types_{i}_c', 0) for i in range(1, 14)
        ])

        # Normalize user preference vector
        user_preference_vector = user_preference_vector / (np.linalg.norm(user_preference_vector) + 1e-9)

        # Project user preference vector to match the dimensionality of BERT embeddings (768)
        projection_layer = torch.nn.Linear(user_preference_vector.shape[0], 768)  # Map 29 -> 768
        projection_layer = projection_layer.to(torch.float32)  # Ensure matching dtype

        # Project the user vector
        user_preference_embedding = projection_layer(
            torch.tensor(user_preference_vector, dtype=torch.float32)
        ).detach().numpy().flatten()  # Shape (768,)

        # Normalize projected embedding
        user_preference_embedding /= (np.linalg.norm(user_preference_embedding) + 1e-9)

        # Precomputed normalized embeddings from `unseen_items`
        item_embeddings = np.vstack(unseen_items['normalized_embeddings'])

        # Compute similarity between user vector and item embeddings
        similarity_scores = cosine_similarity(item_embeddings, user_preference_embedding.reshape(1, -1)).flatten()

        unseen_items['embedding_score'] = similarity_scores
        return unseen_items

    # Hybrid Scoring
    @staticmethod
    def hybrid_scoring(unseen_items, profile_weight=0.5, embedding_weight=0.5):
        """
        Combine profile-based and embedding-based scores into a final score.

        Args:
        - unseen_items: DataFrame with profile and embedding scores.
        - profile_weight: Weight for profile-based scores.
        - embedding_weight: Weight for embedding-based scores.

        Returns:
        - unseen_items: DataFrame with final scores.
        """
        unseen_items['final_score'] = (
            profile_weight * unseen_items['profile_score'] +
            embedding_weight * unseen_items['embedding_score'] +
            unseen_items['adjusted_score']
        )
        return unseen_items.sort_values(by='final_score', ascending=False)
    
    def predict(self, dataset: RankSkillModel.Dataset) -> tuple:
        """
        Predict both ranking scores and skill-level outputs for the given dataset.
        :param dataset: The dataset to evaluate.
        :return: Tuple of (predictions, skill_predictions, skill_truths).
        """
        dataset.model.eval()  # Set the model to evaluation mode
        predictions, skill_predictions, skill_truths, rec_unseen_items, unseen_items_score = list(), list(), list(), list(), list()

        # Unseen items to evaluate
        unseen_items = dataset.corpus.unseen_items_df if hasattr(dataset.corpus, 'unseen_items_df') else None
        if unseen_items is not None and dataset.corpus.include_unseen_items:
            logging.info(f'Loaded {len(unseen_items)} unseen items for prediction.')
            unseen_items = unseen_items.copy()

        # Create a DataLoader for the dataset
        dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False,
                        num_workers=self.num_workers, collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
        
        # Loop through batches in the dataset
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            # Move the batch to GPU if necessary
            batch = utils.batch_to_gpu(batch, dataset.model.device)
            
            # Forward pass through the model
            out_dict = dataset.model(batch)
            batch_predictions = out_dict['prediction'].cpu().data.numpy()
            predictions.append(batch_predictions)

            # Collect skill-level predictions and truths
            skill_predictions.extend(out_dict['skill_pred'].cpu().data.numpy())
            skill_truths.extend(batch['skill_truth'].cpu().data.numpy())
            
            # Process unseen items for each user
            if unseen_items is not None and dataset.corpus.include_unseen_items:
                for user_id in batch['user_id']:
                    user_id = user_id.item()  # Convert 0-d tensor to Python int
                    user_row = dataset.corpus.user_meta_df[dataset.corpus.user_meta_df['user_id'] == user_id]
                    if user_row.empty:
                        continue  # Skip if user profile not found
                    user_profile = user_row.iloc[0]

                    # Reset temporary columns for the current user
                    unseen_items['adjusted_score'] = 1.0  # Reset for each user
                    unseen_items['profile_score'] = 0.0
                    unseen_items['embedding_score'] = 0.0
                    unseen_items['final_score'] = 0.0
                    
                    # Compute user skill level
                    duration, total_hours = utils.process_user_data(dataset.corpus.user_meta_df, user_id)
                    user_skill_level = utils.determine_skill_level(duration, total_hours)

                    # Update unseen_items with user-specific scores
                    unseen_items = self.filter_and_adjust_scores(user_profile, unseen_items)
                    unseen_items = self.profile_based_scoring(unseen_items, user_skill_level)
                    unseen_items = self.embedding_based_scoring(user_profile, unseen_items)
                    unseen_items = self.hybrid_scoring(unseen_items)

                    # Append top-k unseen items to predictions
                    # top_k_unseen = unseen_items.nlargest(20, 'final_score')   # Get the top 20 items
                    top_k_unseen = unseen_items.head(self.topk[0])
                    if not top_k_unseen.empty:
                        unseen_items_score.append(top_k_unseen['final_score'].values)
                        rec_unseen_items.append(top_k_unseen['item_id'].values)
                    else:
                        unseen_items_score.append([])
                        rec_unseen_items.append([])

        # Convert lists to numpy arrays
        predictions = np.vstack(predictions)
        rec_unseen_items = np.array(rec_unseen_items)
        unseen_items_score = np.array(unseen_items_score)
        skill_predictions = np.array(skill_predictions)
        skill_truths = np.array(skill_truths)
        
        return predictions, skill_predictions, skill_truths, rec_unseen_items, unseen_items_score

    def print_res(self, dataset: RankSkillModel.Dataset) -> str:
        result_dict = self.evaluate(dataset, self.topk, self.metrics)
        res_str = '(' + utils.format_metric(result_dict) + ')'
        return res_str

    def _build_optimizer(self, model):
        logging.info(f'Optimizer: {self.optimizer_name}')
        optimizer = eval(f'torch.optim.{self.optimizer_name}')(
            model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        return optimizer

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def eval_termination(self, criterion: List[float]) -> bool:
        """
        Determine whether to stop training early based on dev performance.
        """
        if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False
