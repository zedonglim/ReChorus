# -*- coding: UTF-8 -*-

import os
import random
import logging
import torch
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, NoReturn, Any


def init_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def df_to_dict(df: pd.DataFrame) -> dict:
	res = df.to_dict('list')
	for key in res:
		res[key] = np.array(res[key])
	return res


def batch_to_gpu(batch: dict, device) -> dict:
	for c in batch:
		if type(batch[c]) is torch.Tensor:
			batch[c] = batch[c].to(device)
	return batch


def check(check_list: List[tuple]) -> NoReturn:
	# observe selected tensors during training.
	logging.info('')
	for i, t in enumerate(check_list):
		d = np.array(t[1].detach().cpu())
		logging.info(os.linesep.join(
			[t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]
		) + os.linesep)


def eval_list_columns(df: pd.DataFrame) -> pd.DataFrame:
	for col in df.columns:
		if pd.api.types.is_string_dtype(df[col]):
			df[col] = df[col].apply(lambda x: eval(str(x)))  # some list-value columns
	return df


def format_metric(result_dict: Dict[str, Any]) -> str:
	assert type(result_dict) == dict
	format_str = []
	metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
	topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys() if '@' in k])
	if not len(topks):
		topks = ['All']
	for topk in np.sort(topks):
		for metric in np.sort(metrics):
			name = '{}@{}'.format(metric, topk)
			m = result_dict[name] if topk != 'All' else result_dict[metric]
			if isinstance(m, (float, np.float32, np.float64)):
				format_str.append('{}:{:<.4f}'.format(name, m))
			elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
				format_str.append('{}:{}'.format(name, m))
	return ','.join(format_str)


def format_arg_str(args, exclude_lst: list, max_len=20) -> str:
	linesep = os.linesep
	arg_dict = vars(args)
	keys = [k for k in arg_dict.keys() if k not in exclude_lst]
	values = [arg_dict[k] for k in keys]
	key_title, value_title = 'Arguments', 'Values'
	key_max_len = max(map(lambda x: len(str(x)), keys))
	value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
	key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
	horizon_len = key_max_len + value_max_len + 5
	res_str = linesep + '=' * horizon_len + linesep
	res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
			   + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
	for key in sorted(keys):
		value = arg_dict[key]
		if value is not None:
			key, value = str(key), str(value).replace('\t', '\\t')
			value = value[:max_len-3] + '...' if len(value) > max_len else value
			res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
					   + value + ' ' * (value_max_len - len(value)) + linesep
	res_str += '=' * horizon_len
	return res_str


def check_dir(file_name: str):
	dir_path = os.path.dirname(file_name)
	if not os.path.exists(dir_path):
		print('make dirs:', dir_path)
		os.makedirs(dir_path)


def non_increasing(lst: list) -> bool:
	return all(x >= y for x, y in zip([lst[0]]*(len(lst)-1), lst[1:])) # update the calculation of non_increasing to fit ealry stopping, 2023.5.14, Jiayu Li


def get_time():
	return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def process_user_data(user_meta, user_id):
	"""
    Extract user-specific features such as duration and total hours from the user_meta DataFrame.

    Args:
    - user_meta (DataFrame): DataFrame containing user metadata.
    - user_id (int or str): The ID of the user to fetch data for.

    Returns:
    - duration (float): User's standardized duration in years.
    - total_hours (float): User's calculated total hours.
    """
	# Ensure user_id is correctly used based on user_meta structure
	if 'user_id' in user_meta.columns:
		user_data = user_meta.loc[user_meta['user_id'] == user_id]
	elif user_meta.index.name == 'user_id':
		user_data = user_meta.loc[user_id]
	else:
		raise KeyError(f"User ID {user_id} not found in user_meta.")
	if user_data.empty:
		raise ValueError(f"No data found for user_id {user_id}.")

	# Define mappings
	duration_mapping = {1: 0.5, 2: 2, 3: 5, 4: 8, 5: 13, 6: 18, 7: 25}
	frequency_mapping = {1: 0, 2: 2, 3: 4, 4: 6, 5: 7}
	duration_per_session_mapping = {1: 30, 2: 60, 3: 90, 4: 120, 5: 150}

	# Get user-specific metadata
	user_data = user_data.iloc[0]

	# Map numeric values to actual ranges
	duration = duration_mapping.get(user_data['u_duration_years_c'], 0)  # Duration in years
	frequency = frequency_mapping.get(user_data['u_exercise_frequency_c'], 0)  # Sessions per week
	session_duration = duration_per_session_mapping.get(user_data['u_average_duration_c'], 0)  # Minutes per session

	# Calculate total hours per week
	total_hours = (frequency * session_duration) / 60  # Convert minutes to hours

	return duration, total_hours

def determine_skill_level(duration, total_hours):
	"""
	Determines the skill level based on duration and total hours.
	Skill Levels (Zero-Based Indices):
	0 - Beginner
	1 - Proficient
	2 - Advanced
	3 - Expert
	"""
	if duration is None or total_hours is None:
		return 0  # Beginner
	if duration < 1:
		return 0  # Beginner
	elif 1 <= duration <= 3:
		return 1  # Proficient
	elif 3 < duration <= 5:
		return 2 if total_hours >= 12 else 1  # Advanced or Proficient
	elif duration > 5:
		return 3 if total_hours > 12 else (2 if total_hours >= 12 else 1)  # Expert, Advanced, or Proficient
	return 0  # Default to Beginner