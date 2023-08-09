import json
import os
import random

import numpy as np
import torch

from collections import defaultdict

from .config import Config


def fetch_to_tensor(dicts, dict_type, device):
    return torch.tensor(dicts[dict_type], dtype=torch.float, device=device)


def aggregate(adj_dict, incoming, other_type_num, softmax=False):
    aggregate_output = []
    for i in range(other_type_num):
        adj = adj_dict[str(0) + str(i + 1)]

        if softmax:
            adj = adj.masked_fill(adj.le(0), value=-1e9).softmax(-1)
        aggregate_output.append(torch.matmul(adj, incoming[i]) / (torch.sum(adj, dim=-1).unsqueeze(-1) + 1e-9))
    return aggregate_output


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


def save_res(config: Config, acc: float, f1: float):
    result = defaultdict(list)
    result[(acc, f1)] = \
        {
            'seed': config.seed,
            'weigh_decay': config.weight_decay,
            'lr': config.learning_rate,
            'drop_out': config.drop_out,
            'threshold': config.threshold,
            'dataset': config.dataset,
            'preprocessed': config.need_preprocess,
            'removed_stopwords': config.delete_stopwords,
        }

    os.makedirs(config.save_path, exist_ok=True)
    file_name = config.save_name
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as f:
            f.write(json.dumps({str(k): result[k] for k in result}, cls=MyEncoder, indent=4))
        return

    with open(file_name, mode='a') as f:
        f.write(json.dumps({str(k): result[k] for k in result}, cls=MyEncoder, indent=4))
