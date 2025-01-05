import numpy
import torch
import random

from simulation.space import DesignSpace


def random_env(seed=1):
    # seed = int(time.time())
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


# 状态归一化
def states_normalize(space: DesignSpace, states=None):
    res = dict()
    if states is None:
        for item in space.dimension_box:
            res[item.name] = item.value / item.sample_box[-1]
    else:
        for item in space.dimension_box:
            res[item.name] = states[item.name] / item.sample_box[-1]
    return res


def dict_to_tensor(states):
    return torch.tensor([[value for value in states.values()]], dtype=torch.float)


def dict_to_list_tensor(states):
    return torch.tensor([value for value in states.values()], dtype=torch.float)
