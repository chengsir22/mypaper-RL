import logging
import torch
import numpy
import random
from space import DesignSpace

def setup_logger(log_file,level=logging.INFO):
    # 创建日志记录器，指定日志级别
    logger = logging.getLogger("logger")
    logger.setLevel(level)
    # 创建文件处理器，指定日志文件地址
    handler = logging.FileHandler(log_file)
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    # 将处理器添加到日志记录器
    logger.addHandler(handler)
    return logger


def random_env(seed=1):
    # seed = int(time.time())
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    
    
# 状态归一化
def states_normalize(space:DesignSpace, states=None):
    res = dict()
    if states is None:
        for item in space.dimension_box:
            res[item.name] = item.value / item.sample_box[-1]
    else:
        for item in space.dimension_box:
            res[item.name] = states[item.name] / item.sample_box[-1]
    return res
        
def dict_to_tensor(states):
    return torch.tensor([value for value in states.values()],dtype=torch.float32)