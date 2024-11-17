import utils
from utils import space
import random
from utils import *

log_file = f"./out/log/03_random.log"
logger = get_logger(log_file=log_file)


class Random:
    def __init__(self):
        random_env()
        # init 设计空间
        self.space = space.create_space()
        self.train_eps = 500

    def choose_action(self, dimension_index):
        scale = self.space.dimension_box[dimension_index].scale
        return random.randint(0, scale - 1)

    def train(self):
        for i in range(self.train_eps):
            logger.info(f"第{i + 1}个episode")
            self.space.states_reset()
            for j in range(self.space.len):
                action = self.choose_action(j)
                next_states = self.space.sample_one_dimension(j, action)
                if j == self.space.len - 1:
                    logger.info(f"第{i + 1}个episode,第{j + 1}个维度,action:{action},next_states:{next_states}")
                    evaluation(next_states)
                else:
                    logger.info(f"第{i + 1}个episode,第{j + 1}个维度,action:{action}")


if __name__ == "__main__":
    print(f"多核处理器设计空间探索!算法: Random")

    DSE = Random()
    DSE.train()

    print("DSE结束")
