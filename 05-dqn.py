import pandas as pd
import torch
from utils import *
from net import QNet
import numpy as np
import collections

log_file = f"./out/log/05_dqn.log"
logger = get_logger(log_file=log_file)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, step, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, step, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, step, reward, next_state, done = zip(*transitions)
        return state, action, step, reward, next_state, done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class DQN:
    def __init__(self):
        random_env()
        self.config = Config1()
        self.constraints = self.config.constraints
        self.config.config_check()

        self.space = create_space()
        action_scale_list = list()
        for dimension in self.space.dimension_box:
            action_scale_list.append(dimension.scale)

        self.device = torch.device("cpu")

        self.q_net = QNet(self.space.len, action_scale_list).to(self.device)
        self.target_q_net = QNet(self.space.len, action_scale_list).to(self.device)
        self.replay_buffer = ReplayBuffer(1000)

        self.batch_size = 5
        self.target_update = 5  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数

        self.gamma = 0.98
        self.epsilon = 0.01
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss()

        self.losses = []
        self.rewards = []

        self.train_eps = 500

    def take_action(self, dimension_index):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.space.len)
        else:
            best_action_index = 0
            best_qvalue = 0
            for index in range(self.space.dimension_box[dimension_index].scale):
                self.space.sample_one_dimension(dimension_index, index)
                with torch.no_grad():
                    state = dict_to_tensor(states_normalize(self.space))
                    x = torch.cat(
                        (state, torch.tensor(dimension_index).float().view(-1, 1)),
                        dim=-1,
                    )
                    qvalue = self.q_net(x)

                if qvalue > best_qvalue:
                    best_action_index = index
                    best_qvalue = qvalue
            return best_action_index

    def update(self, transition_dict):
        print(transition_dict)
        for i in range(self.batch_size):

            states = transition_dict['states'][i]
            actions = torch.tensor([transition_dict['actions'][i]]).view(-1, 1).to(self.device)
            step = torch.tensor([transition_dict['step'][i]]).view(-1, 1).to(self.device)
            rewards = torch.tensor([transition_dict['rewards'][i]], dtype=torch.float).view(-1, 1).to(self.device)
            next_states = transition_dict['next_states'][i]
            dones = torch.tensor([transition_dict['dones'][i]], dtype=torch.float).view(-1, 1).to(self.device)

            x = torch.cat(
                (states, step),
                dim=-1,
            )
            q_values = self.q_net(x)
            # 初始化 next_q_values 为0（如果回合结束）
            next_q_values = torch.zeros_like(rewards)

            if dones[0] == 0:
                with torch.no_grad():
                    x2 = torch.cat(
                        (next_states, step + 1),
                        dim=-1,
                    )
                    next_q_values = self.target_q_net(x2)

            q_targets = rewards + self.gamma * next_q_values * (1 - dones)
            loss = self.loss_fn(q_values, q_targets.detach())
            print(f"Loss: {loss}")
            self.losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.count % self.target_update == 0:
                self.target_q_net.load_state_dict(
                    self.q_net.state_dict())  # 更新目标网络
            self.count += 1

    def train(self):
        for eps in range(self.train_eps):
            logger.info(f"Episode: {eps + 1}")
            self.space.states_reset()

            for dimension_index in range(self.space.len):
                state = self.space.states
                action = self.take_action(dimension_index)
                next_states = self.space.sample_one_dimension(dimension_index, action)
                logger.info(f"动作维度：{dimension_index}, 动作索引：{action}")

                done = 0
                reward = float(0)

                if dimension_index < (self.space.len - 1):
                    metrics = evaluation(next_states)
                    if metrics is None:
                        reward = float(0)
                        logger.error(f"metrics is None")
                    else:
                        self.constraints.update({"area": metrics["area"]})
                        reward = 1000 / (metrics["latency"] * metrics["area"] * metrics[
                            "power"] * self.constraints.get_punishment())
                else:
                    done = 1
                    metrics = evaluation(next_states)
                    if metrics is None:
                        reward = float(0)
                        logger.error(f"metrics is None")
                    else:
                        self.constraints.update({"area": metrics["area"]})
                        reward = 1000 / (metrics["latency"] * metrics["area"] * metrics[
                            "power"] * self.constraints.get_punishment())
                    self.rewards.append(reward)

                state = dict_to_tensor(states_normalize(self.space, state))
                next_states = dict_to_tensor(states_normalize(self.space, next_states))
                self.replay_buffer.add(state, action, dimension_index, reward, next_states, done)
                if self.replay_buffer.size() >= self.batch_size:
                    b_s, b_a, b_step, b_r, b_ns, b_d = self.replay_buffer.sample(self.batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'step': b_step,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    self.update(transition_dict)

        # 将 self.losses 转换为 DataFrame
        df = pd.DataFrame(self.losses, columns=["loss"])
        # 保存为 CSV 文件
        df.to_csv("losses.csv", index=False)
        # 将 self.losses 转换为 DataFrame
        df = pd.DataFrame(self.rewards, columns=["reward"])
        # 保存为 CSV 文件
        df.to_csv("rewards.csv", index=False)


if __name__ == "__main__":
    print(f"多核处理器设计空间探索!算法: DQN")
    dqn = DQN()
    dqn.train()

    print("训练结束!")
