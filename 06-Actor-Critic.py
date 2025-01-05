import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
import pandas as pd
from utils import *
from config import *


class PolicyNet(torch.nn.Module):
    def __init__(self, space_length, action_scale_list):
        super(PolicyNet, self).__init__()
        self.space_length = space_length
        self.action_scale_list = action_scale_list
        self.fc1 = torch.nn.Linear(self.space_length + 1, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        # layer fc3_list is a list of linear layers
        self.fc3_list = torch.nn.ModuleList(
            [torch.nn.Linear(128, action_scale) for action_scale in self.action_scale_list])

    def forward(self, input):
        x1 = torch.relu(self.fc1(input))
        x2 = torch.relu(self.fc2(x1))
        dimension_index = int(input[0][-1].item())  # 转为整数类型
        x3 = self.fc3_list[dimension_index](x2)
        return torch.softmax(x3, dim=-1)


class ValueNet(torch.nn.Module):
    def __init__(self, space_length, action_scale_list):
        super(ValueNet, self).__init__()
        self.space_length = space_length
        self.action_scale_list = action_scale_list
        self.fc1 = torch.nn.Linear(self.space_length + 1, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, input):
        x1 = torch.relu(self.fc1(input))
        x2 = torch.relu(self.fc2(x1))
        return self.fc3(x2)


class ActorCritic:
    def __init__(self, space_length, action_scale_list, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(space_length, action_scale_list).to(device)
        self.critic = ValueNet(space_length, action_scale_list).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        actor_loss = torch.tensor(0)
        critic_loss = torch.tensor(0)
        for _ in reversed(range(len(transition_dict['states']))):
            states = torch.tensor([transition_dict['states'][i]], dtype=torch.float).to(self.device)
            actions = torch.tensor([transition_dict['actions'][i]]).view(-1, 1).to(self.device)
            rewards = torch.tensor([transition_dict['rewards'][i]], dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor([transition_dict['next_states'][i]], dtype=torch.float).to(self.device)
            dones = torch.tensor([transition_dict['dones'][i]], dtype=torch.float).view(-1, 1).to(self.device)
            # 时序差分目标
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)  # 时序差分误差
            log_probs = torch.log(self.actor(states).gather(1, actions))
            actor_loss = torch.mean(-log_probs * td_delta.detach())
            # 均方误差损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数


if __name__ == "__main__":
    print("A2C start")

    log_file = f"./out/log/06_a2c.log"
    logger = get_logger(log_file=log_file)

    config = Config1()
    config.config_check()
    constraints = config.constraints

    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500

    gamma = 0.98
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    env = Simulation()
    torch.manual_seed(0)
    space_length = env.space_dim()
    action_scale_list = env.action_scale_list()
    agent = ActorCritic(space_length, action_scale_list, actor_lr, critic_lr, gamma, device)

    i_episode = 1

    reward_list = []

    while i_episode <= num_episodes:
        logger.info(f"Episode: {i_episode}")
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()
        flag = True
        for i in range(space_length):
            action = agent.take_action(state)
            next_state, metrics, done, info = env.step(action)

            if i == space_length - 1 and info is False:
                logger.info(f"state: {next_state} 运行失败")
                flag = False
                break

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)

            reward = 0
            if i == space_length - 1:
                constraints.update({"area": metrics["area"]})
                reward = 0.01 / (metrics["latency"] * constraints.get_punishment())
                logger.info(f"state: {next_state} 总 reward: {reward}")

            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward

        if flag is True:
            i_episode += 1
            agent.update(transition_dict)
            reward_list.append(episode_return)

        # 将 self.losses 转换为 DataFrame
        df = pd.DataFrame(reward_list, columns=["reward"])
        # 保存为 CSV 文件
        df.to_csv("rewards.csv", index=False)

    logger.info("训练结束!")
    print("训练结束!")


