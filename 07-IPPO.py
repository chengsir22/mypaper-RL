import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from simulation import MASimulation
import pandas as pd
from utils import *
from config import *
import math


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, context):
        super(PolicyNet, self).__init__()
        self.context = torch.tensor(context, dtype=torch.float).view(1, -1)  # 将 context 转为张量
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, action_dim)

    def forward(self, x):
        # 将 context 扩展到与 x 相同的 batch size
        batch_size = x.size(0)
        context_expanded = self.context.repeat(batch_size, 1)  # 扩展 context 使其与 x 的 batch_size 匹配
        # 拼接 context 和 x
        x = torch.cat((x, context_expanded), dim=-1)
        # print(x)
        # raise False
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, context):
        super(ValueNet, self).__init__()
        self.context = torch.tensor(context, dtype=torch.float).view(1, -1)  # 将 context 转为张量
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        # 将 context 扩展到与 x 相同的 batch size
        batch_size = x.size(0)
        context_expanded = self.context.repeat(batch_size, 1)  # 扩展 context 使其与 x 的 batch_size 匹配
        # 拼接 context 和 x
        x = torch.cat((x, context_expanded), dim=-1)

        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, space_length, context, action_dim, actor_lr, critic_lr,
                 lmbda, eps, gamma, device):
        self.actor = PolicyNet(1 + len(context), action_dim, context).to(device)
        self.critic = ValueNet(1 + len(context), context).to(device)
        # self.actor = PolicyNet(1, action_dim, context).to(device)
        # self.critic = ValueNet(1, context).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        log_probs = torch.log(self.actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps,
                            1 + self.eps) * advantage  # 截断
        actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


if __name__ == "__main__":
    print("IPPO start")

    log_file = f"./out/log/07_IPPO.log"
    logger = get_logger(log_file=log_file)

    config = Config2()
    config.config_check()
    constraints = config.constraints

    context = constraints.get_threshold_list()
    context = [x / 100 for x in context]

    actor_lr = 3e-4
    critic_lr = 1e-3
    num_episodes = 500

    gamma = 0.98
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    env = MASimulation()
    torch.manual_seed(0)
    space_length = env.space_dim()
    action_scale_list = env.action_scale_list()
    # agent = ActorCritic(space_length, action_scale_list, actor_lr, critic_lr, gamma, device)

    agents = []
    for i in range(space_length):
        agents.append(
            PPO(space_length, context, action_scale_list[i], actor_lr, critic_lr, 0.98, 0.2, gamma, device)
        )

    i_episode = 1

    reward_list = []

    while i_episode <= num_episodes:
        logger.info(f"Episode: {i_episode}")
        episode_return = 0
        transition_dicts = []
        for i in range(space_length):
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            transition_dicts.append(transition_dict)

        state = env.reset()
        flag = True

        reward = 0
        for i in range(space_length):
            action = agents[i].take_action([state[i]])
            next_state, metrics, done, info = env.step(action)

            # print(next_state, metrics, done, info)

            if i == space_length - 1 and info is False:
                logger.info(f"state: {next_state} 运行失败")
                flag = False
                break

            transition_dicts[i]['states'].append([state[i]])
            transition_dicts[i]['actions'].append(action)
            transition_dicts[i]['next_states'].append([next_state[i]])

            if i == space_length - 1:
                constraints.update({"area": metrics["area"], "power": metrics["power"]})
                # reward = 0.01 / (metrics["latency"] * constraints.get_punishment())
                reward = 1 / (math.sqrt(metrics["latency"] * metrics["power"]) * constraints.get_punishment())
                logger.info(f"state: {next_state} 总 reward: {reward}")

        if flag is True:
            i_episode += 1
            for i in range(space_length):
                for j in range(len(transition_dicts[i]['states'])):
                    transition_dicts[i]['rewards'].append(reward)
                    transition_dicts[i]['dones'].append(True)

            for i in range(space_length):
                agents[i].update(transition_dicts[i])
            reward_list.append(reward)

        # 将 self.losses 转换为 DataFrame
        df = pd.DataFrame(reward_list, columns=["reward"])
        # 保存为 CSV 文件
        df.to_csv("rewards.csv", index=False)

    logger.info("训练结束!")
    print("训练结束!")
