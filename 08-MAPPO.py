import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from simulation import MAPPOSimulation
from utils import *
from config import *

# 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# Actor 网络
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, context):
        super(Actor, self).__init__()
        self.context = torch.tensor(context, dtype=torch.float).view(1, -1)  # 将 context 转为张量
        self.fc1 = nn.Linear(obs_dim + len(context), 128)
        self.fc2 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, action_dim)

    def forward(self, x):
        # 将 context 扩展到与 x 相同的 batch size
        batch_size = x.size(0)
        context_expanded = self.context.repeat(batch_size, 1)  # 扩展 context 使其与 x 的 batch_size 匹配
        # 拼接 context 和 x
        x = torch.cat((x, context_expanded), dim=-1)
        # print(x)
        # raise False
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs


# Critic 网络
class Critic(nn.Module):
    def __init__(self, total_obs_dim, context):
        super(Critic, self).__init__()
        self.context = torch.tensor(context, dtype=torch.float).view(1, -1)  # 将 context 转为张量
        self.fc1 = nn.Linear(total_obs_dim + len(context), 128)
        self.fc2 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        # 将 context 扩展到与 x 相同的 batch size
        batch_size = x.size(0)
        context_expanded = self.context.repeat(batch_size, 1)  # 扩展 context 使其与 x 的 batch_size 匹配
        # 拼接 context 和 x
        x = torch.cat((x, context_expanded), dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value = self.value_head(x)
        return state_value


# 存储经验的类
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []


# MAPPO 代理
class MAPPO:
    def __init__(self, action_scale_list, context):
        self.num_agents = len(action_scale_list)
        self.gamma = 0.98
        self.eps_clip = 0.2
        self.epochs = 5

        # 为每个智能体创建 Actor 网络
        obs_dim = 1
        self.actors = [Actor(obs_dim, action_dim, context).to(device) for action_dim in action_scale_list]

        # 共享 Critic 网络，输入为所有智能体的观察拼接
        total_obs_dim = len(action_scale_list)  # 所有智能体观察的总和
        self.critic = Critic(total_obs_dim, context).to(device)

        # 优化器，包含所有网络参数
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=3e-4) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = RolloutBuffer()  # 存储经验

    def select_action(self, states):
        actions = []
        action_log_probs = []

        for i, actor in enumerate(self.actors):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
            action_probs = actor(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            actions.append(action.item())
            action_log_probs.append(dist.log_prob(action))

        # Critic 评估当前全局状态价值
        state_concat = np.concatenate(states, axis=None)
        state_tensor = torch.FloatTensor(state_concat).unsqueeze(0).to(device)
        value = self.critic(state_tensor).item()

        return actions, action_log_probs, value

    def store_transition(self, states, actions, log_probs, rewards, dones, values):
        # 分离 log_probs 和 values
        detached_log_probs = [lp.detach() for lp in log_probs]

        self.buffer.states.append(states)
        self.buffer.actions.append(actions)
        self.buffer.log_probs.append(detached_log_probs)
        self.buffer.rewards.append(rewards)
        self.buffer.dones.append(dones)
        self.buffer.values.append(values)

    def compute_returns_and_advantages(self, next_value):
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values

        returns = []
        advantages = []
        G = next_value
        for step in reversed(range(len(rewards))):
            done_any = any(dones[step])
            reward = rewards[step][0]

            G = reward + self.gamma * G * (1 - done_any)
            returns.insert(0, G)
            advantage = G - values[step]
            advantages.insert(0, advantage)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        return returns, advantages

    def update(self, next_value):
        returns, advantages = self.compute_returns_and_advantages(next_value)

        # 转换为张量
        states = torch.FloatTensor(np.array(self.buffer.states)).to(device)  # [T, num_agents, obs_dim]
        actions = torch.FloatTensor(self.buffer.actions).to(device)  # [T, num_agents]
        old_log_probs = torch.stack([torch.stack(lp) for lp in self.buffer.log_probs]).to(device)  # [T, num_agents]
        old_log_probs = old_log_probs.view(-1)  # [T * num_agents]
        values = torch.FloatTensor(self.buffer.values).to(device)  # [T]

        for _ in range(self.epochs):
            # 优化每个 Actor
            for i, actor in enumerate(self.actors):
                actor_optimizer = self.actor_optimizers[i]
                # 重新计算 log_probs
                new_log_probs = []
                for t in range(states.size(0)):
                    state = states[t, i, :]
                    action_probs = actor(state.unsqueeze(0))
                    dist = torch.distributions.Categorical(action_probs)
                    log_prob = dist.log_prob(actions[t, i])
                    new_log_probs.append(log_prob)
                new_log_probs = torch.stack(new_log_probs)  # [T]

                # 计算比例
                ratios = torch.exp(new_log_probs - old_log_probs[i::self.num_agents])

                # 计算损失
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 反向传播和优化
                actor_optimizer.zero_grad()
                policy_loss.backward()
                actor_optimizer.step()

            # 优化 Critic
            self.critic_optimizer.zero_grad()
            state_concat = states.view(states.size(0), -1)  # [T, total_obs_dim]
            state_values = self.critic(state_concat).squeeze(-1)  # [T]
            value_loss = F.mse_loss(state_values, returns)
            value_loss.backward()
            self.critic_optimizer.step()

        # 清空记忆
        self.buffer.clear()


if __name__ == "__main__":
    print("MAPPO start")

    log_file = f"./out/log/08_MAPPO.log"
    logger = get_logger(log_file=log_file)

    config = Config2()
    config.config_check()
    constraints = config.constraints

    context = constraints.get_threshold_list()
    context = [x / 100 for x in context]

    env = MAPPOSimulation(constraints)
    torch.manual_seed(0)
    action_scale_list = env.action_scale_list()

    agent = MAPPO(action_scale_list, context)

    num_episodes = 500
    i_episode = 1
    reward_list = []
    while i_episode <= num_episodes:
        logger.info(f"Episode: {i_episode}")

        # 初始化环境和状态
        states = env.reset()
        done = False
        flag = True
        reward = 0
        while not done:
            actions, log_probs, value = agent.select_action(states)
            next_states, rewards, dones, info = env.step(actions)

            if info is False:
                logger.warning(f"state: {next_states} 运行失败")
                flag = False
                break

            done = any(dones)
            agent.store_transition(states, actions, log_probs, rewards, dones, value)
            states = next_states
            reward = rewards[0]
            logger.info(f"reward: {reward}")

        # 在每个回合结束后，进行网络更新
        # 假设下一个状态的值为0，因为回合结束
        if flag is True:
            i_episode += 1
            agent.update(next_value=0)
            reward_list.append(reward)

        # 将 self.losses 转换为 DataFrame
        df = pd.DataFrame(reward_list, columns=["reward"])
        # 保存为 CSV 文件
        df.to_csv("rewards.csv", index=False)

    logger.info("训练结束!")
    print("训练结束!")
