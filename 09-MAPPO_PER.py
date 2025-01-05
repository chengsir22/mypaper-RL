import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        return action_logits  # 返回 logits 以便后续使用Categorical分布

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

# 优先级经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.Transition = namedtuple('Transition',
                                     ('states', 'actions', 'log_probs', 'rewards', 'dones', 'values'))

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        batch = self.Transition(*zip(*samples))
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# MAPPO 代理（修改为离策略并使用优先级经验回放）
class MAPPO:
    def __init__(self, action_scale_list, context, buffer_capacity=100, batch_size=5, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.num_agents = len(action_scale_list)
        self.gamma = 0.98
        self.eps_clip = 0.2
        self.K_epochs = 10

        # 为每个智能体创建 Actor 网络
        obs_dim = 1
        self.actors = [Actor(obs_dim, action_dim, context).to(device) for action_dim in action_scale_list]

        # 共享 Critic 网络，输入为所有智能体的观察拼接
        total_obs_dim = len(action_scale_list)  # 所有智能体观察的总和
        self.critic = Critic(total_obs_dim, context).to(device)

        # 优化器，包含所有网络参数
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=3e-4) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 优先级经验回放缓冲区
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=alpha)
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # 用于调整beta

    def select_action(self, states):
        actions = []
        action_log_probs = []
        entropy = 0

        for i, actor in enumerate(self.actors):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
            logits = actor(state)
            dist = torch.distributions.Categorical(logits=F.softmax(logits, dim=-1))
            action = dist.sample()
            actions.append(action.item())
            log_prob = dist.log_prob(action)
            action_log_probs.append(log_prob.detach())  # 分离 log_prob
            entropy += dist.entropy().mean()

        # Critic 评估当前全局状态价值
        state_concat = np.concatenate(states, axis=None)
        state_tensor = torch.FloatTensor(state_concat).unsqueeze(0).to(device)
        value = self.critic(state_tensor).detach().item()  # 分离并转换为 float

        return actions, action_log_probs, value

    def store_transition(self, states, actions, log_probs, rewards, dones, values):
        # 确保所有 log_probs 都被分离，避免计算图的重复使用
        detached_log_probs = [lp.detach() for lp in log_probs]
        # 由于 values 是 float，无需分离
        transition = self.buffer.Transition(states, actions, detached_log_probs, rewards, dones, values)
        self.buffer.push(transition)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return  # 缓冲区不足以进行一次采样

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        batch, indices, weights = self.buffer.sample(self.batch_size, beta)

        # 转换为张量
        states = torch.FloatTensor(np.array(batch.states)).to(device)  # [batch_size, num_agents, obs_dim]
        actions = torch.LongTensor(batch.actions).to(device)  # [batch_size, num_agents]
        old_log_probs = torch.stack([torch.stack(lp) for lp in batch.log_probs]).to(device)  # [batch_size, num_agents]
        rewards = torch.FloatTensor(batch.rewards).to(device)  # [batch_size, num_agents]
        dones = torch.FloatTensor(batch.dones).to(device)  # [batch_size, num_agents]
        values = torch.FloatTensor(batch.values).to(device)  # [batch_size]

        # 计算当前值
        state_concat = states.view(self.batch_size, -1)  # [batch_size, total_obs_dim]
        state_values = self.critic(state_concat).squeeze(-1)  # [batch_size]

        # 计算 TD 目标
        returns = rewards[:,0] + self.gamma * (1 - dones[:,0]) * state_values.detach()
        advantages = returns - values

        # 优化每个 Actor
        for i, actor in enumerate(self.actors):
            actor_optimizer = self.actor_optimizers[i]

            # 重新计算 log_probs
            logits = actor(states[:, i, :])  # [batch_size, action_dim]
            dist = torch.distributions.Categorical(logits=F.softmax(logits, dim=-1))
            new_log_probs = dist.log_prob(actions[:, i])  # [batch_size]
            ratios = torch.exp(new_log_probs - old_log_probs[:, i])

            # 计算损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2) * weights
            policy_loss = policy_loss.mean()

            # 添加熵正则项以促进探索
            entropy_loss = -dist.entropy().mean()

            total_loss = policy_loss + 0.01 * entropy_loss

            # 反向传播和优化
            actor_optimizer.zero_grad()
            total_loss.backward()
            actor_optimizer.step()

        # 优化 Critic
        self.critic_optimizer.zero_grad()
        value_loss = F.mse_loss(state_values, returns)
        (value_loss * weights).mean().backward()
        self.critic_optimizer.step()

        # 更新经验的优先级
        td_errors = (returns.detach().cpu().numpy() - state_values.detach().cpu().numpy())
        new_priorities = np.abs(td_errors) + 1e-6  # 添加一个小常数以确保优先级非零
        self.buffer.update_priorities(indices, new_priorities)

    def save_models(self, path):
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f"{path}_actor_{i}.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load_models(self, path):
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(f"{path}_actor_{i}.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))

if __name__ == "__main__":
    print("MAPPO Off-Policy with Prioritized Experience Replay start")

    log_file = f"./out/log/09_MAPPO_off_policy.log"
    logger = get_logger(log_file=log_file)

    config = Config1()
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
        episode_reward = 0
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
            episode_reward += rewards[0]
            logger.info(f"reward: {rewards[0]}")

        # 在每个回合结束后，进行网络更新
        if flag is True:
            i_episode += 1
            agent.update()
            reward_list.append(episode_reward)

            # 将 rewards 存储为 CSV
            df = pd.DataFrame(reward_list, columns=["reward"])
            df.to_csv("rewards_off_policy.csv", index=False)

    logger.info("训练结束!")
    print("训练结束!")
