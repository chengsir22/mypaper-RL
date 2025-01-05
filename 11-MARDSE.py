from collections import deque

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from simulation import MARDSESimulation
import pandas as pd
from utils import *
from config import *
import math


class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 128)  # 修改：输入维度增加1
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, action_dim)  # 统一动作空间大小

    def forward(self, input):
        x = torch.relu(self.fc1(input))  # 直接使用输入，不再拼接
        x = torch.relu(self.fc2(x))
        probs = F.softmax(self.fc3(x), dim=-1)
        return probs


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)

        # 线性层将输入状态映射到嵌入维度
        self.input_fc = nn.Linear(state_dim, 64)

        self.fc1 = torch.nn.Linear(64, 128)  # 修改：输入维度包括 dimension_index
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        # 将输入映射到嵌入空间
        x = self.input_fc(x)  # (batch_size, embed_dim)

        # 添加时间步维度以适应自注意力层
        # 假设每个输入是一个单一的时间步
        x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)

        # 自注意力层需要的输入形状是 (batch_size, seq_length, embed_dim)
        attn_output, attn_weights = self.self_attn(x, x, x)  # attn_output: (batch_size, seq_length, embed_dim)

        # 如果有多个时间步，可以对 attn_output 进行池化
        # 这里假设只有一个时间步，直接去除序列维度
        attn_output = attn_output.squeeze(1)  # (batch_size, embed_dim)

        x = F.relu(self.fc1(attn_output))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()  # 确保在CPU上进行计算
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


class SILBuffer:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def add(self, transition):
        """
        添加一个转移到缓冲区。如果缓冲区未满，则直接添加。
        如果缓冲区已满，则仅在新转移的奖励高于缓冲区中最低奖励的情况下添加。
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # 找到缓冲区中最低奖励的转移
            min_reward = min(t['reward'] for t in self.buffer)
            if transition['reward'] > min_reward:
                # 找到第一个具有最低奖励的转移并移除
                for idx, t in enumerate(self.buffer):
                    if t['reward'] == min_reward:
                        del self.buffer[idx]
                        break
                self.buffer.append(transition)
            # 如果新转移的奖励不高于任何现有转移，则不添加

    def sample(self, batch_size):
        """
        从缓冲区中随机采样 batch_size 个转移。如果缓冲区中的转移数量不足，则返回所有转移。
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def clear(self):
        self.buffer = deque(maxlen=self.capacity)


class MARDSE:
    def __init__(self, space_length, action_scale_list):
        self.num_agents = space_length
        self.gamma = 0.98
        self.lmbda = 0.98
        self.eps = 0.2  # PPO中截断范围的参数
        self.device = torch.device("cpu")
        self.K_epochs = 5

        obs_dim = 1
        self.actors = [PolicyNet(obs_dim, action_dim).to(self.device) for action_dim in action_scale_list]
        self.critic = ValueNet(space_length).to(self.device)  # 修改：ValueNet 输入维度增加1
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=3e-4) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # 初始化SIL缓冲区
        self.sil_buffer = SILBuffer(capacity=10)  # 阈值可根据实际情况调整
        self.sil_batch_size = 5
        self.sil_epochs = 3
        self.sil_loss_weight = 0.8  # SIL损失的权重
        
        # 定义梯度裁剪的最大范数
        # self.max_grad_norm = 1.0

    def take_action(self, states):
        actions = []
        for i in range(self.num_agents):
            state = torch.tensor(states[i], dtype=torch.float).unsqueeze(0).to(self.device)
            probs = self.actors[i](state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
            actions.append(action)
        return actions

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).to(self.device)  # [batch_size, num_agents]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        # 添加具有正优势的经验到SIL缓冲区
        # 在update方法中添加旧策略概率
        for i in range(len(transition_dict['rewards'])):
            if advantage[i].item() > 0:
                agent_states = states[:, i].unsqueeze(1)
                agent_actions = actions[:, i].unsqueeze(1)

                # 计算旧的 log 概率
                with torch.no_grad():
                    old_probs = self.actors[i](agent_states)
                    old_action_probs = old_probs.gather(1, agent_actions).squeeze(-1)
                    old_log_probs = torch.log(old_action_probs + 1e-10)  # 避免log(0)

                transition = {
                    'state': transition_dict['states'][i],
                    'action': transition_dict['actions'][i],
                    'reward': transition_dict['rewards'][i],
                    'next_state': transition_dict['next_states'][i],
                    'done': transition_dict['dones'][i],
                    'advantage': advantage[i].item(),
                    'old_log_prob': old_log_probs
                }
                self.sil_buffer.add(transition)

        for _ in range(self.K_epochs):
            # 为每个智能体分别计算损失
            for i in range(self.num_agents):
                agent_states = states[:, i].unsqueeze(1)
                agent_actions = actions[:, i].unsqueeze(1)

                # 计算旧的 log 概率
                with torch.no_grad():
                    old_probs = self.actors[i](agent_states)
                    old_action_probs = old_probs.gather(1, agent_actions).squeeze(-1)
                    old_log_probs = torch.log(old_action_probs + 1e-10)  # 避免log(0)

                # 计算当前的概率
                probs = self.actors[i](agent_states)
                action_probs = probs.gather(1, agent_actions).squeeze(-1)
                log_probs = torch.log(action_probs + 1e-10)

                # 计算比率
                ratio = torch.exp(log_probs - old_log_probs)

                # 计算 surrogate losses
                surr1 = ratio * advantage.view(-1)
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
                actor_loss = -torch.mean(torch.min(surr1, surr2))

                # 更新 Actor
                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                
                # 应用梯度裁剪
                # torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                
                self.actor_optimizers[i].step()

            # 更新 Critic
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            
            # 应用梯度裁剪
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

    def SILUpdate(self):
        if len(self.sil_buffer.buffer) < self.sil_batch_size:
            return  # 缓冲区中经验不足，跳过SIL更新

        logger.info(f"SILUpdate")
        
        for _ in range(self.sil_epochs):
            sil_samples = self.sil_buffer.sample(self.sil_batch_size)
            states = torch.tensor([sample['state'] for sample in sil_samples], dtype=torch.float).to(self.device)
            actions = torch.tensor([sample['action'] for sample in sil_samples], dtype=torch.long).to(self.device)
            rewards = torch.tensor([sample['reward'] for sample in sil_samples], dtype=torch.float).to(self.device)
            advantages = torch.tensor([sample['advantage'] for sample in sil_samples], dtype=torch.float).to(
                self.device)
            old_log_probs = torch.tensor([sample['old_log_prob'] for sample in sil_samples], dtype=torch.float).to(
                self.device)

            # 计算目标值，这里可以根据具体任务调整
            targets = rewards.unsqueeze(1)

            # 更新 Critic
            critic_preds = self.critic(states)
            critic_loss = F.mse_loss(critic_preds, targets.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            
            # 应用梯度裁剪
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        
            self.critic_optimizer.step()

            # 更新 Actor
            for i in range(self.num_agents):
                agent_states = states[:, i].unsqueeze(1)
                agent_actions = actions[:, i].unsqueeze(1)

                # 获取当前策略的概率
                probs = self.actors[i](agent_states)
                action_probs = probs.gather(1, agent_actions).squeeze(-1)
                log_probs = torch.log(action_probs + 1e-10)

                # 获取旧策略的log概率
                sample_old_log_probs = old_log_probs.view(-1)

                # 计算比率
                ratio = torch.exp(log_probs - sample_old_log_probs)

                # 计算 surrogate losses
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
                sil_actor_loss = -torch.mean(torch.min(surr1, surr2))
                
                sil_actor_loss = sil_actor_loss * self.sil_loss_weight  # scalar

                # 更新 Actor
                self.actor_optimizers[i].zero_grad()
                sil_actor_loss.backward()
                
                # 应用梯度裁剪
                # torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)

                self.actor_optimizers[i].step()


if __name__ == "__main__":
    print("MARDSE start")

    log_file = f"./out/log/11_MARDSE.log"
    logger = get_logger(log_file=log_file)

    # config = Config1()
    config = Config2()
    config.config_check()
    constraints = config.constraints

    env = MARDSESimulation()
    torch.manual_seed(0)
    space_length = env.space_length
    action_scale_list = env.action_scale_list

    agent = MARDSE(space_length, action_scale_list)

    return_list = []
    num_episodes = 500
    i_episode = 1

    while i_episode <= num_episodes:
        logger.info(f"Episode: {i_episode}")

        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []
        }

        # state 其实就是 【state,dimension_index】
        state = env.reset()

        # only one step
        actions = agent.take_action(state)
        next_state, metrics, done, info = env.step(actions)

        if info is False:
            logger.info(f"state: {next_state} 运行失败")
            continue

        constraints.update({"area": metrics["area"], "power": metrics["power"]})
        # reward = 0.01 / (metrics["latency"] * constraints.get_punishment())
        # reward = 1 / (math.sqrt(metrics["latency"] * metrics["power"]) * constraints.get_punishment())
        reward = 10 / (math.pow(metrics["latency"] * metrics["power"] * metrics["area"],
                                1 / 3) * constraints.get_punishment())

        logger.info(f"state: {next_state} 总 reward: {reward}")

        # 存储转移
        transition_dict['states'].append(state)
        transition_dict['actions'].append(actions)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)

        i_episode += 1
        return_list.append(reward)
        agent.update(transition_dict)

        # 将 self.losses 转换为 DataFrame
        df = pd.DataFrame(return_list, columns=["reward"])
        # 保存为 CSV 文件
        df.to_csv("rewards.csv", index=False)

        if i_episode % 20 == 0:
            agent.SILUpdate()
            
        if i_episode % 40 == 0:
            agent.sil_buffer.clear()

    logger.info("训练结束!")
    print("训练结束!")
