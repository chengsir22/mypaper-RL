import torch.nn.functional as F
import numpy as np
from simulation import TDMSimulation
import pandas as pd
from utils import *
from config import *
import math


class PolicyNet(torch.nn.Module):
    def __init__(self, space_length, action_scale_list):
        super(PolicyNet, self).__init__()
        self.space_length = space_length
        self.action_scale_list = action_scale_list
        # 输入维度增加1，用于dimension_index
        self.fc1 = torch.nn.Linear(self.space_length + 1, 128)  # 修改：输入维度增加1
        self.fc2 = torch.nn.Linear(128, 64)
        # layer fc3 是一个线性层，输出统一的最大动作空间大小
        self.max_action_scale = max(self.action_scale_list)  # 获取最大动作空间大小
        self.fc3 = torch.nn.Linear(64, self.max_action_scale)  # 统一动作空间大小

    def forward(self, input):
        """
        input: Tensor of shape (batch_size, space_length + 1)
        最后一个维度是 dimension_index (整数)
        """
        # 提取 state 和 dimension_index
        state = input[:, :-1]  # 提取状态
        dimension_index = input[:, -1].long()  # 提取 dimension_index 并转换为长整型

        x = torch.relu(self.fc1(input))  # 直接使用输入，不再拼接
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)  # 生成统一长度的 logits

        # 创建掩码，标记有效动作的位置为1，填充部分为0
        mask = torch.zeros_like(logits, dtype=torch.bool, device=input.device)
        for i, idx in enumerate(dimension_index.tolist()):
            mask[i, :self.action_scale_list[idx]] = 1  # 有效动作位置为1

        # 将无效动作的 logits 赋值为极低的值
        logits[~mask] = -1e9

        # 计算 softmax，只对有效动作进行
        probs = F.softmax(logits, dim=-1)

        return probs, mask  # 返回概率和掩码


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)  # 修改：输入维度包括 dimension_index
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
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


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, space_length, action_scale_list, actor_lr, critic_lr,
                 lmbda, eps, gamma, device):
        self.actor = PolicyNet(space_length, action_scale_list).to(device)
        self.critic = ValueNet(space_length + 1).to(device)  # 修改：ValueNet 输入维度增加1
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.epochs = 1

    def take_action(self, state_with_dim):
        """
        state_with_dim: numpy array of shape (space_length + 1,)
        最后一个元素是 dimension_index
        """
        state = torch.tensor([state_with_dim], dtype=torch.float).to(self.device)
        probs, _ = self.actor(state)  # 获取概率，不需要掩码
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        # actions = torch.tensor(transition_dict['actions'], dtype=torch.long).to(self.device)  # [batch_size, num_agents]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        # 计算旧的 log 概率
        with torch.no_grad():
            old_probs, old_masks = self.actor(states)  # 获取旧的概率和掩码
            # 选取对应的动作概率
            old_action_probs = old_probs.gather(1, actions).squeeze(-1)  # (batch_size,)
            # 使用掩码，只保留有效动作的概率
            # 由于无效动作已被赋予极低的概率，这里不需要额外的掩码处理
            old_log_probs = torch.log(old_action_probs + 1e-10)  # 防止log(0)

        for _ in range(self.epochs):
            # 计算当前的概率和掩码
            probs, masks = self.actor(states)
            # 选取对应的动作概率
            action_probs = probs.gather(1, actions).squeeze(-1)  # (batch_size,)
            log_probs = torch.log(action_probs + 1e-10)  # 防止log(0)

            # 计算比率
            ratio = torch.exp(log_probs - old_log_probs)

            # 计算 surrogate losses
            surr1 = ratio * advantage.view(-1)
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage.view(-1)
            actor_loss = -torch.mean(torch.min(surr1, surr2))  # PPO损失函数

            # 计算 critic loss
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            # 反向传播
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == "__main__":
    print("PPO-TDM start")

    log_file = f"./out/log/10-PPO-TDM.log"
    logger = get_logger(log_file=log_file)

    # config = Config1()
    config = Config2()
    config.config_check()
    constraints = config.constraints

    actor_lr = 3e-4
    critic_lr = 1e-3
    gamma = 0.98
    device = torch.device("cpu")

    env = TDMSimulation()
    torch.manual_seed(0)
    space_length = env.space_dim()
    action_scale_list = env.action_scale_list()
    agent = PPO(space_length, action_scale_list, actor_lr, critic_lr, 0.98, 0.2, gamma, device)

    return_list = []
    num_episodes = 500
    i_episode = 1

    while i_episode <= num_episodes:
        logger.info(f"Episode: {i_episode}")
        episode_return = 0
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []
        }

        # state 其实就是 【state,dimension_index】
        state = env.reset()
        done = False
        info = False
        reward = 0
        while not done:
            action = agent.take_action(state)
            next_state, metrics, done, info = env.step(action)

            if info is False:
                logger.info(f"state: {next_state} 运行失败")
                flag = False
                break

            if done:
                constraints.update({"area": metrics["area"], "power": metrics["power"]})
                reward = 0.01 / (metrics["latency"] * constraints.get_punishment())
                # reward = 1 / (math.sqrt(metrics["latency"] * metrics["power"]) * constraints.get_punishment())
                # reward = 10 / (math.pow(metrics["latency"] * metrics["power"] * metrics["area"],
                                # 1 / 3) * constraints.get_punishment())
                
                logger.info(f"state: {next_state} 总 reward: {reward}")

            # 存储转移
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            state = next_state
            episode_return += reward

        if info is True:
            i_episode += 1
            return_list.append(episode_return)
            agent.update(transition_dict)

            # 将 self.losses 转换为 DataFrame
            df = pd.DataFrame(return_list, columns=["reward"])
            # 保存为 CSV 文件
            df.to_csv("rewards.csv", index=False)

    logger.info("训练结束!")
    print("训练结束!")
