import torch.nn.functional as F
import numpy as np
from simulation import Simulation
import pandas as pd
from utils import *
from config import *


class PolicyNet(torch.nn.Module):
    def __init__(self, space_length, action_scale_list):
        super(PolicyNet, self).__init__()
        self.space_length = space_length
        self.action_scale_list = action_scale_list
        self.num_params = len(action_scale_list)

        self.fc1 = torch.nn.Linear(space_length, 128)
        self.fc2 = torch.nn.Linear(128, 64)

        # 为每个参数维度创建一个独立的输出层
        self.action_heads = torch.nn.ModuleList([
            torch.nn.Linear(64, scale) for scale in action_scale_list
        ])

    def forward(self, state):
        """
        state: Tensor of shape (batch_size, space_length)
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        action_probs = []
        for head in self.action_heads:
            logits = head(x)
            probs = F.softmax(logits, dim=-1)
            action_probs.append(probs)

        return action_probs  # List of tensors, each of shape (batch_size, action_scale)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
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
        self.critic = ValueNet(space_length).to(device)  # 修改：ValueNet 输入维度增加1
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.epochs = 1
        self.num_params = len(action_scale_list)

    def take_action(self, state):
        """
        state: numpy array of shape (space_length,)
        """
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action_probs = self.actor(state)  # List of tensors
        actions = []
        log_probs = []
        for i, probs in enumerate(action_probs):
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
        return actions, torch.stack(log_probs).sum()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).to(
            self.device)  # Shape: (batch_size, num_params)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).unsqueeze(1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).unsqueeze(1)

        # 计算 TD 目标和 TD 差分
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        # 计算旧的 log 概率
        with torch.no_grad():
            old_action_probs = []
            for i in range(self.num_params):
                probs = self.actor(states)[i]
                dist = torch.distributions.Categorical(probs)
                old_action_probs.append(dist.log_prob(actions[:, i]))
            old_log_probs = torch.stack(old_action_probs, dim=1).sum(dim=1)

        for _ in range(self.epochs):
            action_probs = self.actor(states)
            log_probs = []
            for i in range(self.num_params):
                dist = torch.distributions.Categorical(action_probs[i])
                log_probs.append(dist.log_prob(actions[:, i]))
            new_log_probs = torch.stack(log_probs, dim=1).sum(dim=1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage.view(-1)
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage.view(-1)
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            # 反向传播
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == "__main__":
    print("13-PPO start")

    log_file = f"./out/log/13-PPO.log"
    logger = get_logger(log_file=log_file)

    # config = Config1()
    config = Config2()
    config.config_check()
    constraints = config.constraints

    actor_lr = 3e-4
    critic_lr = 1e-3
    gamma = 0.98
    device = torch.device("cpu")

    env = Simulation()
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

        # state 是环境的初始状态
        state = env.reset()
        done = False
        info = False
        reward = 0

        while not done:
            actions, log_prob = agent.take_action(state)
            next_state, metrics, done, info = env.step(actions)

            if not info:
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
            transition_dict['actions'].append(actions)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            state = next_state
            episode_return += reward

        if info:
            i_episode += 1
            return_list.append(episode_return)
            agent.update(transition_dict)

            # 将 return_list 转换为 DataFrame
            df = pd.DataFrame(return_list, columns=["reward"])
            # 保存为 CSV 文件
            df.to_csv("rewards.csv", index=False)

    logger.info("训练结束!")
    print("训练结束!")
