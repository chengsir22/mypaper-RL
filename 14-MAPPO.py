import torch.nn.functional as F
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

                self.actor_optimizers[i].step()

            # 更新 Critic
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            self.critic_optimizer.step()


if __name__ == "__main__":
    print("MAPPO start")

    log_file = f"./out/log/14-MAPPO.log"
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


    logger.info("训练结束!")
    print("训练结束!")
