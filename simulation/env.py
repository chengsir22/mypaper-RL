import math

from .space import create_space
from .gem5_mcpat_evaluation import evaluation
import numpy as np


class TDMSimulation:
    def __init__(self):
        self.space = create_space()
        self.done = False  # 终止标志
        self.dim = 0
        self.info = True  # 模拟是否成功

    def reset(self):
        self.dim = 0
        self.info = True
        self.done = False
        return np.append(self.state, 0)

    @property
    def state(self):
        return np.array(list(self.space.states.values()))

    #  next_state, reward, done, 运行成功标志
    def step(self, action_index):
        self.space.sample_one_dimension(self.dim, action_index)

        metrics = None
        if self.dim == self.space.len - 1:
            self.done = True
            metrics = evaluation(self.space.states)

        self.dim += 1
        if self.done and metrics is None:
            self.info = False

        return np.append(self.state, self.dim), metrics, self.done, self.info

    def space_dim(self):
        return self.space.len

    def action_scale_list(self):
        action_scale_list = list()
        for dimension in self.space.dimension_box:
            action_scale_list.append(dimension.scale)
        return action_scale_list


class Simulation:
    def __init__(self):
        self.space = create_space()
        self.done = False  # 终止标志
        self.info = True  # 模拟是否成功

    def reset(self):
        self.info = True
        self.done = False
        return self.state

    @property
    def state(self):
        return np.array(list(self.space.states.values()))

    #  next_state, reward, done, 运行成功标志
    def step(self, actions):
        if len(actions) != self.space.len:
            print("严重错误，请立即检查 actions 维度")
            assert False

        for i in range(len(actions)):
            self.space.sample_one_dimension(i, actions[i])

        metrics = evaluation(self.space.states)

        if metrics is None:
            self.info = False

        return self.state, metrics, True, self.info

    def space_dim(self):
        return self.space.len

    def action_scale_list(self):
        action_scale_list = list()
        for dimension in self.space.dimension_box:
            action_scale_list.append(dimension.scale)
        return action_scale_list


class MARDSESimulation:
    def __init__(self):
        self.space = create_space()
        self.space_length = self.space.len

        action_scale_list = list()
        for dimension in self.space.dimension_box:
            action_scale_list.append(dimension.scale)

        self.action_scale_list = action_scale_list

    def reset(self):
        return self.state

    @property
    def state(self):
        return np.array(list(self.space.states.values()))

    #  next_state, metrics, done, 运行成功标志
    def step(self, actions):
        for i in range(len(actions)):
            self.space.sample_one_dimension(i, actions[i])

        metrics = evaluation(self.space.states)

        if metrics is None:
            return None, None, True, False

        return self.state, metrics, True, True


class MASimulation:
    def __init__(self):
        self.space = create_space()
        self.done = False  # 终止标志
        self.failure = False
        self.dim = 0

    def reset(self):
        self.dim = 0
        return self.state

    @property
    def state(self):
        return np.array(list(self.space.states.values()))

    #  next_state, reward, done, 运行成功标志
    def step(self, action_index):
        self.space.sample_one_dimension(self.dim, action_index)

        metrics = None
        done = False
        if self.dim == self.space.len - 1:
            done = True
            metrics = evaluation(self.space.states)

        self.dim += 1

        if done and metrics is None:
            return self.state, metrics, done, False
        else:
            return self.state, metrics, done, True

    def space_dim(self):
        return self.space.len

    def action_scale_list(self):
        action_scale_list = list()
        for dimension in self.space.dimension_box:
            action_scale_list.append(dimension.scale)
        return action_scale_list


class MAPPOSimulation:
    def __init__(self, constraints):
        self.space = create_space()
        self.num_agents = self.space.len
        self.constraints = constraints

    def reset(self):
        return self.state

    @property
    def state(self):
        return np.array(list(self.space.states.values())).reshape(-1, 1)

    #  next_states, rewards, dones, 运行成功标志
    def step(self, actions):
        if len(actions) != self.space.len:
            print("严重错误，请立即检查 actions 维度")
            assert False

        for i in range(len(actions)):
            self.space.sample_one_dimension(i, actions[i])

        metrics = evaluation(self.space.states)

        if metrics is None:
            return [], [], [], False
        else:
            reward = self.get_reward(metrics)
            rewards = [reward for _ in range(self.num_agents)]
            return self.state, rewards, [True for _ in range(self.num_agents)], True

    def get_reward(self, metrics):
        self.constraints.update({"area": metrics["area"], "power": metrics["power"]})
        # reward = 0.01 / (metrics["latency"]  * self.constraints.get_punishment())

        # reward = 1 / (math.sqrt(metrics["latency"] * metrics["power"]) * self.constraints.get_punishment())

        reward = 10 / (math.pow(metrics["latency"] * metrics["power"] * metrics["area"],
                                1 / 3) * self.constraints.get_punishment())
        return reward

    def action_scale_list(self):
        action_scale_list = list()
        for dimension in self.space.dimension_box:
            action_scale_list.append(dimension.scale)
        return action_scale_list


if __name__ == "__main__":
    print(f"模拟环境测试")
    env = Simulation()
    state = env.reset()
    print(state)
    print(env.step((1, 2)))
