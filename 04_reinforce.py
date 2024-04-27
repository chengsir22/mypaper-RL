import utils
import config
import space
import torch
from gem5_mcpat_evaluation import evaluation

class mlp_policyNet(torch.nn.Module):
    def __init__(self,space_length,action_scale_list):
        super(mlp_policyNet, self).__init__()
        self.space_length = space_length
        self.action_scale_list = action_scale_list
        self.fc1 = torch.nn.Linear(self.space_length+1, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        # layer fc3_list is a list of linear layers
        self.fc3_list = torch.nn.ModuleList([torch.nn.Linear(128, action_scale) for action_scale in self.action_scale_list])
    
    def forward(self, input, dimension_index):
        dimension_index_normalize = dimension_index / self.space_length
        x = torch.cat(
            (input, torch.tensor(dimension_index_normalize).float().view(-1,1)),
            dim = -1,
        )
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = self.fc3_list[dimension_index](x2)
        return torch.softmax(x3, dim=-1)
    

class Reinforce():
    def __init__(self):
        self.log_file = f"./out/log/04_reinforce.log"
        global logger
        logger = utils.setup_logger(log_file = self.log_file)
        utils.random_env()
        
        self.config = config.config_1()
        self.constraints = self.config.constraints
        self.config.config_check()
        
        self.space = space.create_space()
        
        action_scale_list = list()
        for dimension in self.space.dimension_box:
            action_scale_list.append(dimension.scale)
            
        # 模型选择
        self.device = torch.device("cpu")
        self.plicy_net = mlp_policyNet(self.space.len,action_scale_list).to(self.device)
        
        self.gamma = 0.98
        self.optimizer = torch.optim.Adam(self.plicy_net.parameters(), lr=0.001)
        
        self.train_eps = 500
        # self.train_eps = 1
        
    def take_action(self, dimension_index):
        states_normalize = utils.states_normalize(self.space)
        probs = self.plicy_net(utils.dict_to_tensor(states_normalize), dimension_index)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
        
    def train(self):
        for eps in range(self.train_eps):
            logger.info(f"Episode: {eps+1}")
            self.space.states_reset()
            
            reward_list = list()
            states_list = list()
            action_list =list()
            
            states = self.space.states
            
            for dimension_index in range(self.space.len):
                action = self.take_action(dimension_index)
                next_states = self.space.sample_one_dimension(dimension_index, action)
                if dimension_index <  (self.space.len-1):
                    reward = float(0)
                else:
                    metrics = evaluation(next_states)
                    if metrics is None:
                        reward = float(0)
                        logger.error(f"metrics is None")
                    else:
                        self.constraints.update({"area":metrics["area"]})
                        reward = 1000/(metrics["latency"]*metrics["area"]*metrics["power"]*self.constraints.get_punishment())
                reward_list.append(reward)
                action_list.append(action)
                states_list.append(states)
                states = next_states
            
            G = 0
            self.optimizer.zero_grad()
            for i in reversed(range(len(reward_list))):
                reward = reward_list[i]
                states = utils.dict_to_tensor(utils.states_normalize(self.space,states_list[i]))
                action = torch.tensor(action_list[i]).view(-1,1).to(self.device)
                log_prob = torch.log(self.plicy_net(states,i).gather(1,action))
                G=self.gamma*G+reward
                loss = -log_prob*G
                loss.backward()
            self.optimizer.step()
            
            if (eps+1) % 10 == 0:
                print(f"Episode: {eps+1}, Reward: {reward}")
                
if __name__ == "__main__":
    print(f"多核处理器设计空间探索!算法: Reinforce")
    reinforce = Reinforce()
    reinforce.train()
    print("训练结束!")