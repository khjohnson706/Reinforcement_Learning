import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import matplotlib.pyplot as plt


torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

class ReplayBuffer():
    def __init__(self,param):
        super(ReplayBuffer,self).__init__()
        self.capacity = param.buffer_capacity
        self.buffer = []
        
        self.num_transitions = 0
        self.index = 0
    
    def update_buffer(self, state, action, reward, next_state, done):
        
        if self.num_transitions < self.capacity:
          self.buffer.append(None)
        
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity
        self.num_transitions += 1

    def sample_buffer(self,param):
        self.batch_size = param.batch_size
        batch = random.sample(self.buffer, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = map(np.stack, zip(*batch))
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

class Actor(nn.Module):
    def __init__(self,param):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(param.state_dim, param.hidden_layers)
        self.fc2 = nn.Linear(param.hidden_layers,param.hidden_layers)
        self.avg_layer = nn.Linear(param.hidden_layers,param.action_dim)
        self.logstd_dev_layer = nn.Linear(param.hidden_layers,param.action_dim)

    def forward(self,x): #determines a gaussian representation for an action
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        avg = self.avg_layer(x2)
        logstddev = self.logstd_dev_layer(x2)
        logstddev = torch.clamp(logstddev,-10, 2)
        return avg, logstddev


class Critic(nn.Module): #determines a state value
    def __init__(self,param):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(param.state_dim,param.hidden_layers)
        self.fc2 = nn.Linear(param.hidden_layers,param.hidden_layers)
        self.fc3 = nn.Linear(param.hidden_layers,1)

    def forward(self,x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2) 
        return x3

class Q(nn.Module): #determine Q value
    def __init__(self,param):
        super(Q,self).__init__()
        self.fc1 = nn.Linear(param.state_dim + param.action_dim,param.hidden_layers)
        self.fc2 = nn.Linear(param.hidden_layers,param.hidden_layers)
        self.fc3 = nn.Linear(param.hidden_layers,1)

    def forward(self,state,action):
        x = torch.cat((state,action),-1)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3

class SAC():
    def __init__(self,param):
        super(SAC,self).__init__()
        self.policy_nn = Actor(param).to(param.device)
        self.value_nn = Critic(param).to(param.device)
        self.TargetValue_nn = Critic(param).to(param.device)
        self.Q1_nn = Q(param).to(param.device)
        self.Q2_nn = Q(param).to(param.device)

        self.policy_optim = optim.Adam(self.policy_nn.parameters(), lr=param.learn_rate)
        self.value_optim = optim.Adam(self.value_nn.parameters(), lr=param.learn_rate)
        self.Q1_optim = optim.Adam(self.Q1_nn.parameters(), lr=param.learn_rate)
        self.Q2_optim = optim.Adam(self.Q2_nn.parameters(), lr=param.learn_rate)
        self.replay_buffer = ReplayBuffer(param)

        self.num_training_episodes = 0
        

        self.value_criteria = nn.MSELoss()
        self.Q1_criteria = nn.MSELoss()
        self.Q2_criteria = nn.MSELoss()
        for targ_param, parameters in zip(self.TargetValue_nn.parameters(),self.value_nn.parameters()):
            targ_param.data.copy_(parameters.data)

    def choose_action(self,state):
        State = torch.FloatTensor(state).to(param.device)
        avg, log_stddev = self.policy_nn.forward(State)
        stddev = torch.exp(log_stddev)
        #print(f"mean {avg}")
        #print(f"std dev {stddev}")
        distribution = Normal(avg,stddev)
        z = distribution.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action

    def evaluate(self,state):
        batch_avg, batch_log_stddev = self.policy_nn.forward(state)
        batch_stddev = torch.exp(batch_log_stddev)
        dist = Normal(batch_avg,batch_stddev)
        noise = Normal(0,1)
        z = noise.sample()
        action = torch.tanh(batch_avg+batch_stddev*z.to(param.device)) 
        log_prob = dist.log_prob(batch_avg+batch_stddev*z.to(param.device)) - torch.log(1-action.pow(2) + 1e-7)
        return action, log_prob, batch_avg, batch_log_stddev

    def batch_training(self):
        b_state, b_action, b_rew, b_ns, b_done = self.replay_buffer.sample_buffer(param)
        b_state = torch.tensor(b_state).float().to(param.device)
        b_action = torch.tensor(b_action).float().to(param.device)
        b_rew = torch.tensor(b_rew).float().to(param.device)
        b_ns = torch.tensor(b_ns).float().to(param.device)
        b_done = torch.tensor(b_done).float().to(param.device)

        target_value_ns = self.TargetValue_nn(b_ns)
        
        next_q_hat = b_rew + (1-b_done)*param.gamma * target_value_ns #if next state is terminal, just reward
        
        value_expected = self.value_nn(b_state)
        Q1_expected = self.Q1_nn(b_state,b_action)
        Q2_expected = self.Q2_nn(b_state,b_action)

        sample_act, log_prob, batch_avg, batch_stddev = self.evaluate(b_state)
        log_prob1 = torch.div(log_prob,param.action_dim)
        #print(log_prob1.size())
        log_prob2 = torch.norm(log_prob1, dim=1)
        #print(log_prob2.size())
        log_prob3 = torch.reshape(log_prob2,(param.batch_size,1))
        #print(log_prob3.size())
        #print(value_expected.size())


        Q_expected_new = torch.min(self.Q1_nn(b_state,sample_act),self.Q2_nn(b_state,sample_act))
        value_next = Q_expected_new - log_prob3
        

        V_loss = self.value_criteria(value_expected, value_next.detach()).mean()

        Q1_loss = self.Q1_criteria(Q1_expected, next_q_hat.detach()).mean()
        Q2_loss = self.Q2_criteria(Q2_expected, next_q_hat.detach()).mean()

        policy_loss = (log_prob3 - Q_expected_new.detach()).mean()
        # print(policy_loss)
        # print(policy_loss.size())

        self.value_optim.zero_grad()
        V_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.value_nn.parameters(), 0.5)
        self.value_optim.step()

        self.Q1_optim.zero_grad()
        Q1_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.Q1_nn.parameters(), 0.5)
        self.Q1_optim.step()

        self.Q2_optim.zero_grad()
        Q2_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.Q2_nn.parameters(), 0.5)
        self.Q2_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 0.5)
        self.policy_optim.step()

        #if self.num_training_episodes % param.target_update_freq == 0:
        for targ_param, parameters in zip(self.TargetValue_nn.parameters(),self.value_nn.parameters()):
            targ_param.data.copy_(targ_param * (1 - 0.005) + parameters * 0.005)

def run(param):
    reward_vector = []
    total_t = 0
    eps_reward = 0
    env = NormalizedActions(gym.make(param.env_name))
    #env.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)
    agent = SAC(param)
    t=0
    eps_reward = 0
    initial_state = env.reset()
    state = initial_state
    done = False
    for i in range(param.total_steps):
        if i % 1000 == 0:
          plt.plot(range(len(reward_vector)),reward_vector)
          plt.show()
          clear_output = True
          print(f"total time steps: {i}")
          print(f"reward from previous episode: {eps_reward}")

        if agent.replay_buffer.num_transitions > param.batch_size: action = agent.choose_action(state)
        else: action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        eps_reward += reward

        agent.replay_buffer.update_buffer(state, action, reward, next_state, done)
        state = next_state

        if i % param.train_freq == 0 and agent.replay_buffer.num_transitions >= param.batch_size:
            agent.batch_training()
        if done:
            reward_vector.append(eps_reward)
            eps_reward = 0
            env.reset()
    return reward_vector

class parameters(object):
    def __init__(self):
        super(parameters,self).__init__()
        self.env_name = 'Humanoid-v4'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.state_dim = 376
        self.action_dim = 17
        self.hidden_layers = 256
        self.total_steps = 1000000
        self.train_freq = 1
        self.max_time_steps = 1500
        self.learn_rate = 0.0003
        self.gamma = 0.99
        self.buffer_capacity = 1000000
        self.batch_size = 256
        self.target_update_freq = 20

if __name__ == '__main__':
    param = parameters()
    reward_vector = run(param)
    # plt.plot(range(len(reward_vector)),reward_vector)
    # plt.show()
