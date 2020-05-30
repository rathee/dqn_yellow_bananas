import numpy as np
import random
from collections import deque, namedtuple
from Model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
import config
from replay_buffer import ReplayBuffer
from replay_buffer_prioritized import ReplayBufferPrioritized
from prioritized_memory import Memory
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    '''Interacts and learns with Environment'''

    def __init__(self, state_size, action_size, seed=42, filename = None):
        '''

        :param state_size(int): dimension for each state
        :param action_size(int): total actions
        :param seed(int):
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        #defining two networks as per DQN
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed)
        self.qnetwork_local.apply(self.weights_init)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = config.LR)

        if filename:
            weights = torch.load(filename)
            self.qnetwork_local.load_state_dict(weights)
            self.qnetwork_target.load_state_dict(weights)

        #Replay Memory
        self.memory = Memory(config.BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.initial_epsilon = 0.9
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay_rate = 0.99

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def step_old(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % config.UPDATE_EVERY

        if self.t_step == 0:
            if len(self.memory) > config.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, config.GAMMA)

    def step(self, state, action, reward, next_state, done):
        '''

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        '''

        '''
            1. change this to prioritize experience replay
            2. find delta between actual reward and the model value
            3. Pass this delta to replay buffer to prioritize
        '''
        #self.memory.add(state, action, reward, next_state, done)

        q_targets_next_all = self.qnetwork_target(Variable(torch.FloatTensor(next_state))).data
        q_targets_next = torch.max(q_targets_next_all)
        q_target = reward + (config.GAMMA * q_targets_next * (1 - done))

        q_expected_all = self.qnetwork_local(torch.FloatTensor(state)).data
        q_expected = q_expected_all.gather(0, torch.from_numpy(np.array([action])))[0]
        #q_expected = q_expected_all[0][action]
        loss = abs(q_expected - q_target)

        self.memory.add(loss, (state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % config.UPDATE_EVERY

        if self.t_step == 0:
            if self.memory.tree.n_entries > config.BATCH_SIZE:
                self.learn()

    def act(self, state, eps):
        '''
        Returns action for given state as per current policy
        :param state:
        :return:
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def get_sample(self):

        mini_batch, idxs, is_weights = self.memory.sample(config.BATCH_SIZE)
        mini_batch = np.array(mini_batch).transpose()

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4]

        # bool to binary
        dones = dones.astype(int)

        # Q function of current state
        states = torch.Tensor(states)
        states = Variable(states).float()

        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float()

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        return states, actions, rewards, next_states, dones, idxs, is_weights

    def update_memory(self, idxs, errors):

        for i in range(config.BATCH_SIZE):
            idx = idxs[i]
            error = errors[i]
            self.memory.update(idx, error)

    def learn(self):
        '''
        learn over a given sample

        :param experiences:
        :param gamma:
        :return:
        '''
        gamma = config.GAMMA
        states, actions, rewards, next_states, dones, idxs, is_weights = self.get_sample()

        q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        #one hot encoding of actions
        a = torch.LongTensor(actions).view(-1, 1)
        one_hot_action = torch.FloatTensor(config.BATCH_SIZE, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)

        q_expected_all = self.qnetwork_local(states)
        q_expected = torch.sum(q_expected_all.mul(Variable(one_hot_action)), dim=1)
        errors = torch.abs(q_expected - q_targets).data.numpy()
        loss = F.mse_loss(q_expected, q_targets)

        self.update_memory(idxs, errors[0])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, config.TAU)


    def soft_update(self, local_model, target_model, tau):
        '''
        update target model parameters
        theta_target = tau*local + (1 - tau) * theta_target

        :param local_model:
        :param target_model:
        :param tau:
        :return:
        '''

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

