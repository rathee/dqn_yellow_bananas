from collections import deque, namedtuple
import random
import torch
import numpy as np
import heapq
import config

class ReplayBuffer:
    '''Fixed size buffer to store experience tuples'''

    def __init__(self,action_size, buffer_size, batch_size, seed, device):
        '''
        Initialize a replay buffer object

        :param action_size:
        :param buffer_size:
        :param batch_size:
        :param seed:
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.device = device

    def add(self, state, action, reward, next_state, done, loss = 0):
        '''
        adds a new experience to memory

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        '''

        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
         '''Randomly sample a batch of experiences from memory'''

         experiences = random.sample(self.memory, k=self.batch_size)

         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

         return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)