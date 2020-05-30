from collections import deque, namedtuple
import random
import torch
import numpy as np
import heapq
import config

class ReplayBufferPrioritized:
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
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.experience = namedtuple("Experience", field_names=['loss','state', 'action', 'reward', 'next_state', 'done'])
        self.device = device
        self.memory = []
        self.sample_factor = config.SAMPLE_FACTOR

    def add(self, state, action, reward, next_state, done, loss):
        '''
            adds the tuple into the prioritized heap
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :param loss:
        :return:
        '''

        loss = abs(loss)

        if len(self.memory) == self.buffer_size:
            self.memory.pop()

        #creating max heap and hence multiplying loss by -1
        experience = self.experience(-1*loss, state, action, reward, next_state, done)
        try:
            heapq.heappush(self.memory, experience)
        except Exception as e:
            print(e)

    def sample(self):
        '''Picks a sample according to x% from highest loss and 100-x % as random'''
        top_elements = int(self.sample_factor * self.batch_size)
        random_elements = self.batch_size - top_elements

        experiences_top = heapq.nsmallest(top_elements, self.memory)
        experiences_random = random.sample(self.memory, k=random_elements)
        experiences = experiences_top + experiences_random

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        return len(self.memory)