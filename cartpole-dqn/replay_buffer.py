"""
The replay buffer that stores experiences, from which the agent will randomly sample batches from/
This exists in order to make the agent less reliant on the most recent experience
Stores the experiences in a queue of set length
"""
    
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        
    def store(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        return 
    
    def sample(self, batch_size):
        if len(self.buffer) >= batch_size:
            return random.sample(self.buffer, k=batch_size)
        else:
            return None