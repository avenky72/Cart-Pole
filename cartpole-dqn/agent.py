from model import Model
from replay_buffer import ReplayBuffer
import torch.nn as nn
import torch.optim as optim
import copy
import random
import torch


class agent:
    def __init__(self, gamma, epsilon, epsilon_decay, batch_size, target_update):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.policy_network = Model()
        self.buffer = ReplayBuffer(10000)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.target_network = copy.deepcopy(self.policy_network)

        
    # Generate a random number. If it's less than epsilon take a random value, if not then argmax possible actions
    def choose_action(self, state):
        choice = random.random()
        if choice < self.epsilon:
            return random.randint(0, 1)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.policy_network(state)
            return q_values.argmax().item()


    # Loss computation and learning function for the policy network
    def learn(self):
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return
        prediction_list = []
        target_list = []
        
        for experience in batch:
            state, action, reward, next_state, done = experience
            
            state = torch.tensor(state, dtype=torch.float32)
            q_val = self.policy_network(state)
            prediction = q_val[action]
            prediction_list.append(prediction)
            
            next_state = torch.tensor(next_state, dtype=torch.float32)
            future_val = self.target_network(next_state)
            next_q_value = future_val.max()
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * next_q_value
            target_list.append(target)
            
        # Convert list to tensors for the MSELoss
        predictions = torch.stack(prediction_list)
        targets = torch.tensor(target_list)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon to decrease exploration rate over time
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)
                

    # Update target network every 1000 or so steps (called in training)
    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
            
        
    