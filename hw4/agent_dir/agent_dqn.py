import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from torch.optim import RMSprop, Adam
import random
import numpy as np
from collections import namedtuple
from itertools import count
from agent_dir.agent import Agent

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            del self.memory[0]
            self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(1568, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        
        self.args = args
        self.batch_size = args.batch_size
        self.replayMem = ReplayMemory(args.mem_cap)
        self.model = DQN(self.env.get_action_space().n)
        self.target_model = DQN(self.env.get_action_space().n)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.latest_reward = []
        self.steps_done = 0

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            
    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        for episode_n in range(1, self.args.num_episodes+1):
            
            episode_done = False
            episode_reward_sum = 0
            round_n = 1
            
            # Initialize the environment and state
            last_observation = self.init_game_setting()
            action = self.env.get_random_action()
            observation, _, _, _ = self.env.step(action)
            
            while not episode_done:
                # Select and perform an action
                state = observation - last_observation
                action = self.make_action(state)
                last_observation = observation
                observation, reward, episode_done, info = self.env.step(action)
                episode_reward_sum += reward

                # Observe new state
                next_state = observation - last_observation
                    
                # Store the transition in memory
                self.replayMem.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss = self.__optimize_model()
                
                if reward != 0:
                    round_n += 1
            
            print("Episode {} finished after {} rounds".format(episode_n, round_n))
            
            # save latest rewards
            self.latest_reward.append(episode_reward_sum)
            
            print("Total reward: {:.0f}".format(episode_reward_sum))
            print("Latest 30 episodes average reward: {}".format(sum(self.latest_reward[-30:])/30))
            print('---------------------------------------------------------------')
                
            # Update the target network
            if episode_n % self.args.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if episode_n % self.args.save_freq == 0:
                log = {
                    'episode': episode_n,
                    'state_dict': self.model.state_dict(),
                    'loss': loss,
                    'latest_reward': self.latest_reward
                }
                torch.save(log, 'checkpoints/dqn/checkpoint_episode{}.pth.tar'.format(episode_n))

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        observation = self.env.reset()
        return observation

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        EPS_START = 1.0
        EPS_END = 0.025
        EPS_DECAY = 200
        sample = np.random.rand()
        # perform epsilon greedy exploration
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            observation = np.transpose(observation, (2, 0, 1))
            observation = Variable(torch.FloatTensor(observation)).contiguous()
            observation = observation.view(1, *observation.size())
            return int(self.model(observation).max(1)[1].view(1, 1).data.numpy())
        else:
            return self.env.get_random_action()
        
    def __optimize_model(self):
        if len(self.replayMem) < self.batch_size: return
        transitions = self.replayMem.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        ## BUGGY ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]))
        state_batch = Variable(torch.cat(batch.state))
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss
