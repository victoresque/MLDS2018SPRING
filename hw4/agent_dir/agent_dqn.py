from agent_dir.agent import Agent
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import os
import sys


class DQN(nn.Module):

    def __init__(self, in_channels=4, num_actions=3, duel_net=False):
        super(DQN, self).__init__()
        self.duel_net = duel_net

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7*7*64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        if self.duel_net:
            print("Using Dueling Network......")
            self.fc_value = nn.Linear(512, 1)
            self.fc_advantage = nn.Linear(512, num_actions)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x.permute(0, 3, 1, 2)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        if self.duel_net:
            value = self.fc_value(x)
            advantange = self.fc_advantage(x)
            q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
            return q
        else:
            return self.fc5(x)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        ##################
        # YOUR CODE HERE #
        ##################

        super(Agent_DQN, self).__init__(env)

        self.args = args
        self.batch_size = 32
        self.gamma = 0.99
        self.episode = 1000000
        self.eps_min = 0.025
        self.eps_max = 1.0
        self.eps_step = 1000000
        self.memory = deque(maxlen=10000)
        self.target_Q = DQN(duel_net=args.duel).cuda()
        self.current_Q = DQN(duel_net=args.duel).cuda()
        self.target_Q.load_state_dict(self.current_Q.state_dict())
        self.optimizer = torch.optim.RMSprop(self.current_Q.parameters(), lr=0.00015)

        self.reward_list = []
        # print("============ Breakout ============")
        # print(self.env.env.unwrapped.get_action_meanings())
        # print("==================================")

        if args.duel:
            if not os.path.exists('checkpoints/dqn_duel'):
                os.makedirs('checkpoints/dqn_duel')
        else:
            if not os.path.exists('checkpoints/dqn'):
                os.makedirs('checkpoints/dqn')

        if args.test_dqn:
            # you can load your model here
            # print('loading trained model')
            checkpoint = torch.load('4-2.pth.tar', map_location=lambda storage, loc: storage)
            self.current_Q.load_state_dict(checkpoint['state_dict'])

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def epsilon(self, step):
        if step > self.eps_step:
            return 0
        else:
            return self.eps_min + (self.eps_max - self.eps_min) * ((self.eps_step - step) / self.eps_step)

    def update_param(self):
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        batch_state, batch_next, batch_action, batch_reward, batch_done = zip(*batch)

        batch_state = Variable(torch.stack(batch_state)).cuda().squeeze()
        batch_next = Variable(torch.stack(batch_next)).cuda().squeeze()
        batch_action = Variable(torch.stack(batch_action)).cuda()
        batch_reward = Variable(torch.stack(batch_reward)).cuda()
        batch_done = Variable(torch.stack(batch_done)).cuda()

        current_q = self.current_Q(batch_state).gather(1, batch_action)
        next_q = batch_reward + (1 - batch_done) * self.gamma * self.target_Q(batch_next).detach().max(-1)[0].unsqueeze(-1)

        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q, next_q)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        step = 0
        loss = []

        for episode in range(1, self.episode + 1):
            state = self.env.reset()
            state = state.astype(np.float64)
            done = False
            reward_sum = 0

            while not done:
                epsilon = self.epsilon(step)
                action = random.randint(0, 2) if random.random() < epsilon else self.make_action(state, False)
                next_state, reward, done, _ = self.env.step(action + 1)
                next_state = next_state.astype(np.float64)
                reward_sum += reward
                step += 1

                self.memory.append((
                    torch.FloatTensor([state]),
                    torch.FloatTensor([next_state]),
                    torch.LongTensor([action]),
                    torch.FloatTensor([reward]),
                    torch.FloatTensor([done])
                ))

                state = next_state

                if step % 4 == 0:
                    loss.append(self.update_param())
                if step % 1000 == 0:
                    self.target_Q.load_state_dict(self.current_Q.state_dict())

            print("Episode: {} | Step: {} | Reward: {}".format(episode, step, reward_sum), end='\r')
            sys.stdout.write('\033[K')

            self.reward_list.append(reward_sum)

            if episode % 100 == 0:
                print("---------------------------------------------")
                if step < self.eps_step:
                    print("** Exploring phase **")
                print("Episode:", episode)
                print("Latest 30 episode average reward: {:.4f}".format(sum(self.reward_list[-30:])/30))
                if self.epsilon(step):
                    print("Epsilon max:", self.eps_max, "min:", self.eps_min)
                    print("Current epsilon: {:.4f}".format(self.epsilon(step)))

            if episode % self.args.save_freq == 0:
                log = {
                    'episode': episode,
                    'state_dict': self.current_Q.state_dict(),
                    'loss': loss,
                    'latest_reward': self.reward_list
                }
                if self.args.duel:
                    torch.save(log, 'checkpoints/dqn_duel/checkpoint_episode{}.pth.tar'.format(episode))
                else:
                    torch.save(log, 'checkpoints/dqn/checkpoint_episode{}.pth.tar'.format(episode))

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
        ##################
        # YOUR CODE HERE #
        ##################

        # return 0-2 but only 1-3 valid
        action = self.current_Q(Variable(torch.FloatTensor(observation).unsqueeze(0)).cuda()).max(-1)[1].data[0]
        return action + 1 if test else action
