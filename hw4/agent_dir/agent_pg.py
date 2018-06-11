from agent_dir.agent import Agent
from scipy.misc import imresize, imsave
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import RMSprop


def prepro(I):
    # method 1
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py

    Input:
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array
        Grayscale image, shape: (80, 80, 1)

    """
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32), axis=2)
    """

    # method 2
    """ 
        prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector 
    """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 255  # everything else (paddles, ball) just set to 1

    return I.astype(np.float).ravel()


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        self.__build_model()
        if args.test_pg:
            # you can load your model here
            print('loading trained model')


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        #self.prev_observation = self.env.reset()
        #self.prev_observation = prepro(self.prev_observation)

    def __build_model(self):

        class PG(nn.Module):
            def __init__(self):
                super(PG, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(80*80, 200),
                    nn.ReLU(),
                    nn.Linear(200, 1),
                    nn.Sigmoid()
                )
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal(m.weight)

            def forward(self, observation):
                action = self.fc(observation)
                return action

        self.model = PG()
        self.optimizer = RMSprop(self.model.parameters(), lr=0.0001)

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        prev_observation = None
        curr_observation = self.env.reset()
        xs, ys, rs = [], [], []
        running_reward = None
        reward_sum = 0
        episode_number = 0

        self.model.train()

        while True:
            curr_observation = prepro(curr_observation)
            residual = curr_observation - prev_observation if prev_observation is not None \
                                                                     else np.zeros(6400)
            prev_observation = curr_observation

            # predict policy network
            action, _, y = self.make_action(residual)

            xs.append(residual)
            ys.append(y)

            curr_observation, reward, done, info = self.env.step(action)
            reward_sum += reward
            rs.append(reward)

            if done:
                episode_number += 1
                print("[Episode {}]".format(episode_number), end=' ')
                episode_residuals = np.vstack(xs)
                episode_rewards = np.vstack(rs)
                episode_y = np.vstack(ys)
                xs, rs, ys = [], [], []

                discounted_epr = self.discount_rewards(episode_rewards)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                episode_residuals = Variable(torch.FloatTensor(episode_residuals))
                discounted_epr = Variable(torch.FloatTensor(discounted_epr))
                episode_y = Variable(torch.FloatTensor(episode_y))

                self.optimizer.zero_grad()

                prob = self.model(episode_residuals)
                loss = F.binary_cross_entropy(prob, episode_y, discounted_epr)
                loss.backward()
                self.optimizer.step()

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('Loss: {:.6f}. Episode total reward {}. Running mean: {:.6f}.'.format(loss,
                                                                                            reward_sum,
                                                                                            running_reward))

                reward_sum = 0
                curr_observation = self.env.reset()
                prev_observation = None

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = Variable(torch.FloatTensor(observation))
        probability = self.model(observation.unsqueeze(0)).data.cpu().numpy()[0]
        action = 2 if np.random.uniform() < probability else 3
        y = 1 if action == 2 else 0

        return action, probability, y

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        gamma = 0.99
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r


if __name__ == '__main__':
    import argparse
    from environment import Environment
    parser = argparse.ArgumentParser(description="MLDS 2018 HW4")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    args = parser.parse_args()
    env_name = args.env_name or 'Pong-v0'
    env = Environment(env_name, args)
    from agent_dir.agent_pg import Agent_PG
    agent = Agent_PG(env, args)
    agent.train()
