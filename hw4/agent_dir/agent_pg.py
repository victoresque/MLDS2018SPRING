from agent_dir.agent import Agent
from scipy.misc import imresize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import RMSprop, Adam


def prepro(observation):
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
    observation = observation[35:195]  # crop
    observation = observation[::2, ::2, 0]  # downsample by factor of 2
    observation[observation == 144] = 0  # erase background (background type 1)
    observation[observation == 109] = 0  # erase background (background type 2)
    observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1

    return observation.astype(np.float).ravel()


class PG(nn.Module):
    def __init__(self, hidden_size):
        super(PG, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(80 * 80, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)

    def forward(self, observation):
        action = self.fc(observation)
        return action


UP_ACTION = 2
DOWN_ACTION = 3
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG, self).__init__(env)
        self.args = args
        self.latest_reward = []
        self.__build_model()
        if args.test_pg:
            # you can load your model here
            print('loading trained model')

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass

    def __build_model(self):
        self.model = PG(self.args.hidden_size)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        batch_state_action_reward = []
        smoothed_reward = None
        episode_n = 1

        while True:
            print('---------------------------------------------------------------')
            print("Starting episode {}".format(episode_n))

            episode_done = False
            episode_reward_sum = 0

            round_n = 1

            last_observation = self.env.reset()
            last_observation = prepro(last_observation)
            action = self.env.action_space.sample()
            observation, _, _, _ = self.env.step(action)
            observation = prepro(observation)

            while not episode_done:
                observation_delta = observation - last_observation
                last_observation = observation

                action, up_probability = self.make_action(observation_delta)

                observation, reward, episode_done, info = self.env.step(action)
                observation = prepro(observation)
                episode_reward_sum += reward

                tup = (observation_delta, action_dict[action], reward)
                batch_state_action_reward.append(tup)

                if reward != 0:
                    round_n += 1

            print("Episode {} finished after {} rounds".format(episode_n, round_n))

            # save latest rewards
            self.latest_reward.append(episode_reward_sum)

            if smoothed_reward is None:
                smoothed_reward = episode_reward_sum
            else:
                smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum * 0.01
            print("Total reward: {:.0f}; Smoothed average reward {:.4f}".format(episode_reward_sum, smoothed_reward))
            print("Latest 30 episodes average reward: {}".format(sum(self.latest_reward[-30:])/30))
            print('---------------------------------------------------------------')
            states, actions, rewards = zip(*batch_state_action_reward)
            rewards = self.discount_rewards(rewards)
            rewards -= np.mean(rewards)
            rewards /= np.std(rewards)

            self.model.train()

            states = Variable(torch.FloatTensor(states))
            actions = Variable(torch.FloatTensor(actions))
            rewards = Variable(torch.FloatTensor(rewards))

            self.optimizer.zero_grad()
            pred = self.model(states)
            loss = F.binary_cross_entropy(pred.squeeze(), actions, weight=rewards)
            loss.backward()
            self.optimizer.step()

            batch_state_action_reward = []

            episode_n += 1
            if episode_n % self.args.save_freq == 0:
                log = {
                    'episode': episode_n,
                    'state_dict': self.model.state_dict(),
                    'loss': loss,
                    'latest_reward': self.latest_reward
                }
                torch.save(log, 'checkpoints/checkpoint_episode{}.pth.tar'.format(episode_n))

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
        self.model.eval()
        observation = Variable(torch.FloatTensor(observation))
        up_probability = self.model(observation).data.cpu().numpy()[0]
        action = UP_ACTION if np.random.uniform() < up_probability else DOWN_ACTION

        return action, up_probability

    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, len(r))):
            if r[t] != 0:
                running_add = 0
            running_add = running_add * self.args.gamma + r[t]
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
    parser.add_argument('--save-freq', type=int, default=1,
                        help='saving frequency')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate for training')
    parser.add_argument('--hidden-size', type=int, default=200,
                        help='hidden size for the training model')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for reward in training')
    args = parser.parse_args()
    env_name = args.env_name or 'Pong-v0'
    env = Environment(env_name, args)
    from agent_dir.agent_pg import Agent_PG
    agent = Agent_PG(env, args)
    agent.train()
