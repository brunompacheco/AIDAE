import gym
import numpy as np
import matplotlib.pyplot as plt


class hill_climbing_agent():
    def __init__(self, env):
        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.parameters = np.zeros(self.n_observations)

    def get_action(self, observation, parameters=None):
        if parameters is None:
            action = 0 if np.matmul(self.parameters, observation) < 0 else 1
        else:
            action = 0 if np.matmul(parameters, observation) < 0 else 1
        return action

    def run_episode(self, parameters):
        observation = self.env.reset()
        totalreward = 0
        counter = 0
        for _ in range(200):
            # env.render()
            action = self.get_action(observation=observation, parameters=parameters)
            observation, reward, done, info = env.step(action)
            totalreward += reward
            counter += 1
            if done:
                break
        return totalreward

    def train(self, train_steps, noise_scaling=0.1, episodes_per_update=10):

        # train process is missing logic for parameters update. We need to fix it to make our agent to learn!
        # Implementation tip:
        #       1) implement parameters update in train function, following logic new_parameters = parameters + random_noise * noise_scaling
        #       2) check if new parameters are better than the old one
        #       3) if step 2 is True, take this parameters as the parameter set used In the agent
        #       4) repeat

        bestreward = -np.inf
        counter = 0

        plot_data = {
            'bestreward': list(),
            'reward': list()
        }

        new_parameters = self.parameters
        rng = np.random.default_rng()
        for _ in range(train_steps):
            counter += 1
            reward = 0
            for _ in range(episodes_per_update):
                run = self.run_episode(parameters=new_parameters)
                reward += run
            average_reward_per_episode = reward / episodes_per_update

            if average_reward_per_episode > bestreward:
                self.parameters = new_parameters
                bestreward = average_reward_per_episode

            new_parameters = rng.normal(loc=self.parameters, scale=noise_scaling)

            plot_data['bestreward'].append(bestreward)
            plot_data['reward'].append(average_reward_per_episode)

            noise_scaling = noise_scaling / 1.025
            
            if bestreward == 200:
                break

        return bestreward, plot_data


if __name__=='__main__':
    env = gym.make('CartPole-v0')
    agent = hill_climbing_agent(env=env)
    best_reward, plot_data = agent.train(train_steps=1000)
    print(f'final average reward over 5 episodes - {best_reward}')
    print('achieved after {} training steps'.format(len(plot_data['reward'])))

    plt.plot(plot_data['reward'], label='reward')
    plt.plot(plot_data['bestreward'], label='best reward')
    plt.show()

    # visualize agent's behavior
    totalreward = 0
    observation = env.reset()
    for _ in range(200):
        env.render()
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done == True:
            print(totalreward)
            break
