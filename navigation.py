from agent import Agent, LearningStrategy
from collections import deque
import os
import torch
import numpy as np
import gym
import click
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas
from itertools import chain
from enum import Enum


solution_directory = os.path.abspath(os.path.dirname(__file__))
environments_directory = os.path.join(solution_directory, 'unity_environments')


class BananaEnv(Enum):
    STANDARD = 'Banana_Linux/Banana.x86_64'
    HEADLESS = 'Banana_Linux_NoVis/Banana.x86_64'
    VISUAL = 'VisualBanana_Linux/Banana.x86_64'


def get_environment_executable(environment: BananaEnv):
    assert type(environment) is BananaEnv
    print(environments_directory, environment)
    return os.path.join(environments_directory, environment.value)


def unity_dqn(
    env: UnityEnvironment,
    agent: Agent,
    n_episodes=10,
    eps_start=1.0, 
    eps_end=0.01, 
    eps_decay=0.995,
    checkpoint_directory = "./checkpoints",
    keep_training: bool = True
    ):

    """Deep Q-Learning.
    
    Params
    ======
        env (UnityEnvironment): UnityEnvironment with which the agent will interact
        agent (Agent): Specific implmentation of the agent that will interact with the environment
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    assert type(env) is UnityEnvironment, "Misspecfied environment, expected {}".format(UnityEnvironment)

    if not os.path.exists(checkpoint_directory):
        print(f"Creating directory at {checkpoint_directory}")
        os.makedirs(checkpoint_directory)

    # Obtain the brain that will control the agent.
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    solution_saved = False
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        while True:
            action = agent.act(state, eps)                 # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            if not solution_saved:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                output_path = os.path.join(checkpoint_directory, 'solved_checkpoint.pth')
                torch.save(agent.qnetwork_local.state_dict(), output_path)
            if not keep_training:
                break
        
    output_path = os.path.join(checkpoint_directory, f'checkpoint_{i_episode}.pth')
    torch.save(agent.qnetwork_local.state_dict(), output_path)

    return scores


def run_agent(env, trained_agent):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = trained_agent.act(state, 0.0)         # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        

def get_env(headless):
    env = BananaEnv.HEADLESS if headless else BananaEnv.STANDARD
    env_location = get_environment_executable(env)
    print(f"Getting environment from {env_location}")
    env = UnityEnvironment(file_name=env_location)
    return env


def get_env_properties(env: UnityEnvironment):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('State size:', state_size)
    return action_size, state_size


@click.group()
def cli():
    pass


@cli.command()
@click.option("--learning-strategy", required=False, default = 'DQN', type = click.Choice(['DQN', 'DDQN']))
def run(learning_strategy):
    env = get_env(False)

    action_size, state_size = get_env_properties(env)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0, learning_strategy=LearningStrategy[learning_strategy])

    checkpoint = os.path.join(solution_directory, f'./checkpoints/{learning_strategy}/solved_checkpoint.pth')

    agent.qnetwork_local.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))

    run_agent(env, agent)


@cli.command()
@click.option("--learning-strategy", required=False, default = 'DQN', type = click.Choice(['DQN', 'DDQN']))
@click.option("--n-episodes", required=False, default=4000)
@click.option("--headless", required=False, is_flag=True)
@click.option("--keep-training", required=False, is_flag=True)
@click.option("--checkpoint", required=False, type=str, default=None)
def train(learning_strategy, n_episodes, headless, checkpoint, keep_training):
    env = get_env(headless)

    action_size, state_size = get_env_properties(env)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0, learning_strategy=LearningStrategy[learning_strategy])

    if checkpoint is not None:
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))

    checkpoint_directory = f'./checkpoints/{learning_strategy}'
    scores = unity_dqn(env, agent, n_episodes=n_episodes, checkpoint_directory=checkpoint_directory, keep_training=keep_training)
    np.savetxt(f'{checkpoint_directory}/scores.txt', scores)


def load_results(learning_strategy: str, n_points: int):
    checkpoint_directory = f'checkpoints/{learning_strategy}'
    scores_file = os.path.join(solution_directory, checkpoint_directory, 'scores.txt')
    scores = pandas.Series(np.loadtxt(scores_file)[:n_points])
    rolling_window = 100
    ma = scores.rolling(rolling_window, center=True).mean()
    x = np.arange(len(ma))
    return scores, ma, x


@cli.command()
@click.option("--learning-strategy", required=False, default = 'DQN', type = click.Choice(['DQN', 'DDQN']))
def plot_results(learning_strategy):
    conf_path = os.path.join(solution_directory, 'conf/plot_results.yml')
    with open(conf_path, 'r') as f:
        conf =  yaml.load(f, Loader=yaml.BaseLoader)
        n_points = int(conf['n_points'])
    
    checkpoint_directory = f'checkpoints/{learning_strategy}'
    scores_file = os.path.join(solution_directory, checkpoint_directory, 'scores.txt')
    scores = pandas.Series(np.loadtxt(scores_file)[:n_points])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scores, alpha=0.5)

    rolling_window = 100
    ma = scores.rolling(rolling_window, center=True).mean()
    x = np.arange(len(ma))
    ax.plot(x, ma, label='100-episode average')
    print(min(x), max(x))
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Score")

    plt.axhline(y=13, ls='dashed', color='k', label='solved')
    yticks = range(0, 25, 2)
    ax.set_yticks(yticks)

    scores_plot = os.path.join(solution_directory, f'resources/{learning_strategy}_scores.png')
    plt.legend()

    plt.tight_layout()
    plt.savefig(scores_plot, dpi=400)
    plt.show()


if __name__ == "__main__":
    cli()


