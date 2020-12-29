import os
from collections import deque
from itertools import chain
import torch
import numpy as np
import click
import matplotlib.pyplot as plt
import yaml
from unityagents import UnityEnvironment
from agent import Agent, LearningStrategy, TargetNetworkUpdateStrategy
from utils import get_env, get_env_properties, get_next_results_directory, \
    load_results, solution_directory, results_directory


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

    :param env: UnityEnvironment with which the agent will interact.
    :type env: UnityEnvironment
    :param agent: Specific instance of the agent that will interact with the environment.
    :type agent: Agent
    :param n_episodes: Maximum number of training episodes, defaults to 10.
    :type n_episodes: int, optional
    :param eps_start: Starting value of epsilon, for epsilon-greedy action selection, defaults to 1.0
    :type eps_start: float, optional
    :param eps_end: Minimum value of epsilon, defaults to 0.01
    :type eps_end: float, optional
    :param eps_decay: Multiplicative factor (per episode) for decreasing epsilon, defaults to 0.995
    :type eps_decay: float, optional
    :param checkpoint_directory: [description], defaults to "./checkpoints"
    :type checkpoint_directory: str, optional
    :param keep_training: [description], defaults to True
    :type keep_training: bool, optional
    :return: [description]
    :rtype: [type]
    """

    assert type(env) is UnityEnvironment, "Misspecfied environment, expected {}".format(UnityEnvironment)

    if not os.path.exists(checkpoint_directory):
        print(f"Creating directory at {checkpoint_directory}")
        os.makedirs(checkpoint_directory)

    # Obtain the brain that will control the agent.
    brain_name = env.brain_names[0]

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


def run_agent(env: UnityEnvironment, trained_agent: Agent):
    """Run a trained agent in an environment.
    
    :param env: Initialised environment object.
    :type env: UnityEnvironment
    :param trained_agent: Initialised agent object.
    :type trained_agent: Agent
    """

    brain_name = env.brain_names[0]

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
        

@click.group()
def cli():
    pass


@cli.command()
@click.option("--checkpoint-path", required=False, type = str, \
    help="Path to a checkpoints file with which to obtain learned DQN weights.")
def run(checkpoint_path):
    """Program that initialises an agent using saved DQN weights and allows the
    agent to explore an environment.
    """
    env = get_env(False)

    action_size, state_size = get_env_properties(env)

    # Create an agent to receive the values of the DQNs. Only the state size
    # and action size are important when running a trained agent.
    agent = Agent(state_size=state_size, action_size=action_size, seed=0, 
    learning_strategy=LearningStrategy.DDQN, 
    target_network_update_strategy=TargetNetworkUpdateStrategy.SOFT)

    if checkpoint_path is None:
        checkpoint = os.path.join(solution_directory, f'checkpoints/solved_checkpoint.pth')

    agent.qnetwork_local.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))

    run_agent(env, agent)


@cli.command()
@click.option("--learning-strategy", required=False, default = 'DQN', \
    type = click.Choice(['DQN', 'DDQN']), help="Train the agent using DQN or DDQN.")
@click.option("--update-type", required=False, default = 'soft', \
    type = click.Choice(['soft', 'hard']), \
        help="Use soft updates or hard updates for 'fixed-Q' TD targets.")
@click.option("--n-episodes", required=False, default=4000, \
    help="Number of episodes after which training will terminate.")
@click.option("--headless", required=False, is_flag=True, \
    help="Train the agent using the headless environment.")
@click.option("--keep-training", required=False, is_flag=True, \
    help="Continue training the agent up to n-episodes after the solved condition is met.")
@click.option("--checkpoint", required=False, type=str, default=None, \
    help="""
    Path to a previously trained Agent's PyTorch checkpoint, if specified the 
    Agents network will be initialised using the weights therein.
    """)
def train(learning_strategy, update_type, n_episodes, headless, checkpoint, keep_training):
    """Program to train an agent using a training strategy specified by LEARNING-STRATEGY
    and UPDATE-TYPE. The agent is trained up to N-EPISODES if KEEP-TRAINING is set,
    otherwise the training can terminate earlier if the solved condition is met.
    """
    env = get_env(headless)

    action_size, state_size = get_env_properties(env)

    agent = Agent(
        state_size=state_size, 
        action_size=action_size, 
        seed=0,
        learning_strategy=LearningStrategy[learning_strategy],
        target_network_update_strategy=TargetNetworkUpdateStrategy[update_type.upper()])

    if checkpoint is not None:
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))

    checkpoint_directory = f'./checkpoints/{learning_strategy}'
    scores = unity_dqn(env, agent, n_episodes=n_episodes, checkpoint_directory=checkpoint_directory, keep_training=keep_training)
    np.savetxt(f'{checkpoint_directory}/scores.txt', scores)

    params = {
        'learning_strategy': learning_strategy, 
        'update_type': update_type
        }

    next_results_directory = os.path.join(results_directory, get_next_results_directory())
    if not os.path.exists(next_results_directory):
        os.mkdir(next_results_directory)
    
    np.savetxt(f'{next_results_directory}/scores.txt', scores)
    with open(f'{next_results_directory}/parameters.yml', 'w') as f:
        yaml.dump(params, f)


@cli.command()
def plot_results():
    """Plot results from scores stored in the results directory.
    """
    results_directories = os.listdir(results_directory)
    n_directories = len(results_directories)
    scale = 3.0

    fig, axes = plt.subplots(ncols=1, nrows=(n_directories+1), figsize=(1.5*scale, scale*(n_directories+1)))
    fig2, axes2 = plt.subplots(figsize=(4.5, 3))
    
    to_compare = []
    for i, directory in enumerate(results_directories):
        scores, ma, x, conf = load_results(directory)
        ax = axes[i]
        ax.plot(scores)
        series_label = f"{conf['learning_strategy']}, {conf['update_type']}"
        ax.plot(x, ma, label=series_label)
        to_compare.append([x, ma, series_label])
        ax.axhline(y=13, ls='dashed', c='k')

    for ax in [axes[-1], axes2]:
        [ax.plot(x[0], x[1], label=x[2]) for x in to_compare]

    for ax in chain(axes, [axes2]):
        ax.axhline(y=13, ls='dashed', c='k', label='solved')
        ax.legend()
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Score")


    plt.figure(fig.number)
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(solution_directory, 'resources', 'comparison_full.png'), dpi=400)
    plt.figure(fig2.number)
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(solution_directory, 'resources', 'comparison_summary.png'), dpi=400)
    plt.show()


if __name__ == "__main__":
    cli()


