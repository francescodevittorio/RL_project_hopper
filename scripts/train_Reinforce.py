import argparse
import time
import os
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

from env.custom_hopper import *
from agent_Reinforce import Agent, Policy

### HO CAMBIATO n-episodes da 100000 a 10000 e print-every da 20000 a 2000
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=500000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    #parser.add_argument('--load-model', action='store_true', help='Load model from file')
    #parser.add_argument('--model-path', default='model.mdlTask2', type=str, help='Path to the model file')
    return parser.parse_args()

args = parse_args()

def set_seed(seed, env):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def main():
    start_time = time.time()

    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    # Set seed for reproducibility
    set_seed(args.seed, env)

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    """
        Training
    """
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    #
    # TASK 2: interleave data collection to policy updates
    #
    # Carica i parametri del modello se l'opzione --load-model è abilitata
    #if args.load_model:
    #    policy.load_state_dict(torch.load(args.model_path))
    episode_rewards = []
    episode_lengths = []
    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        episode_lenght = 0
        state = env.reset()  # Reset the environment and observe the initial state
        #agent.states = []
        #agent.next_states = []
        agent.action_log_probs = []
        agent.rewards = []
        #agent.done = []
        while not done:  
            action, log_action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, log_action_probabilities, reward, done)
            episode_lenght += 1
            train_reward += reward
        
        agent.update_policy()
        episode_rewards.append(train_reward)
        episode_lengths.append(episode_lenght)
        if (episode+1) % args.print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', train_reward)

    torch.save(agent.policy.state_dict(), "model.mdlTask2")

    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Plot and save reward per episode
    plt.figure(figsize=(12, 6))
    plt.plot(range(args.n_episodes), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.savefig('results/reward_per_episode.png')
    plt.close()

    # Plot and save episode lengths
    plt.figure(figsize=(12, 6))
    plt.plot(range(args.n_episodes), episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.title('Episode length per Episode')
    plt.savefig('results/episode_length_per_episode.png')
    plt.close()

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total execution time: {total_time:.2f} seconds')

if __name__ == '__main__':
    main()
