import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb

def evaluate_trained_model():
    wandb.init(project='UDR_test_source', name='UDR_30%_source', monitor_gym=True)
    
    env = gym.make('CustomHopper-source-v0')
    env = Monitor(env)

    model = PPO.load("./logs_UDR_30%/best_model")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    
    wandb.log({
        'mean_reward': mean_reward,
        'std_reward': std_reward
    })

    print(f"Mean reward: {mean_reward} ± {std_reward}")

if __name__ == '__main__':
    wandb.login()
    evaluate_trained_model()
