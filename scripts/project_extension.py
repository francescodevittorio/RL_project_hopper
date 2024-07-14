import torch
from torch import nn, optim
import numpy as np
import csv
import wandb
import gym
import os
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Impostare il seed
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

# Classe del generatore
class Generator(nn.Module):
    def __init__(self, latent_dim, param_dim, output_range):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, param_dim)
        )
        self.output_range = output_range
        self._initialize_weights()  # Inizializzare i pesi solo una volta qui

    def forward(self, x):
        x = self.network(x)
        x = torch.tanh(x)
        min_val, max_val = self.output_range
        x = min_val + (max_val - min_val) * (x + 1) / 2  # Riscalare da (-1, 1) a (0, 1) e poi a (min_val, max_val)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')  # Inizializzazione di He
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Classe del discriminatore
class Discriminator(nn.Module):
    def __init__(self, observation_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(observation_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output

# Callback per wandb
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self):
        if self.locals['dones'][0]:
            episode_rewards = self.locals['infos'][0]['episode']['r']
            episode_length = self.locals['infos'][0]['episode']['l']
            wandb.log({
                "episode_reward": episode_rewards,
                "episode_length": episode_length
            })
        return True

# Funzione per applicare i parametri all'ambiente
def apply_parameters_to_env(env, params):
    env.sim.model.body_mass[2:] = params

# Funzione per simulare l'ambiente
def simulate(env, model):
    observations = []
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        observations.append(obs)
        total_reward += reward
    
    observations = np.array(observations)
    return torch.tensor(observations, dtype=torch.float32), total_reward

# Funzione per fare padding delle traiettorie
def pad_trajectories(trajectories, observation_dim):
    max_length = max(len(traj) for traj in trajectories)
    padded_trajectories = torch.zeros((len(trajectories), max_length, observation_dim))
    
    for i, traj in enumerate(trajectories):
        length = len(traj)
        padded_trajectories[i, :length] = torch.tensor(traj, dtype=torch.float32)
    
    return padded_trajectories

# Parametri fissi
latent_dim = 10
param_dim = 3
observation_dim = 11
output_range = (0.5, 6)
num_epochs = 12  # Cambia
num_simopt_iterations = 5  # Cambia
real_batch_size = 16
simulated_batch_size = 64 
num_env = 4

# Inizializzazione degli ambienti
source_env = gym.make('CustomHopper-source-v0')
target_env = gym.make('CustomHopper-target-v0')

wandb.init(project="extension", name=f"prova_1.5")

# Caricamento del modello pre-addestrato
model = PPO.load("./logs_safe/safe_model.zip")

generator = Generator(latent_dim, param_dim, output_range)
discriminator = Discriminator(observation_dim)
loss_function = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=1e-3)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-2)

# Aprire i file CSV per scrivere i risultati
with open('generated_params_finals_prova_1_5.csv', mode='w', newline='') as params_file, \
     open('simulated_results_finals_prova_1_5.csv', mode='w', newline='') as simulated_file, \
     open('real_results_finals_prova_1_5.csv', mode='w', newline='') as real_file:

    params_writer = csv.writer(params_file)
    simulated_writer = csv.writer(simulated_file)
    real_writer = csv.writer(real_file)

    # Scrivere le intestazioni dei file CSV
    params_writer.writerow(["simopt_iteration", "epoch", "params"])
    simulated_writer.writerow(["simopt_iteration", "epoch", "mean_simulated_reward", "std_simulated_reward", "d_loss_real", "d_loss_fake", "d_loss", "g_loss"])
    real_writer.writerow(["simopt_iteration", "mean_real_reward", "std_real_reward"])

    all_real_trajectories = []  # Lista per accumulare tutte le traiettorie reali

    for simopt_iteration in range(num_simopt_iterations):
        real_trajectories = []
        real_rewards = []
        for _ in range(real_batch_size):
            traj, reward = simulate(target_env, model)
            real_trajectories.append(traj)
            real_rewards.append(reward)
        
        # Accumulare tutte le traiettorie reali
        all_real_trajectories.extend(real_trajectories)

        for epoch in range(num_epochs):
        # Definire il numero di timesteps in base all'epoca
            if epoch < 3:
                timesteps = 10000
            elif epoch < 6:
                timesteps = 30000
            elif epoch < 9:
                timesteps = 60000
            else:
                timesteps = 100000

            z = torch.randn(num_env, latent_dim)
            generated_params = generator(z)

            for params in generated_params:
                apply_parameters_to_env(source_env, params.detach().numpy())
                model.set_env(source_env)
                model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=WandbCallback())

            simulated_trajectories = []
            simulated_rewards = []
            for _ in range(simulated_batch_size):
                traj, reward = simulate(source_env, model)
                simulated_trajectories.append(traj)
                simulated_rewards.append(reward)

            all_real_trajectories_tensor = pad_trajectories(all_real_trajectories, observation_dim)
            real_labels = torch.ones(len(all_real_trajectories), 1)
            real_outputs = discriminator(all_real_trajectories_tensor)
            d_loss_real = loss_function(real_outputs, real_labels)

            simulated_trajectories_tensor = pad_trajectories(simulated_trajectories, observation_dim)
            fake_labels = torch.zeros(simulated_batch_size, 1)
            fake_outputs = discriminator(simulated_trajectories_tensor)
            d_loss_fake = loss_function(fake_outputs, fake_labels)

            optimizer_D.zero_grad()
            real_weight = simulated_batch_size / (len(all_real_trajectories) + simulated_batch_size)
            simulated_weight = len(all_real_trajectories) / (len(all_real_trajectories) + simulated_batch_size)
            d_loss = d_loss_real * real_weight + d_loss_fake * simulated_weight
            d_loss.backward()
            #torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  # Gradient clipping
            optimizer_D.step()

            optimizer_G.zero_grad()
            fake_outputs_for_generator = discriminator(simulated_trajectories_tensor)
            g_loss = loss_function(fake_outputs_for_generator, torch.ones(simulated_batch_size, 1))
            g_loss.backward()
            #torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)  # Gradient clipping
            optimizer_G.step()

            params_log = [param.detach().cpu().numpy().tolist() for param in generated_params]

            # Scrivere i parametri generati nel file CSV
            for param_set in params_log:
                params_writer.writerow([simopt_iteration + 1, epoch + 1, param_set])
                params_file.flush()  # Assicurarsi che i dati siano scritti immediatamente

            mean_simulated_reward = np.mean(simulated_rewards)
            std_simulated_reward = np.std(simulated_rewards)

            simulated_writer.writerow([simopt_iteration + 1, epoch + 1, mean_simulated_reward, std_simulated_reward, d_loss_real.item(), d_loss_fake.item(), d_loss.item(), g_loss.item()])
            simulated_file.flush()  # Assicurarsi che i dati siano scritti immediatamente

            # Loggare le loss e i reward simulati con lo scorrere delle epoche
            wandb.log({
                "epoch": epoch + 1,
                "d_loss_real": d_loss_real.item(),
                "d_loss_fake": d_loss_fake.item(),
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item(),
                "simulated_rewards_mean": mean_simulated_reward,
                "simulated_rewards_std": std_simulated_reward
            })

            # Salva il modello PPO dopo ogni epoca con il numero dell'epoca nel nome del file
            checkpoint_path = os.path.join("prova_1_6", f"sim_opt_model_simopt_{simopt_iteration}_epoch_{epoch}.zip")
            model.save(checkpoint_path)

        mean_real_reward = np.mean(real_rewards)
        std_real_reward = np.std(real_rewards)
        real_writer.writerow([simopt_iteration + 1, mean_real_reward, std_real_reward])
        real_file.flush()  # Assicurarsi che i dati siano scritti immediatamente

        # Loggare i reward reali con lo scorrere delle simopt_iterations
        wandb.log({
            "simopt_iteration": simopt_iteration + 1,
            "real_rewards_mean": mean_real_reward,
            "real_rewards_std": std_real_reward
        })

wandb.finish()
