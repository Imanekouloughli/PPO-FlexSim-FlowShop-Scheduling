# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:24:23 2026

@author: koulo
"""

#from flexsim_env import FlexSimEnv
from flexsim_envFlow import FlexSimEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

# >>> AJOUT
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    print("Initializing FlexSim environment...")

    # >>> AJOUT: dossier logs (pour récupérer rollout/ep_rew_mean)
    log_dir = r"C:/Users/koulo/OneDrive/Bureau/Articles/article 9/logs_ppo2_etat_enrichi"
    os.makedirs(log_dir, exist_ok=True)

    # === Création de l'environnement FlexSim ===
    env = FlexSimEnv(
        flexsimPath="C:/Program Files/FlexSim 2026/program/flexsim.exe",
        modelPath="C:/Users/koulo/OneDrive/Bureau/Articles/article 9/Flow shop/flow shop nouvelle matrice 8 machines 10 produits.fsm",
        port=5010,              # IMPORTANT : éviter conflit de port
        verbose=True,
        visible=True
    )

    check_env(env)  # Vérification Gym

    # >>> AJOUT: Monitor pour que SB3 calcule et loggue ep_rew_mean
    # Ça crée aussi un monitor.csv (récompenses par épisode) dans log_dir
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    # === Entraînement PPO ===
    model = PPO("MlpPolicy", env, verbose=1, n_steps=1024)

    # >>> AJOUT: logger CSV (progress.csv contiendra rollout/ep_rew_mean)
    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)

    print("Training model...")

    model.learn(total_timesteps=200000)

    # === Sauvegarde du modèle ===
    print("Saving model...")
    model.save("C:/Users/koulo/OneDrive/Bureau/Articles/article 9/MyTrainedModel")

    # === Évaluation du modèle entraîné ===
    n_episodes = 30
    episode_rewards = []

    print("Evaluating trained model...")
    for ep in range(n_episodes):
        observation, info = env.reset()
        done = False
        cumulative_reward = 0.0

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, trunc, info = env.step(action)
            cumulative_reward += reward

        episode_rewards.append(cumulative_reward)
        print(f"Episode {ep+1} | Reward = {cumulative_reward:.2f}")
        print("👉 Notez 'Nbr de sorties' et 'temps passe a la machine' dans FlexSim")
        input("Appuyez sur Entrée pour continuer...")
    # IMPORTANT: si vous aviez env._release_flexsim() avant, gardez-le
    '''env.unwrapped._release_flexsim()
    env.close()'''
    print("Evaluation terminée. Regardez maintenant les Performance Measures dans FlexSim.")
    input("Appuyez sur Entrée pour fermer FlexSim...")

    env.unwrapped._release_flexsim()
    env.close()

    # === Tracé de la courbe des rewards d'évaluation ===
    episodes = np.arange(1, n_episodes + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, episode_rewards, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("PPO Reward Convergence (FlexSim) - Final Eval")
    plt.grid(True)

    plt.savefig(
        "C:/Users/koulo/OneDrive/Bureau/Articles/article 9/ppo_evaluation_rewards.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # >>> AJOUT: messages pour retrouver les fichiers de rewards training
    print("\n✅ Training rewards/logs saved in:", log_dir)
    print("➡️ Courbe type 'training' (rollout/ep_rew_mean) :",
          os.path.join(log_dir, "progress.csv"))
    print("➡️ Monitor (récompense par épisode pendant training) :",
          os.path.join(log_dir, "monitor.csv"))

if __name__ == "__main__":
    main()
