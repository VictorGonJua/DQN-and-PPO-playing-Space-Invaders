import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# Helper function to initialize layers (used during training)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Define the agent class (should be the same as the one used during training)
class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, env.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_action(self, x):
        with torch.no_grad():
            hidden = self.network(x / 255.0)
            logits = self.actor(hidden)
            action = torch.argmax(logits, dim=1)
        return action

# Define the environment creation function
def make_env(env_id):
    env = gym.make(env_id, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env

# Main function to run the simulation
def main():
    # Load the pre-trained model
    model_path = r".\SpaceInvadersNoFrameskip-v4__SpaceInvaders_PPO__1__1722884080_model_10000.pth"  # Use raw string literal for the path
    env_id = "SpaceInvadersNoFrameskip-v4"
    env = make_env(env_id)
    agent = Agent(env)
    agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.get_action(obs)
        obs, reward, done, _, _ = env.step(action.item())
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()
