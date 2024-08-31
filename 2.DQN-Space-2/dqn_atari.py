import os
import random
import time
from dataclasses import dataclass
import csv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer

# Data class to store the arguments required for the experiment
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm-specific arguments
    env_id: str = "SpaceInvadersNoFrameskip-v4"
    total_timesteps: int = 10000000
    learning_rate: float = 1e-4
    num_envs: int = 1
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 32
    start_e: float = 1
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80000
    train_frequency: int = 4
    save_frequency: int = 100

# Function to create and wrap the environment with necessary preprocessing
def make_env(env_id, seed, idx, run_name):
    def thunk():
        env = gym.make(env_id)
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
        env.action_space.seed(seed)
        return env
    return thunk

# Neural network model representing the Q-Network
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

# Function to linearly schedule epsilon (exploration rate)
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# Main execution block
if __name__ == "__main__":
    import stable_baselines3 as sb3

    # Ensure the correct version of dependencies is installed
    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )

    # Parse arguments from command line or defaults
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Setup CSV file to log training performance
    csv_file = f"./{run_name}_training_log.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, mode="w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Episode", "Frames", "Reward", "Epsilon", "Loss", "Max Q-Value", "Time"])

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Select device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create environment(s)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize Q-Networks (policy and target)
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Initialize Replay Buffer for experience replay
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # Reset the environment and initialize variables
    obs, _ = envs.reset(seed=args.seed)
    episode_number = 0  
    loss = None
    last_loss = None

    # Main training loop
    for global_step in range(args.total_timesteps):
        # Calculate current epsilon for exploration
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        # Select action: either random (exploration) or based on Q-values (exploitation)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Step the environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Handle end of episodes and log results
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_number += 1  
                    q_values = q_network(torch.Tensor(obs).to(device)) 
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    
                    try:
                        with open(csv_file, mode="a", newline="") as f:
                            writer_csv = csv.writer(f)
                            writer_csv.writerow([
                                episode_number,
                                global_step,
                                info["episode"]["r"][0], 
                                epsilon,
                                last_loss if last_loss is not None else "N/A",
                                q_values.max().item(), 
                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                            ])
                    except Exception as e:
                        print(f"Failed to write to CSV file: {e}")
                        
                    # Save the model at specified intervals
                    if episode_number % args.save_frequency == 0:
                        model_path = f"./{run_name}_episode_{episode_number}.pth"
                        torch.save(q_network.state_dict(), model_path)
                        print(f"Model saved to {model_path} after {episode_number} episodes")

        # Handle truncations and update replay buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # Update current observation
        obs = next_obs

        # Perform learning step after sufficient exploration
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)
                last_loss = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network at specified intervals
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    # Close the environment
    envs.close()
