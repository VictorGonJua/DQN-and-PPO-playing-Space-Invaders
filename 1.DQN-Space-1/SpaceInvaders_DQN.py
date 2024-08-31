# This is a simple implementation of a Deep Q-Network (DQN) to play the Atari game Space Invaders.
# This implementation is originally based on the DQN algorithm described in the paper "Human-level control through deep reinforcement learning" by Mnih et al.
# Also, this implementation is based on the DQN implementation done by Matthew Yee-King in the course Artificial Intelligence, part of the BSc in Computer Science at Goldsmiths, University of London.
# Key differences from the DQN implementation by Matthew Yee-King:
# - Adaptation of the code to work with the Atari environment Space Invaders.
# - Refactoring of the code allowing one-single file execution.
# - Adding the ability to save the model key performance indicators to a CSV file at the end of each episode.

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import sys
import csv
import keras
from keras import layers
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
import tensorflow as tf
from datetime import datetime

# Setup environment
env_name = "SpaceInvadersNoFrameskip-v4"
env = gym.make(env_name)
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
env.seed(42)
num_actions = 6

# Configuration parameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 32
max_steps_per_episode = 10000
max_episodes = 10000
save_interval = 500

# Function to create the Q model
def qmodel_create(num_actions=6) -> keras.Sequential:
    model = keras.Sequential()
    model.add(layers.Input(shape=(4, 84, 84)))
    model.add(layers.Lambda(lambda tensor: tf.transpose(tensor, [0, 2, 3, 1])))
    model.add(layers.Conv2D(32, 8, strides=4, activation="relu"))
    model.add(layers.Conv2D(64, 4, strides=2, activation="relu"))
    model.add(layers.Conv2D(64, 3, strides=1, activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(num_actions, activation="linear"))
    return model

# Create DQN models
model = qmodel_create(num_actions=num_actions)
model_target = qmodel_create(num_actions=num_actions)
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_function = keras.losses.Huber()

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000
update_after_actions = 4
update_target_network = 10000

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Initialize CSV file with timestamp
csv_file = f'episode_stats_{timestamp}.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Frames', 'Total Reward', 'Epsilon', 'Loss', 'Max Q-Value', 'Lives Lost', 'End Time'])

# Function to print training progress
def print_progress(frame, episode, max_episodes, next_save_at, running_reward):
    sys.stdout.write(f"\rF[{frame}] Ep:{episode}/{max_episodes}, save at: {next_save_at} reward: {running_reward:.2f}")
    sys.stdout.flush()

# Function to save the model
def save_model(model: keras.Sequential, env_name, frame, running_reward):
    fname = "dqn-"+env_name+ "-R" + str(round(running_reward, 2)).replace('.', '-') + "-F" + str(frame) + ".keras"
    print(" Saving to ", fname)
    model.save(fname)

# Training loop
while True:
    observation, _ = env.reset()
    initial_lives = 3  # Initial number of lives in the game
    lives = initial_lives
    state = np.array(observation)
    episode_reward = 0
    episode_frames = 0
    episode_loss = 0  # Initialize episode loss
    max_q_value = float('-inf')  # Initialize maximum Q-value
    next_save_at = save_interval - (episode_count % save_interval)

    # Save the model at specified intervals
    if next_save_at == 1:
        save_model(model, env_name, frame_count, running_reward)

    for timestep in range(1, max_steps_per_episode):
        print_progress(frame_count, episode_count, max_episodes, next_save_at, running_reward)
        frame_count += 1
        episode_frames += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_rewards = model(state_tensor, training=False)
            action = tf.argmax(action_rewards[0]).numpy()
            max_q_value = max(max_q_value, tf.reduce_max(action_rewards).numpy())  # Track max Q-value

        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in the environment
        state_next, reward, done, _, info = env.step(action)
        state_next = np.array(state_next)
        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update model every 'update_after_actions' frames
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])
            future_rewards = model_target.predict(state_next_sample, verbose=False)
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = tf.reduce_sum(q_values * masks, axis=1)
                loss = loss_function(updated_q_values, q_action)

            episode_loss += loss.numpy()  # Accumulate loss for the episode
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update the target network periodically
        if frame_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())

        # Limit the state and reward history to the max memory length
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        # Automatically press fire button if life is lost
        if lives > info['lives']:
            env.step(1)
            lives = info['lives']
        if done:
            break

    # Update running reward to check condition for solving the game
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)
    episode_count += 1

    # Calculate number of lives lost
    lives_lost = initial_lives - lives

    # Get the end time of the episode
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save episode statistics to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode_count, episode_frames, episode_reward, epsilon, episode_loss / episode_frames, max_q_value, lives_lost, end_time])

    if running_reward > 10000:  # Condition to consider the task solved
        print(f"Solved at episode {episode_count}!")
        break

    if max_episodes > 0 and episode_count >= max_episodes:  # Stop after max episodes
        print(f"Stopped at episode {episode_count}!")
        break
