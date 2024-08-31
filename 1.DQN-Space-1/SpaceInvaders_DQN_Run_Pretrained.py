import os 
import sys 
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
import tensorflow as tf
import numpy as np

def filename_to_envname(fname="dqn-SpaceInvadersNoFrameskip-v4-R290-55-F800063"):
    """
    parse the sent saved model weight filename and extract the gym env name
    """
    fname = os.path.basename(fname)
    parts = fname.split('-')
    assert len(parts) == 6, "fname looks invalid - expected 6 parts e.g. dqn-SpaceInvadersNoFrameskip-v4-R290-55-F800063 but got " + fname
    envname = parts[1] + "-" + parts[2] 
    return envname  

def run_sim(env:gym.Env, model:keras.Sequential, frame_count:int):
    env.reset()
    state,reward,done,t,i = env.step(1) # fire to start 
    total_reward = 0
    lives = i['lives'] 
    for i in range(frame_count):
        state_tensor = tf.expand_dims(state, axis=0) 
        action_values = model(state_tensor)
        action = tf.argmax(action_values[0]).numpy()
        state,reward,done,t,i = env.step(action)
        total_reward = total_reward + reward
        
        if lives > i['lives']: # we died
            lives = i['lives']
            env.step(1)
        if done: # we lost all lives
            env.reset()
            env.step(1)
    print("Sim ended : rew is ", total_reward)
    
def qmodel_create(num_actions=6) -> keras.Sequential:
    # Define the input shape explicitly
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

# sanity check user inputs. 
assert len(sys.argv) == 2, "Usage: python SpaceInvaders_DQN_Run_Pretrained.py ./Pre-trained_Model/<pre-trained-model.keras>"
weights_file = sys.argv[1] 
assert os.path.exists(weights_file), "Cannot find requested weights file " + weights_file
                      
# Environment preprocessing
seed = 42
env_name = filename_to_envname(weights_file)
print("Creating env of type ", env_name)
env = gym.make(env_name, render_mode="human")
env = AtariPreprocessing(env)
# Stack four frames
env = FrameStack(env, 4)
env.unwrapped.seed(seed)
# env.seed(seed)
num_actions = 6

# setup model and load weights 
print("Loading model from", weights_file)
try:
    model = qmodel_create(num_actions=num_actions)
except:
    # fallback in case its an old version of the model
    model = create_q_model(num_actions=num_actions)

model.load_weights(weights_file)

print("Model loaded, running sim")
run_sim(env, model, 10000)

