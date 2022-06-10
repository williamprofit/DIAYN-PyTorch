import QDgymAurora
import gym
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm

params = get_params()

test_env = gym.make(params["env_name"])
n_states = test_env.observation_space.shape[0]
n_actions = test_env.action_space.shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

params.update(
    {"n_states": n_states, "n_actions": n_actions, "action_bounds": action_bounds}
)
print("params:", params)
test_env.close()
del test_env, n_states, n_actions, action_bounds

env = gym.make(params["env_name"])

p_z = np.full(params["n_skills"], 1 / params["n_skills"])
agent = SACAgent(p_z=p_z, **params)
logger = Logger(agent, **params)

# Load weights
logger.load_weights()
player = Play(env, agent, n_skills=params["n_skills"])
player.evaluate()
