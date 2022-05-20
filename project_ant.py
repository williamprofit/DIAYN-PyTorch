import QDgymAurora
import gym
from Brain.agent import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
import random
from itertools import count

from mapgames import CvtGridArchive, Individual

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

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

archive = CvtGridArchive(1000, 2, 100_000)
Individual._ids = count(0)

# Evaluate all skills
n_skills = params["n_skills"]

for _ in tqdm(range(params["n_evals"])):
    # pick random skill
    z = random.choice(range(n_skills))

    state = env.reset()
    state = concat_state_latent(state, z, n_skills)
    episode_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, r, done, _ = env.step(action)

        state = concat_state_latent(next_state, z, n_skills)
        episode_reward += r

    archive.attempt_add_population([
        Individual(env.desc, env.tot_reward)
    ])

    # print(f"skill: {z}, episode reward:{episode_reward:.1f}, bd:{env.desc}")

env.close()

archive.plot(save_path="./archive")
