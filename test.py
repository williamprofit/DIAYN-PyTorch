import QDgymAurora
import gym
import numpy as np

env = gym.make("QDAntOmnidirectionalBulletEnv-v0")
s = env.reset()
n_actions = env.action_space.shape[0]
d = env.desc
s, r, done, _ = env.step(np.array([1, 2, 1, 1, 4, 5, 6, 2]))
print(s)
print(env.desc)
