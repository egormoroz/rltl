import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import numpy as np

from env1x1 import CityFlow1x1


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, state, episode_start, deterministic): 
        return [self.action_space.sample()], state


class CircularPolicy:
    def __init__(self, action_space, interval):
        self.action_space = action_space
        self.k = 0
        self.interval = interval

    def predict(self, observation, state, episode_start, deterministic): 
        action = (self.k // self.interval) % self.action_space.n
        self.k = (self.k + 1) % (self.action_space.n * self.interval)
        return [action], state


env = CityFlow1x1('data/rl/config.json')
env = Monitor(env)

policies = {
    'random': RandomPolicy(env.action_space),
    'mlp': sb3.PPO.load('ppo_mlp'),
}

for interval in range(1, 30 + 1):
    policies[f'circular_{interval}'] = CircularPolicy(env.action_space, interval)

results = []
for name, policy in policies.items():
    mean, std = evaluate_policy(policy, env, 
                                n_eval_episodes=10, deterministic=True)
    results.append((name, mean, std))


results.sort(key=lambda x: x[1] - x[2], reverse=True)

n = 5
print(f'Top {n} policies')
for name, mean, std in results[:n]:
    print(name, f'{-mean:.2f} +/- {std:.2f}')

