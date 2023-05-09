from tqdm import trange
import stable_baselines3 as sb3

from env1x1 import CityFlow1x1


env = CityFlow1x1('data/rl/config.json', steps_per_episode=100)
env.set_save_replay(True)

policy = sb3.PPO.load('ppo_mlp_penalty2')

obs = env.reset()
cum_reward = 0
for _ in trange(env.steps_per_episode):
    action, _ = policy.predict(obs, deterministic=True)
    # action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    cum_reward += reward


print(cum_reward)

