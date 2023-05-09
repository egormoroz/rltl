from env1x1 import CityFlow1x1
import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy

env = CityFlow1x1('data/rl/config.json')

model = sb3.PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

model.save('ppo_mlp')

mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True)

print(f'mean_reward={mean_reward:.2f} +/- {std_reward:.2f}')

