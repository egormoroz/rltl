import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from env1x1 import CityFlow1x1


class MyCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.eval_env = Monitor(make_env(0, 0)())
        self.best_mean = float('-inf')
        self.save_path = save_path

    def _on_rollout_start(self) -> bool:
        if self.verbose > 0:
            mean, std = evaluate_policy(self.model, self.eval_env, 
                                        n_eval_episodes=4)
            print(f'{self.num_timesteps}. cum_reward={mean:.2f} +/- {std:.2f} ', end='')
            if mean > self.best_mean:
                self.best_mean = mean
                self.model.save(self.save_path)
                print('new best!')
            else:
                print()

        return True

    def _on_step(self):
        return True


def make_env(rank, seed=0, steps=100):
    def _init():
        env = CityFlow1x1('data/rl/config.json', steps_per_episode=steps)
        env.reset(seed=seed+rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    n_envs = 8

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    cb = MyCallback('ppo_mlp_travel_time')

    model = sb3.PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.01)
    # model = sb3.PPO.load("ppo_mlp_penalty_half", env=vec_env, verbose=1, learning_rate=0.001)
    model.learn(total_timesteps=100_000, progress_bar=True, callback=cb)

    del model
    vec_env.close()

    model = sb3.PPO.load(cb.save_path)
    env = Monitor(make_env(0, 0, steps=100)())
    mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10)

    print(f'mean_reward={mean_reward:.2f} +/- {std_reward:.2f}')
    env.close()

    # let's record some banging tunes
    env = make_env(0, 0, steps=200)()
    env.set_save_replay(True)

    obs = env.reset()
    for _ in range(env.steps_per_episode):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

