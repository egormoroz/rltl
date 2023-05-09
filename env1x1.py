import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding

import cityflow
import numpy as np

import json


class CityFlow1x1(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, config_path, steps_per_episode=100):
        with open(config_path) as f:
            cfg = json.load(f)

        with open(cfg['dir'] + cfg['flowFile']) as f:
            flow_cfg = json.load(f)

        self.start_lane_ids = []
        for flow in flow_cfg:
            route = flow['route']
            assert len(route) == 2
            self.start_lane_ids.append(route[0] + '_0')
            self.start_lane_ids.append(route[0] + '_1')

        self.step_interval = float(cfg['interval'])

        # fuck it, let's hardcode the intersection id and number of TL states
        # TODO: Make the agent aware of TL limitations
        self.inter_id = 'intersection_1_1'
        self.action_space = spaces.Discrete(9) # 9 TL phases

        # TODO: I think it's much better to normalize these in some way.
        # Though it's probably a job of an agent.
        max_vehicles_per_lane = 200
        n_start_lanes = len(self.start_lane_ids)
        self.observation_space = spaces.MultiDiscrete([max_vehicles_per_lane] * n_start_lanes)

        self.cf_engine = cityflow.Engine(config_path, thread_num=1)

        self.steps_per_episode = steps_per_episode
        self.current_step = 0
        self.is_done = False
        self.reward_range = (float('-inf'), float('+inf'))

    def step(self, action):
        assert self.action_space.contains(action), f'invalid action specified: {action}'

        self.cf_engine.set_tl_phase(self.inter_id, action)
        self.cf_engine.next_step()
        self.current_step += 1

        state, reward = self._get_state_reward()

        if self.is_done:
            logger.warn("You are calling 'step()' even though this environment "
                        "has already returned done = True. You should always call "
                        "'reset()' once you receive 'done = True' -- any further "
                        "steps are undefined behavior.")
            reward = 0

        if self.current_step == self.steps_per_episode:
            self.is_done = True

        return state, reward, self.is_done, {}

    def seed(self, n):
        self.cf_engine.set_random_seed(n)

    def reset(self):
        self.cf_engine.reset()
        self.is_done = False
        self.current_step = 0
        state, _ = self._get_state_reward()

        return state

    def set_save_replay(self, save_replay):
        self.cf_engine.set_save_replay(save_replay)

    def set_replay_path(self, path):
        self.cf_engine.set_replay_file(path)

    def _get_state_reward(self):
        waiting_per_lane = self.cf_engine.get_lane_waiting_vehicle_count()

        n_start_lanes = len(self.start_lane_ids)
        state = np.zeros(n_start_lanes, dtype=np.int64)
        for i in range(n_start_lanes):
            state[i] = waiting_per_lane[self.start_lane_ids[i]]

        # TODO: Encourage a fairer TL policy by weighting wait times
        reward = -state.sum() * self.step_interval

        return state, reward
