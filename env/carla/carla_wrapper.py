from car_dreamer.carla_follow_env import CarlaFollowEnv
import numpy as np
import torch
import carla
from typing import Tuple

def get_vehicle_velocity(vehicle: carla.Actor) -> Tuple[float, float]:
    """
    Get the velocity of a vehicle.

    :param vehicle: carla.Actor

    :return: tuple(float, float), x, y velocity of the vehicle
    """
    velocity = vehicle.get_velocity()
    return velocity.x, velocity.y

def aggregate_dcts(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct

class CarlaWrapper(CarlaFollowEnv):
    def __init__(self, config, with_speed = False):
        super().__init__(config)
        self.with_speed = with_speed
        # self.action_space = self._get_action_space().

    def get_obs(self, obs):
        """
        Extract image observation from the observation dictionary.
        """
        proprio = np.array(self.state()[:6])
        if self.with_speed:
            speed = get_vehicle_velocity(self.ego)
            proprio = np.concatenate((proprio, speed))
        return {
            'visual': obs['camera'],
            'proprio': proprio,
            'raw': obs
        }

    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        _, init_state = self.reset()
        self.step([0, 0])
        _, goal_state = self.reset()
        return init_state, goal_state

    def reset(self, reset_to_state=None):
        """
        Reset the environment and return the initial observation.
        """
        self.reset_to_state = reset_to_state
        obs = super().reset()
        obs = self.get_obs(obs)
        return obs, self.state()
    
    def update_env(self, env_info):
        """
        Update the environment with the given information.
        This method can be used to update the environment state or configuration.
        """
        self.shape = env_info['shape']
    
    def eval_state(self, goal_state, cur_state):
        """
        Evaluate the current state against the goal state.
        This method can be used to determine how close the current state is to the goal state.
        """
        cur_ego_pos = np.array(cur_state[0][:2])
        cur_nonego_pos = np.array(cur_state[1][:2])
        cur = np.concatenate((cur_ego_pos, cur_nonego_pos))

        goal_ego_pos = np.array(goal_state[0][:2])
        goal_nonego_pos = np.array(goal_state[1][:2])
        goal = np.concatenate((goal_ego_pos, goal_nonego_pos))

        pos_diff = np.linalg.norm(cur - goal)

        success = pos_diff < 1.0  # Define success condition based on distance threshold
        state_dist = np.linalg.norm(np.array(cur_state) - np.array(goal_state))

        return {
            'success': success,
            'state_dist': state_dist
        }
    
    def prepare(self, seed, init_state):
        """
        Prepare the environment with a specific seed and initial state.
        This method can be used to set up the environment before starting an episode.
        """
        self.seed(seed)
        return self.reset(init_state)
    
    def step_multiple(self, actions):
        """
        Perform multiple steps in the environment with the given actions.
        This method can be used to execute a sequence of actions in the environment.
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dcts(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        # infos = aggregate_dcts(infos)
        return obses, rewards, dones, infos
    
    def rollout(self, seed, init_state, actions):
        """
        Perform a rollout in the environment with the given seed, initial state, and actions.
        This method can be used to execute a sequence of actions and collect observations, rewards, and other information.
        """
        obs, state = self.prepare(seed, init_state)
        
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
    
    def step(self, action):
        """
        Perform a single step in the environment with the given action.
        This method can be used to execute an action and return the next observation, reward, done status, and additional information.
        """
        obs, reward, done, info = super().step(action)
        obs = self.get_obs(obs)
        return obs, reward, done, info


