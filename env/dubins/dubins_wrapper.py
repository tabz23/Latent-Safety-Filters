import numpy as np
from .dubins import DubinsEnv

def aggregate_dicts(dicts):
    out = {}
    for k in dicts[0]:
        out[k] = np.stack([d[k] for d in dicts])
    return out

class DubinsWrapper(DubinsEnv):
    """
    Wrapper matching PushTWrapper interface for dino_wm.
    Separates the environment's internal goal from the planner's goal_state.
    """
    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.action_dim = self.action_space.shape[0]
        self.proprio_dim = 3  # agent [x, y, theta]

    def sample_random_init_goal_states(self, seed):
        rs = np.random.RandomState(seed)
        _, init_state = self.reset()
        _, goal_state = self.reset()
        return init_state, goal_state

    def update_env(self, env_info):
            pass

    def eval_state(self, goal_state, cur_state):
        dist = np.linalg.norm(cur_state[:2] - goal_state[:2])
        return {'success': dist < 0.5, 'state_dist': dist}##fix this, ma5as goal size

    def prepare(self, seed, init_state):
        self.seed(seed)
        obs, full = super().reset(init_state)
        return obs, full

    def step_multiple(self, actions):
        obses, rewards, dones, infos = [], [], [], []
        for a in actions:
            o, r, d, info = self.step(a)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        viz = np.stack([o['visual'] for o in obses])
        prop = np.stack([o['proprio'] for o in obses])
        obses_out = {'visual': viz, 'proprio': prop}
        return obses_out, np.array(rewards), np.array(dones), aggregate_dicts(infos)

    def rollout(self, seed, init_state, actions):
        obs0, s0 = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses:
            obses[k] = np.vstack([obs0[k][None], obses[k]])
        states = np.vstack([s0[None], infos['state']])
        return obses, states
