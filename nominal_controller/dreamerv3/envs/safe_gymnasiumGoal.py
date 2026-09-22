import safety_gymnasium
import numpy as np
import gymnasium as gym

class SafeGymnasium:
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=2,
        size=(128, 128),
        # size=(256, 256),
        seed=None,
    ):
        assert size[0] == size[1]
        if self.LOCK is None:
            import multiprocessing as mp

            mp = mp.get_context("spawn")
            self.LOCK = mp.Lock()
        self._action_repeat = action_repeat
        self._size = size
        with self.LOCK:
            self._env = safety_gymnasium.make(name,render_mode='rgb_array')
            self._env.set_seed(seed)
        self._done = True
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape + (3,), np.uint8),
                "vector": gym.spaces.Box(-np.inf, np.inf, shape=(40,), dtype=np.float32)
            }
        )

    def transform_obs(self, observation):
        obs = {}
        vectors = []
        for key in observation.keys():
            if key == "vision":
                obs[key] = observation[key]
            elif "vases" not in key and "hazards" not in key:
                vectors.append(observation[key].flatten())
        obs["vector"] = np.concatenate(vectors, axis=0)
        return obs

    @property
    def action_space(self):
        space = self._env.action_space
        return space

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            obs_dict, cur_reward, cost, terminated, truncated, info = self._env.step(action)
            reward += cur_reward
            if terminated or truncated:
                break
        obs = {}
        obs_dict = self.transform_obs(obs_dict)
        obs["image"] = obs_dict["vision"]
        obs["vector"] = obs_dict["vector"]
        obs["is_terminal"] = terminated or truncated
        obs["is_first"] = False
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self):
        obs_dict, info = self._env.reset()
        obs_dict = self.transform_obs(obs_dict)
        obs = {"is_terminal": False, "is_first": True}
        obs["image"] = obs_dict["vision"]
        obs["vector"] = obs_dict["vector"]
        return obs

    def render(self, *args, **kwargs):
        return self._env.task.render(self._size[0], self._size[1], mode='rgb_array', camera_name='vision', cost={})

    def close(self):
        return self._env.close()