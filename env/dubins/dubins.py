import gym
import numpy as np
import cv2
from gym import spaces

class DubinsEnv(gym.Env):
    """
    Dubins car in 2D with fixed forward speed and controllable heading.
    Two circular hazards and a goal.
    State: [agent_x, agent_y, agent_theta, hazard1_x, hazard1_y, hazard_radius, hazard2_x, hazard2_y, hazard_radius]
    """
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, seed=0):
        super().__init__()
        # state: agent(3) + hazard1(3) + hazard2(3) = 9
        state_dim = 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        # proprio: only agent [x, y, theta]
        self.proprio_space = spaces.Box(
            low=np.array([-3.0, -3.0, -np.pi], dtype=np.float32),
            high=np.array([3.0, 3.0, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        # action: heading rate only
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # hazards static
        self.hazard_size = 0.8
        self.hazards = [
            np.array([0.4, -1.2], dtype=np.float32),
            np.array([-0.4, 1.2], dtype=np.float32)
        ]
        # goal
        self.goal = np.array([2.2, 2.2], dtype=np.float32)
        self.goal_size = 0.3
        self.v_const = 1.0
        self.dt = 0.05
        # rendering
        self.render_size = 224
        self._canvas = np.ones((self.render_size, self.render_size, 3), dtype=np.uint8) * 255
        self.seed(seed)
        self.state = None

    def seed(self, seed=0):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _full_state(self):
        # [agent_x, agent_y, agent_theta, hazard1_x, hazard1_y, hazard1_radius, hazard2_x, hazard2_y, hazard2_radius]
        vec = list(self.state)
        for h in self.hazards:
            vec += [float(h[0]), float(h[1]), float(self.hazard_size)]
        return np.array(vec, dtype=np.float32)

    def reset(self, state=None):
        # agent state: [x, y, theta] or full state of length 9
        if state is None:
            low = [-3.0, -3.0, -np.pi]
            high = [3.0, 3.0, np.pi]
            agent = self.np_random.uniform(low=low, high=high).astype(np.float32)
        else:
            arr = np.array(state, dtype=np.float32)
            agent = arr[:3]
        self.state = agent.copy()
        obs = {'visual': self.render(), 'proprio': self.state.copy()}
        return obs, self._full_state()

    def step(self, action):
        # heading rate only
        dtheta = float(np.clip(action, self.action_space.low, self.action_space.high)[0])
        x, y, theta = self.state
        x2 = x + self.v_const * np.cos(theta) * self.dt
        y2 = y + self.v_const * np.sin(theta) * self.dt
        theta2 = theta + dtheta * self.dt
        self.state = np.array([x2, y2, theta2], dtype=np.float32)
        pos = self.state[:2]
        dist = np.linalg.norm(pos - self.goal)
        reward = -dist
        done = bool(dist <= self.goal_size or abs(x2) > 3.0 or abs(y2) > 3.0)
        obs = {'visual': self.render(), 'proprio': self.state.copy()}
        full = self._full_state()
        dists = [np.linalg.norm(pos - h) for h in self.hazards]
        h_val = min(dists) - self.hazard_size
        info = {'h': float(h_val), 'state': full}
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        img = self._canvas.copy()

        # Functions for mapping (x,y) to image coordinates (y axis up!)
        def to_px_x(x):
            return int((x + 3.0) / 6.0 * (self.render_size - 1))
        def to_px_y(y):
            return int((3.0 - y) / 6.0 * (self.render_size - 1))

        # hazards
        r = int(self.hazard_size / 6.0 * (self.render_size - 1))
        for h in self.hazards:
            cx, cy = to_px_x(h[0]), to_px_y(h[1])
            cv2.circle(img, (cx, cy), r, (0, 0, 255), -1)
        # goal
        rg = int(self.goal_size / 6.0 * (self.render_size - 1))
        gx, gy = to_px_x(self.goal[0]), to_px_y(self.goal[1])
        cv2.circle(img, (gx, gy), rg, (0, 255, 0), -1)
        # agent (triangle, bigger)
        x, y, theta = self.state
        size = 0.45
        pts = []
        for ang in [0, 140, -140]:
            rad = np.deg2rad(ang)
            dx = size * np.cos(theta + rad)
            dy = size * np.sin(theta + rad)
            px = to_px_x(x + dx)
            py = to_px_y(y + dy)
            pts.append((px, py))
        cv2.fillConvexPoly(img, np.array(pts, np.int32), (255, 0, 0))
        return img if mode == 'rgb_array' else None

    def compute_h(self, state=None):
        pos = (state[:2] if state is not None else self.state[:2])
        dists = [np.linalg.norm(pos - h) for h in self.hazards]
        return float(min(dists) - self.hazard_size)


class DubinsEnvOOD(gym.Env):
    """
    Dubins car in 2D with fixed forward speed and controllable heading.
    Two circular hazards and a goal.
    OOD / uncertainty source: environment parametrized differently than during training
        - Affects safety:
            - hazards position, size, and number
            - larger/smaller action space
            - forward speed
        - Doesn't affects safety:
            - goal position and size
    State: [agent_x, agent_y, agent_theta, hazard1_x, hazard1_y, hazard_radius, hazard2_x, hazard2_y, hazard_radius]
    """
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, seed=None, corner_start=True):
        super().__init__()
        # state: agent(3) + hazard1(3) + hazard2(3) = 9
        state_dim = 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        # proprio: only agent [x, y, theta]
        self.proprio_space = spaces.Box(
            low=np.array([-3.0, -3.0, -np.pi], dtype=np.float32),
            high=np.array([3.0, 3.0, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        # action: heading rate only
        self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        # hazards static
        self.hazard_size = 0.8
        self.hazards = [
            np.array([0.4, -1.2], dtype=np.float32),
            np.array([-0.4, 1.2], dtype=np.float32)
        ]
        # goal
        self.goal = np.array([2.2, 2.2], dtype=np.float32)
        self.goal_size = 0.3
        self.v_const = 1.0
        self.dt = 0.05
        # rendering
        self.render_size = 224
        self._canvas = np.ones((self.render_size, self.render_size, 3), dtype=np.uint8) * 255
        self.seed(seed)
        self.state = None
        self.corner_start = corner_start

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _full_state(self):
        # [agent_x, agent_y, agent_theta, hazard1_x, hazard1_y, hazard1_radius, hazard2_x, hazard2_y, hazard2_radius]
        vec = list(self.state)
        for h in self.hazards:
            vec += [float(h[0]), float(h[1]), float(self.hazard_size)]
        return np.array(vec, dtype=np.float32)

    def reset(self, state=None):
        # agent state: [x, y, theta] or full state of length 9
        if state is None:
            if self.corner_start:
                # starting in the bottom left corner is more interesting
                low = [-3, -3, .01]
                high = [-1.4, -1.4, np.pi/2 - .01]
                agent = self.np_random.uniform(low=low, high=high).astype(np.float32)
            else:
                low = [-3.0, -3.0, -np.pi]
                high = [3.0, 3.0, np.pi]
                agent = self.np_random.uniform(low=low, high=high).astype(np.float32)
                
        else:
            arr = np.array(state, dtype=np.float32)
            agent = arr[:3]
        self.state = agent.copy()
        obs = {'visual': self.render(), 'proprio': self.state.copy()}
        return obs, self._full_state()

    def turn_around(self):
        sign_cycle = 2 * int(self.goal[0] > 0) - 1
        self.goal = -self.goal
        low = [sign_cycle * 2.2 - .5, sign_cycle * 2.2 - .5 , - np.pi/2 * (1 + sign_cycle)]
        high = [sign_cycle * 2.2 + .5, sign_cycle * 2.2 + .5, - sign_cycle * np.pi/2]
        agent = self.np_random.uniform(low=low, high=high).astype(np.float32)
        #new_theta = theta + np.pi * (2 * int(theta < 0) - 1)
        #self.state = np.array([x, y, new_theta], dtype=np.float32)
        self.state = agent.copy()
        obs = {'visual': self.render(), 'proprio': self.state.copy()}
        return obs, self._full_state()

    def step(self, action):
        # heading rate only
        dtheta = float(np.clip(action, self.action_space.low, self.action_space.high)[0])
        x, y, theta = self.state
        x2 = x + self.v_const * np.cos(theta) * self.dt
        y2 = y + self.v_const * np.sin(theta) * self.dt
        theta2 = theta + dtheta * self.dt
        self.state = np.array([x2, y2, theta2], dtype=np.float32)
        pos = self.state[:2]
        dist = np.linalg.norm(pos - self.goal)
        reward = -dist
        done = bool(dist <= self.goal_size or abs(x2) > 3.0 or abs(y2) > 3.0)
        obs = {'visual': self.render(), 'proprio': self.state.copy()}
        full = self._full_state()
        dists = [np.linalg.norm(pos - h) for h in self.hazards]
        h_val = min(dists) - self.hazard_size
        info = {'h': float(h_val), 'state': full}
        return obs, reward, done, info

    def render(self, mode='rgb_array', color_ood=0):
        img = self._canvas.copy()

        # Functions for mapping (x,y) to image coordinates (y axis up!)
        def to_px_x(x):
            return int((x + 3.0) / 6.0 * (self.render_size - 1))
        def to_px_y(y):
            return int((3.0 - y) / 6.0 * (self.render_size - 1))

        '''def bg():
            return 255 - color_ood * np.random.randint(32) # color OOD between 0 and 7

        if color_ood:
            tlx, tly = to_px_x(-3.0), to_px_y(3.0)
            brx, bry = to_px_x(3.0), to_px_y(-3.0)
            cv2.rectangle(img, (tlx, tly), (brx, bry), (bg(), bg(), bg()), -1)'''

        # hazards
        r = int(self.hazard_size / 6.0 * (self.render_size - 1))
        for h in self.hazards:
            cx, cy = to_px_x(h[0]), to_px_y(h[1])
            cv2.circle(img, (cx, cy), r, (0, 0, 255), -1)
        # goal
        rg = int(self.goal_size / 6.0 * (self.render_size - 1))
        gx, gy = to_px_x(self.goal[0]), to_px_y(self.goal[1])
        cv2.circle(img, (gx, gy), rg, (0, 255, 0), -1)
        # agent (triangle, bigger)
        x, y, theta = self.state
        size = 0.45
        pts = []
        for ang in [0, 140, -140]:
            rad = np.deg2rad(ang)
            dx = size * np.cos(theta + rad)
            dy = size * np.sin(theta + rad)
            px = to_px_x(x + dx)
            py = to_px_y(y + dy)
            pts.append((px, py))
        cv2.fillConvexPoly(img, np.array(pts, np.int32), (255, 0, 0))

        # Generate Gaussian noise
        if color_ood:
            std_dev = color_ood * 32
            gaussian_noise = self.np_random.normal(0.0, std_dev, img.shape).astype(np.int16)
            gaussian_noise[:,:,2] = gaussian_noise[:,:,2] * 3 # blue bias
            noisy_img = np.clip(img.astype(np.int16) + gaussian_noise, 0, 255)
            img = noisy_img.astype(np.uint8)

        return img if mode == 'rgb_array' else None

    def compute_h(self, state=None):
        pos = (state[:2] if state is not None else self.state[:2])
        dists = [np.linalg.norm(pos - h) for h in self.hazards]
        return float(min(dists) - self.hazard_size)