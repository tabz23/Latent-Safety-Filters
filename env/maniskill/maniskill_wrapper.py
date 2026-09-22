import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
import torch
from pathlib import Path
import numpy as np
##State vector: bowl [0:12], apple [13:25], robot, [39:89]
##obs vector [0:50] - proprio
def aggregate_dct(dcts):
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

class ManiskillWrapper():
    def __init__(
            self,
            env_name = "UnitreeG1PlaceAppleInBowl-v1",
            obs_mode = "state",
            size = (224,224)
    ):
        self._env = gym.make(env_name, obs_mode = obs_mode, sensor_configs = dict(width = size[0], height = size[1]))
        self.action_dim = self._env.action_space.shape[0]

    @property
    def action_space(self):
        return self._env.action_space

    def calculate_cost(self, collision_threshold = 1e-6):
        #Get objects from environment
        bowl = self._env.unwrapped.bowl
        scene = self._env.unwrapped.scene
        robot = self._env.unwrapped.agent.robot

        #Find correct hand link
        all_links = robot.get_links()
        right_hand_link = next((link for link in all_links if link.name == 'right_palm_link'), None)
        contact_forces = scene.get_pairwise_contact_forces(right_hand_link, bowl)
        if contact_forces is not None and len(contact_forces) > 0:
            forces_magnitudes = torch.norm(contact_forces, dim = -1)
            total_force = torch.sum(forces_magnitudes).item()
            #Returns 1.0 if collision detected
            if total_force > collision_threshold:
                return 1.0
            
        return 0.0
    

    def sample_random_init_goal_states(self, seed):
        rs = np.random.RandomState(seed)
        obs, _ = self._env.reset()
        full_state = self._env.unwrapped.get_state().cpu().numpy()  
    
        init_state = full_state.copy()
        goal_state = full_state.copy()
    
        if len(init_state.shape) > 1:
            init_flat = init_state[0]
            goal_flat = goal_state[0] 
        else:
            init_flat = init_state
            goal_flat = goal_state
        init_flat[0:13] = [-0.0133, -0.4014,  0.7530,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000]
        goal_flat[0:13] = [-0.0133, -0.4014,  0.7530,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000]
        apple_xy = rs.uniform(-0.1, 0.1, 2)
        init_flat[13:16] = [apple_xy[0], apple_xy[1], 0.7335]
        init_flat[16:20] = self._random_quaternion_z_only(rs)

        goal_flat[13:26] = [-0.0201, -0.3595,  0.7765, -0.1460,  0.9200,  0.3578,  0.0654,  0.3022,
        -0.0878, -0.0166,  1.1685,  6.9355,  3.8787]

        goal_flat[39:89] = [-2.6336e-01, -9.2672e-01, -6.7181e-01,  1.6503e+00, -1.6400e-01,
        -1.1230e+00, -8.2368e-02, -2.2680e-01,  3.9840e-01,  1.6550e+00,
        -1.6204e+00, -5.2360e-01,  2.9999e-01,  2.9999e-01,  1.1011e-01,
         1.7506e+00, -2.9997e-01, -1.0000e+00, -3.9863e-06, -6.9409e-07,
        -6.2479e-01,  1.8400e+00, -3.2462e-07,  6.5954e-06,  9.9346e-06,
         3.1006e-01,  7.4477e-01,  2.7878e-01,  1.2338e+00,  1.1845e-01,
        -1.2295e+00,  9.6740e-01, -1.2078e-05,  1.2272e+00,  1.1936e+00,
        -1.2038e+00, -2.0584e-06, -7.7568e-05, -4.3718e-05,  1.6187e+00,
         8.7291e-01,  6.0899e-07,  1.6297e-04, -3.9867e-06,  2.6943e-07,
        -5.7832e-01, -4.9636e-05,  6.1666e-06,  1.5266e-05, -6.3714e-05]


        return init_state, goal_state
    

    def _random_quaternion_z_only(self, rs):
        angle = rs.uniform(0, 2*np.pi)
        w = np.cos(angle/2)
        x = 0.0
        y = 0.0
        z = np.sin(angle/2)

        return np.array([w, x, y, z])
    

    def eval_state(self, goal_state, cur_state):
        if isinstance(cur_state, torch.Tensor):
            cur_state = cur_state.cpu().numpy()
        if isinstance(goal_state, torch.Tensor):
            goal_state = goal_state.cpu().numpy()


        apple_cur = cur_state[13:16]
        robot_cur = cur_state[39:89]
        cur = np.concatenate([apple_cur, robot_cur])
    
        apple_goal = goal_state[13:16]
        robot_goal = goal_state[39:89]
        goal = np.concatenate([apple_goal, robot_goal])
    
        pos_diff = np.linalg.norm(cur - goal)
    
        success = pos_diff < 0.1
    
        state_dist = np.linalg.norm(goal_state - cur_state)
    
        return {
            'success': success,
            'state_dist': state_dist
        }
    
    def prepare(self, seed, init_state):
        self._env.reset(seed = seed)
        if len(init_state.shape) == 1:
            state_with_batch = init_state.reshape(1, -1)
        else:
            state_with_batch = init_state

        self._env.unwrapped.set_state(state_with_batch)
        image = self._env.unwrapped.render_sensors().squeeze(0)[:, -224:, :]
        state  = self._env.get_state()
        obs = {
            'proprio': state[0][39:89],
            'visual': image
        }

        return obs, state

    def step_multiple(self, actions):
        obses = []
        rewards = []
        dones = []
        infos = []

        for action in actions:
            obs, reward, truncated, terminated, info = self._env.step(action)
            visual = self._env.unwrapped.render_sensors().squeeze(0)[:, -224:, :]
            state = self._env.get_state()
            proprio = obs[0][0:50]
            obs = {'visual' : visual, 'proprio': proprio}
            obses.append(obs)
            rewards.append(reward)
            done = terminated or truncated
            dones.append(done)
            info['state'] = state
            infos.append(info)
            if terminated or truncated:
                break

        obses = aggregate_dct(obses)
        rewards = np.array(rewards)
        dones = np.array(dones)
        infos = aggregate_dct(infos)

        return obses, rewards, dones, infos 

    def rollout(self, seed, init_state, actions):
        obs, state = self.prepare(seed, init_state)  

        obses, rewards, dones, infos = self.step_multiple(actions)
        
        initial_visual = obs['visual'].unsqueeze(0)
        initial_proprio = obs['proprio'].unsqueeze(0)
    
        result_obses = {
            'visual': torch.cat([initial_visual, obses['visual']], dim=0),
            'proprio': torch.cat([initial_proprio, obses['proprio']], dim=0)
        }
    
        initial_state = state.unsqueeze(0)
        states = torch.cat([initial_state, infos['state']], dim=0)
    

        return result_obses, states
    
# # import multiprocessing
# # if multiprocessing.get_start_method(allow_none=True) is None:
# #     multiprocessing.set_start_method('spawn')
# import gymnasium as gym
# import mani_skill.envs
# from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
# import torch
# from pathlib import Path
# import numpy as np
# import multiprocessing as mp


# ##State vector: bowl [0:12], apple [13:25], robot, [26:88]

# def aggregate_dct(dcts):
#         full_dct = {}
#         for dct in dcts:
#             for key, value in dct.items():
#                 if key not in full_dct:
#                     full_dct[key] = []
#                 full_dct[key].append(value)
#         for key, value in full_dct.items():
#             if isinstance(value[0], torch.Tensor):
#                 full_dct[key] = torch.stack(value)
#             else:
#                 full_dct[key] = np.stack(value)
#         return full_dct


# class ManiskillWrapper(gym.Env):
#     def __init__(
#             self,
#             env_name = "UnitreeG1PlaceAppleInBowl-v1",
#             obs_mode = "rgb+depth",
#             size = (224,224),
            
#     ):
#         self._cuda_initialized = False
#         self._env = gym.make(env_name, obs_mode = obs_mode, 
#                              render_backend = "cpu", 

#                              sensor_configs = dict(width = size[0], height = size[1]))
#         self._env = FlattenRGBDObservationWrapper(self._env)
#         self.action_dim = self._env.action_space.shape[0]
        
#     def _initialize_cuda(self):
#         if not self._cuda_initialized:
#             # Perform CUDA-related operations here
#             self._cuda_initialized = True

#     @property
#     def action_space(self):
#         return self._env.action_space

#     def calculate_cost(self, collision_threshold = 1e-6):
#         #Get objects from environment
#         bowl = self._env.unwrapped.bowl
#         scene = self._env.unwrapped.scene
#         robot = self._env.unwrapped.agent.robot

#         #Find correct hand link
#         all_links = robot.get_links()
#         right_hand_link = next((link for link in all_links if link.name == 'right_palm_link'), None)
#         contact_forces = scene.get_pairwise_contact_forces(right_hand_link, bowl)
#         if contact_forces is not None and len(contact_forces) > 0:
#             forces_magnitudes = torch.norm(contact_forces, dim = -1)
#             total_force = torch.sum(forces_magnitudes).item()

#             if total_force > collision_threshold:
#                 return 0.8
            
#         return 0.0
    

#     def sample_random_init_goal_states(self, seed):
#         self._initialize_cuda()
#         rs = np.random.RandomState(seed)
#         obs, _ = self._env.reset()
#         full_state = self._env.unwrapped.get_state().cpu().numpy()  
    
#         init_state = full_state.copy()
#         goal_state = full_state.copy()
    
#         if len(init_state.shape) > 1:
#             init_flat = init_state[0]
#             goal_flat = goal_state[0] 
#         else:
#             init_flat = init_state
#             goal_flat = goal_state
    
#         bowl_xy = rs.uniform(-0.025, 0.025, 2) + np.array([0.0, -0.4])
#         bowl_z = 0.753
    
#         init_flat[0:3] = [bowl_xy[0], bowl_xy[1], bowl_z]
#         goal_flat[0:3] = [bowl_xy[0], bowl_xy[1], bowl_z]
    
#         apple_xy = rs.uniform(-0.1, 0.1, 2)
#         init_flat[13:16] = [apple_xy[0], apple_xy[1], 0.7335]
#         goal_flat[13:16] = [bowl_xy[0], bowl_xy[1], bowl_z + 0.02]
    
    
#         return init_state, goal_state
    

#     def _random_quaternion_z_only(self, rs):

#         angle = rs.uniform(0, 2*np.pi)

#         w = np.cos(angle/2)
#         x = 0.0
#         y = 0.0
#         z = np.sin(angle/2)

#         return np.array([w, x, y, z])
    

#     def eval_state(self, goal_state, cur_state):
#         if isinstance(cur_state, torch.Tensor):
#             cur_state = cur_state.cpu().numpy()
#         if isinstance(goal_state, torch.Tensor):
#             goal_state = goal_state.cpu().numpy()


#         apple_cur = cur_state[13:16]
#         robot_cur = cur_state[26:89]
#         cur = np.concatenate([apple_cur, robot_cur])
    
#         apple_goal = goal_state[13:16]
#         robot_goal = goal_state[26:89]
#         goal = np.concatenate([apple_goal, robot_goal])
    
#         pos_diff = np.linalg.norm(cur - goal)
    
#         success = pos_diff < 0.1
    
#         state_dist = np.linalg.norm(goal_state - cur_state)
    
#         return {
#             'success': success,
#             'state_dist': state_dist
#         }
    
#     def prepare(self, seed, init_state):
#         self._env.reset(seed = seed)
#         if len(init_state.shape) ==1:
#             state_with_batch = init_state.reshape(1, -1)
#         else:
#             state_with_batch = init_state

#         self._env.unwrapped.set_state(state_with_batch)
#         dummy_action = np.zeros(self.action_dim)
#         obs, _, _, _, _ = self._env.step(dummy_action)
        
#         state  = self._env.get_state()
#         obs = {
#             'visual': obs['rgb'][0][:, :, 3:6],
#             'proprio': obs['state'][0][26:58]
#         }

#         return obs, state

#     def step_multiple(self, actions):
#         obses = []
#         rewards = []
#         dones = []
#         infos = []

#         for action in actions:
#             obs, reward, truncated, terminated, info = self._env.step(action)
#             visual = obs['rgb'][0][:, :, 3:6]
#             state = self._env.get_state()
#             proprio = obs['state'][0][26:58]
#             obs = {'visual' : visual, 'proprio': proprio}
#             obses.append(obs)
#             rewards.append(reward)
#             done = terminated or truncated
#             dones.append(done)
#             info['state'] = state
#             infos.append(info)
#             if terminated or truncated:
#                 break

#         obses = aggregate_dct(obses)
#         rewards = np.array(rewards)
#         dones = np.array(dones)
#         infos = aggregate_dct(infos)

#         return obses, rewards, dones, infos 

#     def rollout(self, seed, init_state, actions):
#         obs, state = self.prepare(seed, init_state)  

#         obses, rewards, dones, infos = self.step_multiple(actions)
        
#         initial_visual = obs['visual'].unsqueeze(0)
#         initial_proprio = obs['proprio'].unsqueeze(0)
    
#         result_obses = {
#             'visual': torch.cat([initial_visual, obses['visual']], dim=0),
#             'proprio': torch.cat([initial_proprio, obses['proprio']], dim=0)
#         }
    
#         initial_state = state.unsqueeze(0)
#         states = torch.cat([initial_state, infos['state']], dim=0)
    

#         return result_obses, states
    
#     def reset(self, **kwargs):
#         return self._env.reset(**kwargs)

#     def step(self, action):
#         return self._env.step(action)

#     def render(self, mode="human"):
#         return self._env.render(mode=mode)

#     def close(self):
#         return self._env.close()

#     @property
#     def observation_space(self):
#         return self._env.observation_space

#     # @property
#     # def unwrapped(self):
#     #     return self._env.unwrapped
    
#     # @property
#     # def spec(self):
#     #     return getattr(self._env, "spec", None)
    
    
#     ##Rollout and step multiple should return observation, containing visual and proprio