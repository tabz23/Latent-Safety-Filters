from gym.envs.registration import register
# from .pointmaze import U_MAZE
# from .dubins.dubins import DubinsEnv
import gym

# Register DubinsEnv with Gym
register(
    id='dubins',
    entry_point='env.dubins.dubins_wrapper:DubinsWrapper',
    max_episode_steps=500,
)

register(
    id='maniskill',
    entry_point='env.maniskill.maniskill_wrapper:ManiskillWrapper',
    max_episode_steps=500,
)
register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

##i added this. commented for now to avoid calling mujoco and installing it
# register(
#     id='point_maze',
#     entry_point='env.pointmaze:PointMazeWrapper',
#     max_episode_steps=300,
#     kwargs={
#         'maze_spec':U_MAZE,
#         'reward_type':'sparse',
#         'reset_target': False,
#         'ref_min_score': 23.85,
#         'ref_max_score': 161.86,
#         'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
#     }
# )
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)