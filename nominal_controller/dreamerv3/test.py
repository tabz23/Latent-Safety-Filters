import safety_gymnasium
import matplotlib.pyplot as plt
env_id = 'SafetyCarFormulaOne1Vision-v0'
env = safety_gymnasium.make(env_id)

obs, info = env.reset()
# plt.imshow(obs['vision'])
# plt.show()
# print()
# plt.imshow(env.task.render(256, 256, mode='rgb_array', camera_name='vision', cost={}))
# plt.show()
# print(env.observation_space)
# print(env.action_space)
# print(info)
while True:
    act = env.action_space.sample()
    obs, reward, cost, terminated, truncated, info = env.step(act)
    # plt.imshow(obs['vision'])
    # plt.show()
    # print(reward,cost,info)
    print(cost)
    print(info)
    break
    if terminated or truncated:
        break
    # print(env.render())