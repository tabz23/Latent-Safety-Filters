import cv2
import os
from env.dubins.dubins_wrapper import DubinsWrapper

if __name__ == '__main__':
    # Video output path and directory setup
    output_path = 'env/dubins/rollout.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize environment
    env = DubinsWrapper(seed=0)
    init_state, goal_state = env.sample_random_init_goal_states(seed=42)
    obs, state = env.prepare(0, init_state)

    # Determine frame size from first observation
    height, width = obs['visual'].shape[:2]

    # Set up VideoWriter for 1 fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 1.0, (width, height))

    # Optionally print initial state info
    print('Initial full state:', state)
    print('Initial proprio (agent):', obs['proprio'])
    print('Initial h:', env.compute_h(state))

    # Write first frame
    writer.write(obs['visual'])

    # Step through the environment and write each frame
    for t in range(1, 60):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        writer.write(obs['visual'])
        if done:
            print('Done at step', t)
            break

    # Finalize video file
    writer.release()
    print(f'Video saved to {output_path}')
