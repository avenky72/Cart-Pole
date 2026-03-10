import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()
print(state)
env.close()