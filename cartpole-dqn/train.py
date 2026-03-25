import gymnasium as gym
from agent import agent

env = gym.make("CartPole-v1")
agent = agent(gamma=0.99, epsilon=1.0, epsilon_decay=0.999, batch_size=64, target_update=1000)
step_counter = 0
max_episodes = 2000

for episode in range(max_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    episode_steps = 0
    
    while not terminated and not truncated:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.buffer.store(state, action, reward, next_state, done)
        state = next_state
        agent.learn()
        
        if step_counter >= agent.target_update:
            step_counter = 0
            agent.update_target_network()
        step_counter += 1
        episode_steps += 1
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Steps: {episode_steps}, Epsilon: {agent.epsilon:.3f}")

print("Training complete")