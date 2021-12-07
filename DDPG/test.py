import gym
import cv2
from ddpg_agent import Agent


env = gym.make('BipedalWalker-v3')

agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])

save_dir = "recording"



observation = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    observation_ , reward , done, _ = env.step(action)
    observation = observation_
    agent.remember(observation,action,reward,observation_,done)

agent.learn()
agent.load_models()
image_counter = 0
for i in range(3):
    observation = env.reset()
    done = False
    steps = 0
    while not done:
        action = agent.choose_action(observation,evaluate=True)
        observation_ , reward , done,_ = env.step(action)
        steps +=1
        image_counter += 1
        env.render()
        frame = env.render("rgb_array")
        cv2.imwrite(f'{save_dir}/{image_counter:06}.png', frame)
        observation = observation_
        if steps > 200:
            done = True
        
   
    



env.close()

