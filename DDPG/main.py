import gym
import numpy as np
from ddpg_agent import Agent
import os


    
if __name__ == '__main__':

    try:
        os.makedirs("temp/ddpg")
    except:
        pass
    
    env = gym.make('BipedalWalker-v3')


    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    
    n_games = 250


    best_score = env.reward_range[0]
    score_history = []
    
    evaluate = False

    for i in range(n_games):
        steps = 0 
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            step +=1
            env.render()
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            if step > 200:
                done = True

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

