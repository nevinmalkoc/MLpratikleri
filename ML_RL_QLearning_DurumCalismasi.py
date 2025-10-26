import gymnasium as gym

import numpy as np
import random
from tqdm import tqdm

env = gym.make("Taxi-v3", render_mode= "ansi")
env.reset()
print(env.render())


"""
0 güney
1 kuzey
2 doğu
3 batı
4yolcu almak
5 yolcu bırakmak
"""
action_space = env.action_space.n
state_space = env.observation_space.n

q_table= np.zeros((state_space, action_space))


alpha= 0.1 #leraning rate
gamma = 0.6 #discounnt rate
epsilon = 0.1 #epsilon


for i in tqdm(range(1,10001)):
    
    state,_= env.reset()
    
    done=False
    
    while not done:
        
        if  random.uniform(0, 1) <epsilon: #explore
        
            action= env.action_space.sample()
        
            
        else: # exploit
            action=np.argmax(q_table[state])
            
        next_state, reward, done, info ,_= env.step(action)
        
        q_table[state,action]=q_table[state,action] + alpha * (reward+gamma * np.max(q_table[state,action])- q_table[state,action])
        
        
        
        state = next_state
        
            
print("training finished")            
#test

total_epoch,total_penalties= 0,0
episodes=100



for i in tqdm(range(1,10001)):
    
    state,_= env.reset()
    
    epochs , penalties ,reward = 0, 0 , 0
    
    done=False
    
    while not done:
     
            
        
        action=np.argmax(q_table[state])
            
        next_state, reward, done, info ,_= env.step(action)
        
        
        
        
        state = next_state
        
        
        
        if reward == -10:
            penalties += 1
            
            epochs += 1
            
            
    total_epoch  +=epochs
    total_penalties +=penalties
    
print("result after {} episodes".format(episodes))
print("average timesteps per episodes :", total_epoch/episodes)
print("average penalties per  episode:", total_penalties/episodes)    
            