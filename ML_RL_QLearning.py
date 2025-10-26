import gymnasium as gym
import random
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


environment= gym.make("FrozenLake-v1", is_slippery =False, render_mode="ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions= environment.action_space.n
qtable=np.zeros((nb_states,nb_actions))


print("Q-table:")
print(qtable)

action=environment.action_space.sample()

#S1 -> (Action 1) -> S2  

#new_state, reward, done,info, _ = environment.step(action)
new_state, reward, terminated, truncated, info = environment.step(action)
done = terminated or truncated


#%%



import gymnasium as gym

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from tqdm import tqdm
import matplotlib.pyplot as plt

environment= gym.make("FrozenLake-v1", is_slippery =False, render_mode="ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions= environment.action_space.n
qtable=np.zeros((nb_states,nb_actions))


print("Q-table:")
print(qtable)  # ajanın beyni


episodes = 1000 #episode
alpha=0.5 #learning rate
gamma =0.9# discount rate

outcomes= []


#training

for _ in tqdm(range(episodes)):
    state, _ = environment.reset()
    done = False  #ajanın başarı durumu
    
    outcomes.append("Failure")
    
    while not done:   # ajan  basarili olana kadar state icerisinde hareket et 
        # action
        if np.max(qtable[state])> 0 :
            action= np.argmax(qtable[state])
        else:
            action=environment.action_space.sample()
            
        new_state, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated    

        # update q table
        qtable[state,action] =qtable[state, action] + alpha * (reward+ gamma * np.max(qtable[new_state])- qtable[state,action])
        
        state=new_state
        
        if reward:
            outcomes[-1]="Success"
        

print("Qtable after training:")
print(qtable)

plt.bar(range(episodes), outcomes)


# test




episodes = 100 #episode
nb_success= 0


for _ in tqdm(range(episodes)):
    state, _ = environment.reset()
    done = False  #ajanın başarı durumu
    

    
    while not done:   # ajan  basarili olana kadar state icerisinde hareket et 
        # action
        if np.max(qtable[state])> 0 :
            action= np.argmax(qtable[state])
        else:
            action=environment.action_space.sample()
            
        new_state, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated    

        state=new_state
        nb_success += reward
        
        
print("success rate:",nb_success/episodes)     




