import gym
import random
import numpy as np
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

