

from monkey import Monkey
from dqn import QLearningTable
# from DQN_M import DeepQNetwork

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()
        print observation

        step = 0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_
            if step%100 == 0 :
                print "episode = ",episode,"step = ",step,"acc = ",observation_[1],"len=",observation_[0],"reward =",reward,"done =",done

            step += 1

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    # q-table
    env = Monkey()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()


    # dqn game
    # env = Monkey()
    # RL = DeepQNetwork(env.n_actions, env.n_features,
    #                   learning_rate=0.01,
    #                   reward_decay=0.9,
    #                   e_greedy=0.9,
    #                   replace_target_iter=200,
    #                   memory_size=2000,
    #                   # output_graph=True
    #                   )
    # env.after(100, update)
    # env.mainloop()
    # RL.plot_cost()