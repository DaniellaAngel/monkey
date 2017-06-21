

from monkey import Monkey
from DQN_M import DeepQNetwork

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf

import true_acc



def update():
    step = 0
    for episode in range(100):
        # initial observation
        GOALACC = env.GOALACC
        print "GOALACC",GOALACC
        observation = env.reset()
        print observation
        # step = 0
        while True:
            # fresh env
            env.render()
            if step < 20:
                action = 0
                observation_, reward, done,result = env.step(action)
                print "step",step
                step += 1
            else:
                # RL choose action based on observation
                action = RL.choose_action(observation)

                # RL take action and get next observation and reward
                observation_, reward, done,result = env.step(action)
                acc_train, acc_test = true_acc.caculate(np.array(result).astype('int'))

                if (observation_[1] > GOALACC or observation_[1] == GOALACC) and observation_[0] > env.REFERLEN:
                    # acc_train, acc_test = true_acc.caculate(np.array(result).astype('int'))
                    print "GOALACC = " ,GOALACC
                    print "choose train acc = ",acc_train,"total_train_acc=",true_acc.acc1
                    print "choose test acc = ",acc_test,"total_test_acc=",true_acc.acc2

                RL.store_transition(observation, action, reward, observation_)

                if (step > 200) and (step % 5 == 0):
                    RL.learn()

                observation = observation_

                if step%100 == 0 :
                    print "|episode = ",episode,"|step = ",step,"|train acc = ",observation_[1],"|len=",observation_[0],"|reward =",reward,"|done =",done
            
                # break while loop when end of this episode
                if done:
                    break
                step += 1

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    # dqn game
    env = Monkey()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.95,
                      e_greedy=0.95,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, update)
    env.mainloop()
    RL.plot_cost()
