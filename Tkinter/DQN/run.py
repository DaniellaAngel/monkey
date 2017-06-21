

from monkey import Monkey
from dqn import QLearningTable


import numpy as np
import tensorflow as tf

import true_acc


def update():
    for episode in range(100):
        # initial observation
        GOALACC = env.GOALACC
        observation = env.reset()
        print observation

        step = 0
        while True:
            # fresh env
            env.render()
            if step < 30:
                # print "step",step
                action = 0
                step += 1
                observation_, reward, done,result = env.step(action)
            else:
            # RL choose action based on observation
                action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
                observation_, reward, done,result = env.step(action)
                acc_train, acc_test = true_acc.caculate(np.array(result).astype('int'))

                if (observation_[1] > GOALACC or observation_[1] == GOALACC) and observation_[0] > env.REFERLEN:
                
                    print "GOALACC = " ,GOALACC
                    print "choose train acc = ",acc_train,"total_train_acc=",true_acc.acc1
                    print "choose test acc = ",acc_test,"total_test_acc=",true_acc.acc2


            # RL learn from this transition
                RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
                observation = observation_

                if step%100 == 0 :
                    print "|episode = ",episode,"|step = ",step,"|train acc = ",observation_[1],"|len=",observation_[0],"|reward =",reward,"|done =",done

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
