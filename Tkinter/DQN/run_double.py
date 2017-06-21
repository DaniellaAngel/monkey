from doubleDQN import DoubleDQN
from monkey import Monkey
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import true_acc
GOALACC = 0.5600


env = Monkey()
# env.seed(1)
MEMORY_SIZE = 3000
# ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()
        env.render()
        action = RL.choose_action(observation)

        # f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        observation_, reward, done, result = env.step(action)

        # reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.
        acc_train, acc_test = true_acc.caculate(np.array(result).astype('int'))

        if observation_[1] > GOALACC or observation_[1] == GOALACC:
            # acc_train, acc_test = true_acc.caculate(np.array(result).astype('int'))
            print "choose train acc = ",acc_train,"total_train_acc=",true_acc.acc1
            print "choose test acc = ",acc_test,"total_test_acc=",true_acc.acc2

        opt = acc_test-true_acc.acc2
        if opt > 0.01 or opt == 0.01:
            print result
            print "opt choose test acc = ",acc_test,"opt total_test_acc=",true_acc.acc2

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:   # stop game
            break

        if total_steps%100 == 0 :
                print "|step = ",total_steps,"|train acc = ",observation_[1],"|len=",observation_[0],"|reward =",reward,"|done =",done

        observation = observation_
        total_steps += 1
    return RL.q

q_natural = train(natural_DQN)
q_double = train(double_DQN)

plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()
