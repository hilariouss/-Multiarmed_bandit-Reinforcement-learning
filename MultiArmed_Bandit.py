import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# [0. Multi-armed bandit initialization]
Arms = [-0.2, 0, 0.2, -3]
num_Arms = len(Arms)

# [1. Function which represents the action of pulling the bandit]
# get random value with the Gaussian distribution.
# If the param is bigger than the criteria(random value, i.e., 'result'), the agent get reward +1 and vice versa.
def GetReward(arm):# arm : the value of 'Arms'
    result = np.random.randn(1)
    if result > arm:
        return 1
    else:
        return -1

tf.reset_default_graph()

# [2. Build the agent(Neural network)]
weights = tf.Variable(tf.ones([num_Arms]))
output = tf.nn.softmax(weights)

reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)

responsible_output = tf.slice(output, action_holder, [1]) # --> policy(NN의 가중치에 대한 확률값(by softmax)을 responsible_output)
# output(by softmax)를 action_holder 값부터 1개 만큼, 즉, action_holder에 담긴 값을 output(policy, 확률 값)에서 잘라냄.
loss = -(tf.log(responsible_output)*reward_holder)# --> should be minimized for learning to select the best bandit among the multi-armed machine.
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
update = optimizer.minimize(loss)

####### Initialize the learning parameter #######
Total_episodes = 1000
Total_reward = np.zeros(num_Arms) # Mark the total reward values of corresponding multi-armed bandits

init = tf.global_variables_initializer() # --> Initialize the gloabal variable of Tensorflow

# [3. Launch the Tensorflow graph]
with tf.Session() as sess:
    sess.run(init) # Init. the global variables
    i = 0
    while i < Total_episodes:
        actions = sess.run(output)               # result of softmax
        a = np.random.choice(actions, p=actions) # choice an action among the actions(1-d array-like) with the prob. 'output'
                                                  # 'a' is not index, but the probability itself.
        action = np.argmax(actions == a)         # Extract the index of 'a' which is corresponding index among Probability list 'actions'

        reward = GetReward(Arms[action])

        _, resp, ww = sess.run([update, responsible_output, weights],\
                               feed_dict={reward_holder:[reward], action_holder:[action]})
        Total_reward[action] += reward
        if i%50 == 0:
            print("Running reward for the " + str(num_Arms) + " arms of the bandit: " + str(Total_reward))
        i+=1

print("\nThe agent thinks arm " + str(np.argmax(ww) + 1) + "is the most promising")
if np.argmax(ww) == np.argmax(-np.array(Arms)):
    print("The agent is smart! :)")
else:
    print("The agent is fool! :(")