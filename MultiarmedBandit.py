import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

## 1. Bandit arms and pulling function

bandit_arms = [-.5, -.3, .2, 2] # 각 팔들에 대한 정보들 _ 기준선과 비교해 각 팔의 분포도상 위치 의미
num_arms = len(bandit_arms)
def pullBanditArms(bandit): # 가우시안 분포 상에서 가장 왼쪽(작은)의 확률 변수인 arm이 최고의 행동을 의미.
    result = np.random.randn(1)
    if result > bandit:
        # return positive REWARD
        return 1
    else:
        return -1

tf.reset_default_graph()

## 2. 뉴럴 네트워크의 피드 포워드 부분 구현

weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weights) # s(y_i) = e^y_i / sigma(e^y_k) for k = 0 ... n-1 ( 0 <= i < n )
#print(tf.shape(output))
#print(output)

## 2.1 학습 과정 구현 - 액션을 네트워크에 피드하면 비용을 계산하고 이를 통해 네트워크 업데이트
action_holder = tf.placeholder(shape=[1], dtype = tf.int32) # x 번째 bandit을 당긴다는 것
reward_holder = tf.placeholder(shape=[1], dtype = tf.float32)

responsible_output = tf.slice(output, action_holder, [1])
#print(tf.shape(responsible_output))
#print(responsible_output)

loss = -(tf.log(responsible_output)*reward_holder)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
update = optimizer.minimize(loss)

# 3. 텐서플로우 모델 구현
total_episodes = 1000
total_reward = np.zeros(num_arms)

init = tf.global_variables_initializer()

# 4. 텐서플로우 그래프 론칭
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        # 4.1. 행동 추출
        actions = sess.run(output)
        a = np.random.choice(actions, p=actions) # np.random.choice(1-d array-like or int, prob.)
        # p확률에 따라 softmax를 거친 action의 output이 추출됨.
        action = np.argmax(actions == a) # index of bandit

        # 4.2. 보상 계산
        reward = pullBanditArms(bandit_arms[action])

        # 4.3. 네트워크 업데이트 (학습)
        _, resp, ww = sess.run([update, responsible_output, weights], \
                               feed_dict={reward_holder:[reward], action_holder:[action]})

        # 4.4. 보상의 총계 업데이트
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the " + str(num_arms) + " arms of the bandit: " + str(total_reward))
        i += 1

print("\nThe agent thinks arm " + str(np.argmax(ww)+1) + " is the most promising...")
if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
    print("...and it was right!")
else:
    print("...and it was wrong!")