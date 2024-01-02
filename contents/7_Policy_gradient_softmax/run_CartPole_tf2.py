import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 游戏环境
env: gym.Env = gym.make("CartPole")
state_shape: tuple = (4, )
action_num: int = 2

# 神经网络，输出为不同动作的概率
policy_gradient_net: keras.Model = keras.Sequential([
    keras.layers.Dense(10, activation=keras.activations.relu, input_shape=state_shape),
    keras.layers.Dense(action_num, activation=keras.activations.softmax)
])
policy_gradient_net.build(input_shape=state_shape)
policy_gradient_net.summary()

# 训练相关参数
learning_rate: float = 0.02
optimizer: keras.optimizers = keras.optimizers.Adam(learning_rate)

# 轨迹
trajectory: list = []

for episode_index in range(3000):
    # 1个epoch
    observation: np.ndarray
    observation, _ = env.reset()
    evolving_reward: float = None
    while True:
        # 1个step,直到env状态是结束

        # 把observation传进神经网络进行正向传播得到每个动作的概率
        # 然后根据概率进行二项分布的采样
        assert observation.shape == state_shape
        pi: tf.Tensor = policy_gradient_net(np.array([observation]))
        assert len(np.squeeze(pi)) == action_num
        action: int = np.random.choice(action_num, p=np.squeeze(pi))  # 根据网络算出来的概率进行随机抽样，得到动作
        next_observation, reward, terminated, _, _ = env.step(action)   # 把动作传给环境，观察执行结果

        one_step_exp = [observation, action, reward]
        trajectory.append(one_step_exp) # 保存【动作前环境观察值，动作，奖励】到轨迹

        observation = next_observation

        if terminated:
            # 计算每一步的折扣奖励
            discounted_cumulative_reward_sum = np.zeros(len(trajectory))
            running_add = 0
            for t in reversed(range(0, len(trajectory))):
                running_add = running_add * 0.9 + trajectory[t][2]
                discounted_cumulative_reward_sum[t] = running_add
            discounted_cumulative_reward_sum -= np.mean(discounted_cumulative_reward_sum)
            discounted_cumulative_reward_sum /= np.std(discounted_cumulative_reward_sum)    # 把折扣累计奖励和归一化

            with tf.GradientTape() as tape:
                action_probs = policy_gradient_net(np.array([one_step_exp[0] for one_step_exp in trajectory]))
                softmax_cross_entropy_between_pi_and_action = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_prob, labels=one_step_exp[1]) for action_prob, one_step_exp in zip(action_probs, trajectory)]
                loss = tf.reduce_mean(tf.multiply(softmax_cross_entropy_between_pi_and_action, discounted_cumulative_reward_sum))
            grads = tape.gradient(loss, policy_gradient_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy_gradient_net.trainable_variables))

            print("EPISODE {} 结束".format(episode_index))
            break