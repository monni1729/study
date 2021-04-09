import tensorflow as tf
import numpy as np
import gym
import time

env = gym.make("CartPole-v1")
initializer = tf.variance_scaling_initializer()

x = tf.placeholder(tf.float32, shape=[None, 4])
hidden1 = tf.layers.dense(x, 4, activation=tf.nn.elu, kernel_initializer=initializer)
hidden2 = tf.layers.dense(hidden1, 4, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden2, 1, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)
prob_left_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(prob_left_right), num_samples=1)
y_pred = 1 - tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_pred, logits=logits)
optimizer = tf.train.AdamOptimizer(0.05)
grads_and_vars = optimizer.compute_gradients(cross_entropy)

gradients = [grad for grad, _ in grads_and_vars]

gradients_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradients_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradients_placeholders.append(gradients_placeholder)
    grads_and_vars_feed.append((gradients_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def discount_rewards(rewards, df=0.95):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * df
        discounted_rewards[step] = cumulative_rewards

    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_factor=0.95):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    rewards_mean = flat_rewards.mean()
    rewards_std = flat_rewards.std()
    temp = []
    for discounted_rewards in all_discounted_rewards:
        temp.append((discounted_rewards - rewards_mean) / rewards_std)
    return temp


def training(iterations=30, load=True):
    with tf.Session() as sess:
        init.run()
        if load:
            saver.restore(sess, "PG_w.ckpt")

        for iteration in range(iterations):
            print("iterations", iteration)
            all_rewards = []
            all_gradients = []

            sum_step = 0
            for game in range(10):
                current_rewards = []
                current_gradients = []

                obs = env.reset()
                for step in range(500):
                    action_val, gradients_val = sess.run([action, gradients], feed_dict={x: obs.reshape(1, 4)})
                    obs, reward, done, _ = env.step(action_val[0, 0])
                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)

                    if done:
                        sum_step += step
                        break

                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)

            print("step", sum_step / 10)
            all_rewards = discount_and_normalize_rewards(all_rewards)
            feed_dict = {}
            for var_index, grad_placeholder in enumerate(gradients_placeholders):
                temp_list = []
                for game_index, rewards in enumerate(all_rewards):
                    for step, reward in enumerate(rewards):
                        temp_list.append(reward * all_gradients[game_index][step][var_index])
                mean_gradients = np.mean(temp_list, axis=0)

                feed_dict[grad_placeholder] = mean_gradients
            sess.run(training_op, feed_dict=feed_dict)

        saver.save(sess, "PG_w.ckpt")


def play():
    with tf.Session() as sess:
        init.run()
        saver.restore(sess, "PG_w.ckpt")

        sum_step = 0
        obs = env.reset()
        for step in range(500):
            action_val, gradients_val = sess.run([action, gradients], feed_dict={x: obs.reshape(1, 4)})
            obs, reward, done, _ = env.step(action_val[0, 0])
            # env.render()
            sum_step += 1
            time.sleep(0.05)
            if done:
                print(sum_step)
                env.close()
                break


if __name__ == '__main__':
    # training(iterations=10, load=True)
    play()
