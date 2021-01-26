"""
Noam Siegel and Dolev Orgad,
Reinforcement Learning,
Ben Gurion University of the Negev, 2020
"""

from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

"""
Assumes installation of tqdm:
pip install tqdm

If this is an issue, remove the tqdm wrapper in line #40
"""


def features_vector(S):
    p, v = S
    x = abs(C_arr - np.asarray([p, v]))
    vec = np.zeros((len(x), 1))
    for i in range(len(x)):
        vec[i] = np.exp(np.dot(np.dot(-x[i].T, inv(covariance_matrix)), x[i]) / 2)
    return vec


def get_action(s, thetas, epsilon=0):
    if np.random.random() < epsilon:
        return np.random.randint(nA)
    else:
        action_probabilites = softmax(s, thetas)
        return np.random.choice(range(nA), p=action_probabilites.flatten())


def get_state_value(S, W):
    """
    Return V(S)
    :param S: current state (p, v)
    :param W: Weights of state Value function (
    :return: state value
    """
    F_arr = features_vector(S)
    return W.T @ F_arr


def softmax(s, theta):
    F_arr = features_vector(s)
    policy = np.exp(theta.T @ F_arr - max(theta.T @ F_arr))
    policy = policy / np.sum(policy)
    return policy


def get_expected_feature(S, policy):
    """
    Return expected features based on policy derived from theta with softmax
    :param theta: vector of weights
    :return:
    """
    F_arr = features_vector(S)
    return F_arr @ policy.T


def actor_critic(alpha_theta, alpha_w):
    """
    Solves for the environment
    :param alpha_w: learning rate for w (state-value func)
    :param alpha_theta: learning rate for theta (policy)
    :return: learned parameters theta, W
    """
    X, Y = [], []
    epsilon = 0.1

    W = np.ones((nC, 1))
    thetas = np.ones((nC, nA))


    S = env.reset()
    t_episode = 0
    I = 1
    for t in tqdm(range(int(20000)), desc="Actor-Critic Steps"):
        env.render()
        epsilon = 0.99999 * epsilon

        phi = features_vector(S)
        policy = softmax(S, thetas)
        A = get_action(S, thetas, epsilon)
        S_tag, R, done, info = env.step(A)

        delta = R + gamma * get_state_value(S_tag, W) - get_state_value(S, W)
        W = W + alpha_w * delta * phi

        F_mat = np.zeros((nC, nA))
        F_mat[:, A] = features_vector(S).flatten()
        thetas = thetas + alpha_theta * I * delta * (F_mat - get_expected_feature(S, policy))

        I = gamma * I
        S = S_tag
        if done or t_episode >= max_episode_steps:
            S = env.reset()
            t_episode = 0

        if t != 0 and t % x_step_size == 0:
            value = policy_value(thetas)
            print("\n***** Appending Data *****)")
            print("X: {}".format(t))
            print("Y: {}".format(value))
            print()
            X.append(t)
            Y.append(value)

    return W, thetas, X, Y


def simulate_agent(thetas):
    S = env.reset()
    for t in range(max_episode_steps):
        env.render()
        A = get_action(S, thetas)
        S, reward, done, info = env.step(A)
        if done:
            env.render()
            break
    env.close()


def policy_value(thetas):
    num_episodes = 100
    returns = []
    for i in tqdm(range(num_episodes), desc="Evaluating Policy Episode"):
        S = env.reset()
        t = 0
        discounted_rewards = []
        while True:  # begin episode
            A = get_action(S, thetas)
            S, R, done, info = env.step(A)
            discounted_rewards.append((gamma ** t) * R)
            t += 1
            if done or t >= max_episode_steps:
                break
        episode_return = np.sum(discounted_rewards)
        returns.append(episode_return)
    return np.mean(returns)


def print_title(s):
    print()
    print("*" * len(s))
    print(s)
    print("*" * len(s))


if __name__ == '__main__':
    # Initialization - GYM
    max_episode_steps = 500
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = max_episode_steps
    nA = 3
    env.reset()

    # Initialization - Actor-Critic
    gamma = 1
    alpha_theta = 0.02
    alpha_W = 0.08
    x_step_size = 500
    sigma_p = 0.04
    sigma_v = 0.0004
    covariance_matrix = np.diag((sigma_p, sigma_v))

    # Initialization
    N_p = 8
    N_v = 8
    nC = N_p * N_v
    position_min, position_max = env.observation_space.low[0] * 0.9, env.observation_space.high[0] * 0.9
    velocity_min, velocity_max = env.observation_space.low[1] * 0.9, env.observation_space.high[1] * 0.9
    C_p = np.linspace(position_min, position_max, num=N_p, endpoint=True)
    C_v = np.linspace(velocity_min, velocity_max, num=N_v, endpoint=True)
    C_arr = np.transpose([np.tile(C_p, len(C_v)), np.repeat(C_v, len(C_p))])

    # 0 Render full episode with learned policy
    print_title("Simulating Learned Policy")
    try:
        thetas = np.loadtxt("thetas.csv")
        print("A weights file was found. Simulating:")
        simulate_agent(thetas)
    except:
        print("A weights file was not found.")

    # 1 Run actor-critic
    print_title("Starting actor-critic with env{}".format(env.unwrapped.spec.id))
    W, thetas, X, Y = actor_critic(alpha_theta, alpha_W)

    # 2 Output/append graphics
    plt.figure(figsize=(12, 7))
    plt.xlabel("Steps")
    plt.ylabel("Estimated Policy Value")
    plt.title("Estimated Policy Value")
    plt.plot(X, Y)

    # 3 Show Graphics
    print_title("Showing Graphics")
    plt.show()

    # 4. Save weights to file
    fname = "weights_{}.csv".format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    print_title("Saving weights to file {}".format(fname))
    np.savetxt(fname, W)
    fname = "thetas_{}.csv".format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    print_title("Saving thetas to file {}".format(fname))
    np.savetxt(fname, thetas)

