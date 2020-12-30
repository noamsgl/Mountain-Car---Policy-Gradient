import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm
import pickle
from datetime import datetime

"""
Assumes installation of tqdm:
pip install tqdm

If this is an issue, remove the tqdm wrapper in line #40
"""


def features_vector(S):
    p, v = S
    x = abs(C_arr - np.asarray([p, v]))
    vec = np.zeros(len(x))
    for i in range(len(x)):
        vec[i] = np.exp(np.dot(np.dot(-x[i].T, inv(covariance_matrix)), x[i]) / 2)
    return vec


def get_action_values(W, S):
    F_arr = features_vector(S)
    return F_arr.dot(W)


def sarsa_lambda(lam, alpha):
    # initialize variables
    X, Y = [], []
    epsilon = 0.05

    W = np.ones((nC, nA))
    e = np.zeros((nC, nA))
    S = env.reset()
    action_values = get_action_values(W, S)
    A = get_action(action_values, epsilon)
    t_episode = 0
    for t in tqdm(range(int(20000))):
        env.render()
        epsilon = 0.99999 * epsilon
        S_tag, R, done, info = env.step(A)
        action_values_tag = get_action_values(W, S_tag)
        A_tag = get_action(action_values_tag, epsilon)
        delta = alpha * (R + gamma * action_values_tag[A_tag] - action_values[A])
        e = gamma * lam * e
        e[:, A] += features_vector(S)
        W = W + e * delta
        S = S_tag
        action_values = get_action_values(W, S)
        A = A_tag
        t_episode += 1

        if done or t_episode >= max_episode_steps:
            S = env.reset()
            action_values = get_action_values(W, S)
            A = get_action(action_values, epsilon)
            t_episode = 0

        if t != 0 and t % x_step_size == 0:
            value = policy_value(W)
            print("\n***** Appending Data ***** (α={}, λ={})".format(alpha, lam))
            print("X: {}".format(t))
            print("Y: {}".format(value))
            print()
            X.append(t)
            Y.append(value)

    return W, X, Y


def get_action(action_values, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, 3)
    else:
        return np.argmax(action_values)


def simulate_agent(W):
    S = env.reset()
    for t in range(max_episode_steps):
        env.render()
        action_values = get_action_values(W, S)
        A = get_action(action_values, 0)
        S, reward, done, info = env.step(A)
        if done:
            env.render()
            break
    env.close()


def policy_value(W):
    num_episodes = 100
    returns = []
    for i in range(num_episodes):
        S = env.reset()
        t = 0
        discounted_rewards = []
        while True:  # begin episode
            action_values = get_action_values(W, S)
            A = get_action(action_values, 0)
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
    # Initialization
    gamma = 1
    alpha = 0.02
    lam = 0.5
    x_step_size = 50000
    sigma_p = 0.04
    sigma_v = 0.0004
    max_episode_steps = 500
    covariance_matrix = np.diag((sigma_p, sigma_v))

    # initialize centers of gaussian distributions
    N_p = 20
    N_v = 5
    nC = N_p * N_v
    position_min, position_max = -1, 0.5
    velocity_min, velocity_max = -0.07, 0.07
    C_p = np.linspace(position_min, position_max, num=N_p, endpoint=True)
    C_v = np.linspace(velocity_min, velocity_max, num=N_v, endpoint=True)
    C_arr = np.transpose([np.tile(C_p, len(C_v)), np.repeat(C_v, len(C_p))])

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = max_episode_steps
    nA = 3
    env.reset()


    # 0 Render full episode with learned policy
    print_title("Simulating Learned Policy")
    try:
        W = np.loadtxt("weights_HW3.csv", delimiter=',')
        print("A weights file was found. Simulating:")
        simulate_agent(W)
    except:
        print("A weights file was not found.")


    # 1 Learn SARSA(λ) with given alpha, lam
    print_title("Starting Sarsa(λ) with α={}, λ={}".format(alpha, lam))
    W, X, Y = sarsa_lambda(lam, alpha)

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
    fname = "weights_HW3_{}.csv".format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    print_title("Saving weights to file {}".format(fname))
    np.savetxt(fname)
