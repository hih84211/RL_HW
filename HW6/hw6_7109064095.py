import copy
from cat_and_mouse import Cat_and_Mouse
import numpy as np
from random import Random

np.set_printoptions(precision=4)


def get_trajectory(cm, prng, policy=None, epsilon=0.1, row=0, col=0, render=False, rand_init=False):
    if rand_init:
        init = [prng.randint(0, row - 1), prng.randint(0, col - 1)]
        cm.reset(initLoc=init)
        # print('Init: ', cm.currentState())
    else:
        cm.reset()
    if render:
        cm.render()
    # print(gameOver)
    gameOver = False
    actions = [a for a in range(cm.numActions)]
    trajectory = [[], [], [], []]  # [S[], A[], R[], S[]]
    newState = cm.currentState()
    while not gameOver:
        if policy is None:
            action = prng.randint(0, cm.numActions - 1)
        else:
            if prng.random() < epsilon:
                action = prng.randint(0, cm.numActions - 1)
            else:
                action = prng.choices(actions, weights=policy[cm.currentState()], k=1)[0]
        trajectory[0].append(newState)
        trajectory[1].append(action)
        newState, reward, gameOver = cm.step(action)
        trajectory[2].append(reward)
        trajectory[3].append(newState)

        # print('New state: {} Reward: {} GameOver: {}'.format(newState, reward, gameOver))
        if render:
            cm.render()
    trajectory[0].append(newState)
    trajectory[1].append(-1)
    trajectory[2].append(0)
    trajectory[3].append(0)
    if render:
        cm.render()
    trajectory = np.array(trajectory)
    return trajectory


def exp_sarsa(trajectory, q, r, alpha=0.01, p=None):
    returns = copy.copy(r)
    q_val = copy.copy(q)
    policy = copy.copy(p)
    length = trajectory.shape[1]
    gamma = 0.9
    for i in range(length):
        if trajectory[0][i] not in trajectory[0][0:i]:
            s = int(trajectory[0][i])
            a = int(trajectory[1][i])
            r_1 = trajectory[2][i]
            s_1 = int(trajectory[3][i])
            if a == -1:
                continue

            if policy is not None:
                q_val[s][a] = q_val[s][a] + alpha * (r_1 + gamma * policy[s_1] @ q_val[s_1] - q_val[s][a])
                '''exp = np.exp(q_val[s])
                policy[s] = exp / np.sum(exp, axis=0)'''
                policy[s] = np.zeros(q_val.shape[1])
                maxes = np.amax(q_val[s])
                in_index = np.argwhere(q_val[s] == maxes)
                for m in in_index:
                    policy[s][m[0]] = 1 / in_index.shape[0]
            else:
                tmp_policy = np.full(q_val.shape[1], 1/q_val.shape[1], float)
                q_val[s][a] = q_val[s][a] + alpha * (r_1 + gamma * tmp_policy @ q_val[s_1] - q_val[s][a])
            returns[s][a] += 1

    return q_val, returns, policy


if __name__ == '__main__':
    '''slippery_world2D = Cat_and_Mouse(rows=5, columns=5,
                               mouseInitLoc=[0, 0], catLocs=[[3, 2], [3, 3], [2, 1], [2, 4]],
                               slipperyLocs=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                                             [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
                                             [2, 0], [2, 2], [2, 3],
                                             [3, 0], [3, 1], [3, 4],
                                             [4, 0], [4, 1], [4, 2], [4, 3]], cheeseLocs=[[4, 4]])'''
    cm_world2D = Cat_and_Mouse(rows=5, columns=5,
                               mouseInitLoc=[0, 0], catLocs=[[3, 2], [3, 3]],
                               stickyLocs=[[3, 4], [2, 4]], slipperyLocs=[[1, 1], [2, 1]], cheeseLocs=[[4, 4]])
    prng = Random()
    prng.seed(19)
    cm_world2D.render()
    print('-------------- Part 1 --------------')
    returns = np.zeros((cm_world2D.numStates, cm_world2D.numActions))
    Q = np.zeros((cm_world2D.numStates, cm_world2D.numActions))
    epsilon = 1
    for j in range(1000):
        # print(trajectory)
        # print(trajectory.shape[1])
        trajectory = get_trajectory(cm_world2D, prng, epsilon=epsilon, rand_init=True, row=5, col=5)
        Q, returns, _ = exp_sarsa(trajectory, r=returns, q=Q, alpha=.0002)
    print('Q-value of the random-explore policy:')
    print(Q)
    print()

    print('-------------- Part 2 1D --------------')
    cm_world1D = Cat_and_Mouse(rows=1, columns=7, mouseInitLoc=[0, 3], cheeseLocs=[[0, 0], [0, 6]], stickyLocs=[[0, 2]],
                               slipperyLocs=[[0, 4]])
    cm_world1D.render()
    Policy = np.full((cm_world1D.numStates, cm_world1D.numActions), 1 / cm_world1D.numActions)
    returns = np.zeros((cm_world1D.numStates, cm_world1D.numActions))
    Q = np.zeros((cm_world1D.numStates, cm_world1D.numActions))
    returns = np.zeros((cm_world2D.numStates, cm_world2D.numActions))
    epsilon = 1
    episodes = 1000
    for j in range(episodes):
        # print(trajectory)
        # print(trajectory.shape[1])
        trajectory = get_trajectory(cm_world1D, prng, policy=Policy, epsilon=epsilon, rand_init=True, row=1, col=7)
        Q, returns, Policy = exp_sarsa(trajectory, r=returns, q=Q, alpha=.0002, p=Policy)

        ''' ϵ-greedy exploratory policy, which decays ϵ towards zero linearly with each episode. '''
        epsilon = epsilon * (1 - (j / episodes))
        if epsilon < 0.1:
            epsilon = 0.1

    print('Q*(s, a): ')
    print(Q)
    print()
    dPolicy = np.argmax(Policy, axis=1)
    print('Pi*(s): ')
    print([cm_world1D.actions[l] for l in dPolicy])
    print()

    print('-------------- Part 2 2D --------------')
    cm_world2D.reset(initLoc=[0, 0])
    Policy = np.full((cm_world2D.numStates, cm_world2D.numActions), 1 / cm_world2D.numActions)
    returns = np.zeros((cm_world2D.numStates, cm_world2D.numActions))
    Q = np.zeros((cm_world2D.numStates, cm_world2D.numActions))
    epsilon = 1
    episodes = 20000
    for j in range(episodes):
        # print(trajectory)
        # print(trajectory.shape[1])
        trajectory = get_trajectory(cm_world2D, prng, policy=Policy, epsilon=epsilon, rand_init=True, row=5, col=5)
        Q, returns, Policy = exp_sarsa(trajectory, r=returns, q=Q, alpha=.0002, p=Policy)

        ''' ϵ-greedy exploratory policy, which decays ϵ towards zero linearly with each episode. '''
        epsilon = epsilon * (1 - (j / episodes))
        if epsilon < 0.1:
            epsilon = 0.1

    print('Q*(s, a): ')
    print(Q)
    print()
    print('Pi*(s): ')
    dPolicy = np.argmax(Policy, axis=1).reshape((5, 5))
    for k in range(5):
        print([cm_world2D.actions[l] for l in dPolicy[k]])

