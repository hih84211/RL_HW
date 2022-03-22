from cat_and_mouse import Cat_and_Mouse
import numpy as np
from random import Random

np.set_printoptions(precision=2)

def get_trajectory(cm, prng, render=False, rand_init=False, row=0, col=0):
    if rand_init:
        cm.reset(initLoc=[prng.randint(row), prng.randint(col)])
    else:
        cm.reset()
    gameOver = False
    if render:
        cm.render()
    # print(gameOver)
    trajectory = [[], [], []]  # [S[], A[], R[]]
    newState = cm.currentState()
    while not gameOver:
        action = prng.choice([i for i in range(cm.numActions)])
        trajectory[0].append(newState)
        trajectory[1].append(action)
        newState, reward, gameOver = cm.step(action)
        trajectory[2].append(reward)
        # print('New state: {} Reward: {} GameOver: {}'.format(newState, reward, gameOver))
        if render:
            cm_world.render()
    trajectory[0].append(newState)
    trajectory[1].append(-1)
    trajectory[2].append(0)
    if render:
        cm.render()
    trajectory = np.array(trajectory)
    return trajectory

if __name__ == '__main__':
    cm_world = Cat_and_Mouse(rows=5, columns=5,
                             mouseInitLoc=[0, 0], catLocs=[[3, 2], [3, 3]],
                             stickyLocs=[[2, 4], [3, 4]], slipperyLocs=[[1, 1], [2, 1]], cheeseLocs=[[4, 4]])
    cm_world.render()
    # print('NumStates: {}'.format(cm_world.numStates))
    # print('NumActions: {}'.format(cm_world.numActions))
    ''''''''''''''''''''''''''' Get trajectory '''''''''''''''''''''''''''''''''
    prng = Random()
    # prng.seed(17)
    returns = np.zeros((cm_world.numStates, cm_world.numActions))
    Q = np.zeros((cm_world.numStates, cm_world.numActions))
    for j in range(10000):
        trajectory = get_trajectory(cm_world, prng)
        # print(trajectory)
        # print(trajectory.shape[1])
        ''''''''''''''''''''''''''' Get trajectory'''''''''''''''''''''''''''''''''
        length = trajectory.shape[1]
        gamma = 0.9
        G = 0
        for i in reversed(range(length)):
            if trajectory[0][i] not in trajectory[0][0:i]:
                s = int(trajectory[0][i])
                a = int(trajectory[1][i])
                r_t1 = trajectory[2][i]
                G = r_t1 + gamma * G
                if a == -1:
                    continue
                Q[s][a] = (Q[s][a] * returns[s][a] + G) / (returns[s][a] + 1)
                returns[s][a] += 1
    print(Q)
