import numpy as np


def get_R():
    shape = (5, 5)
    neg = np.zeros(shape)
    neg[0] = np.ones(shape[1])
    neg[shape[1] - 1] = np.ones(shape[1])
    neg = neg + neg.T
    neg[0][1] = 0
    neg[0][3] = 0

    posA = np.zeros(shape)
    posA[0][1] = 1

    posB = np.zeros(shape)
    posB[0][3] = 1

    return ((-1 / 4) * neg + 10 * posA + 5 * posB).reshape(-1)


def get_P():
    p_mat = np.zeros((25, 25))
    for i in range(25):
        tmp = np.zeros((5, 5))
        index = (i//5, i % 5)
        if index == (0, 1):  # check point A
            tmp[4][1] = 4
        elif index == (0, 3):  # check point B
            tmp[2][3] = 4
        else:
            if index[0] == 0:  # North
                tmp[index[0]][index[1]] += 1
            else:
                tmp[index[0] - 1][index[1]] += 1
            if index[0] == 4:  # South
                tmp[index[0]][index[1]] += 1
            else:
                tmp[index[0] + 1][index[1]] += 1
            if index[1] == 0:  # West
                tmp[index[0]][index[1]] += 1
            else:
                tmp[index[0]][index[1] - 1] += 1
            if index[1] == 4:  # East
                tmp[index[0]][index[1]] += 1
            else:
                tmp[index[0]][index[1] + 1] += 1
        p_mat[i] = (tmp / 4).reshape(-1)
    return p_mat


if __name__ == '__main__':
    gamma = 0.9
    R_pi = get_R()
    P_pi = get_P()

    # print(R_pi.reshape((5, 5)))
    '''
    for i in range(25):
        print((i//5, i%5))
        print(P_pi[i].reshape((5, 5)))
        print()
    '''

    V_pi = np.linalg.inv(np.eye(25) - gamma * P_pi) @ R_pi

    print(V_pi.reshape((5, 5)))
