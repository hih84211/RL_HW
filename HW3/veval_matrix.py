import copy

import numpy as np


def get_rpi(rsa, policy):
    nstates=rsa.shape[0]
    # r(s,a) and p(s',s,a) averaged over policy actions
    Rpi = np.sum(policy * rsa, axis=1)
    Rpi.shape = (nstates, 1)  # reshape into column vector
    return Rpi


def get_ppi(pssa, policy):
    nstates=pssa.shape[0]
    Ppi = np.zeros((nstates, nstates), dtype=np.float32)
    for j in range(nstates):
        Ppi[j] = np.sum(policy[j] * pssa[j], axis=1)
    return Ppi


def veval_matrix(pssa, rsa, policy, gamma):
    #number of states & actions
    nstates=rsa.shape[0]
    nactions=rsa.shape[1]
    #First, let's compute a few useful intermediate matrices,
    Rpi = get_rpi(rsa, policy)
    Ppi = get_ppi(pssa, policy)

    #solve for value function using system of linear eqns: v=(I-Ppi)^-1*Rpi
    #
    ident=np.identity(nstates, dtype=np.float32)
    v=np.linalg.inv(ident-gamma*Ppi)@Rpi
    
    #return computed value function
    return v


def synch_veval_iter(pssa, rsa, policy, gamma, theta=1e-5, show_count=False):
    Rpi = get_rpi(rsa, policy)
    Ppi = get_ppi(pssa, policy)
    vk = np.zeros((25, 1))

    delta = 1
    n = 0
    while delta > theta:
        n += 1
        vk1 = Rpi + gamma * Ppi @ vk
        delta = np.linalg.norm(vk1 - vk)
        vk = vk1
    if show_count:
        print('Iteration count: ', n)
    return vk


def asynch_veval_iter(pssa, rsa, policy, gamma, theta=1e-5, show_count=False):
    Rpi = get_rpi(rsa, policy)
    Ppi = get_ppi(pssa, policy)

    vk1 = np.zeros((25, 1))
    delta = 1
    n = 0
    while delta > theta:
        n += 1
        vk = copy.copy(vk1)
        for i in range(25):
            vk1[i] = Rpi[i] + gamma * Ppi[i] @ vk1
        delta = np.linalg.norm(vk1 - vk)
    if show_count:
        print('Iteration count: ', n)
    return vk1


def policy_imprv(pssa, rsa, policy, gamma, vi=None, theta=1e-5):
    nstates = rsa.shape[0]  # 25
    nactions = rsa.shape[1]  # 4
    if vi is not None:
        vi = asynch_veval_iter(pssa, rsa, policy, gamma, theta)
    delta = 1
    vi1 = None
    while delta > theta:  # check if the value function still improving
        tmp = np.zeros((nactions, nstates))
        for i in range(nactions):
            # temporary policy one-step look-ahead operation
            step = np.zeros((nstates, nactions))
            for j in range(nstates):
                step[j][i] = 1
            one_step_eval = get_rpi(rsa, policy) + 0.9 * get_ppi(pssa, step) @ vi
            tmp[i] = one_step_eval.reshape(-1)
        vix4 = tmp.T
        policy = np.zeros((nstates, nactions))
        # Get greedy
        maxes = np.amax(vix4, axis=1)
        for i in range(nstates):
            in_index = np.argwhere(vix4[i] == maxes[i])
            for j in in_index:
                policy[i][j[0]] = 1 / in_index.shape[0]
        vi1 = synch_veval_iter(pssa, rsa, policy, gamma)
        delta = np.linalg.norm(vi1 - vi)
        vi = vi1

    return {'value': vi1, 'policy': policy}
