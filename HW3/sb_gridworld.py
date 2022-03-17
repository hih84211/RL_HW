import copy

import numpy as np
from enum import IntEnum
import veval_matrix

#numpy print options
np.set_printoptions(precision=2)


#################################################################
#
# Create the Gridworld transition, action & reward model
#

#number of states/actions for this problem
nstates=25
nactions=4

#future discount rate
gamma=0.9

#Action mapping
#    
class Action(IntEnum):
    North=0
    South=1
    West=2
    East=3
A=Action #alias for shorter names!


#The reward vector r(s,a)
#
rsa=np.zeros((nstates,nactions),dtype=np.float32)
for i in range(5):
    rsa[i,A.North]=-1.0
for i in range(20,25):
    rsa[i,A.South]=-1.0
for i in range(0,25,5):
    rsa[i,A.West]=-1.0
for i in range(4,25,5):
    rsa[i,A.East]=-1.0
#special transition A->A' (state 1->21)
for i in range(nactions): rsa[1,i]=10.0
#special transition B->B' (state 3->13)
for i in range(nactions): rsa[3,i]=5.0


#state-action transition table p(s',s,a)
#
pssa=np.zeros((nstates,nstates,nactions),dtype=np.float32)

#move-north pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(5):
        pssa[i,i,A.North]=1.0
    else:
        pssa[i,i-5,A.North]=1.0
        
#move-south pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(20,25):
        pssa[i,i,A.South]=1.0
    else:
        pssa[i,i+5,A.South]=1.0
        
#move-west pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(0,25,5):
        pssa[i,i,A.West]=1.0
    else:
        pssa[i,i-1,A.West]=1.0
        
#move-east pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(4,25,5):
        pssa[i,i,A.East]=1.0
    else:
        pssa[i,i+1,A.East]=1.0
        
#special A, B cells
for move in range(nactions):
    pssa[1,21,move]=1.0
    pssa[3,13,move]=1.0

#
# End Gridworld model creation
#
#################################################################

try:
    #Policy function for uniform random policy
    policy=np.zeros((nstates,nactions),dtype=np.float32)
    policy.fill(1.0/nactions) #4 directions, 25% probability each direction

    #solve for value function

    print('Value function computed by synchronous iterative policy evaluation:')
    vi = veval_matrix.synch_veval_iter(pssa, rsa, policy, gamma, theta=1e-6, show_count=True)
    print(vi.reshape((5, 5)))
    print()

    print('Value function computed by asynchronous iterative policy evaluation:')
    vi = veval_matrix.asynch_veval_iter(pssa, rsa, policy, gamma, theta=1e-6, show_count=True)
    print(vi.reshape((5, 5)))
    print()

    improved = veval_matrix.policy_imprv(pssa, rsa, policy, gamma, vi, theta=1e-6)
    print('The optimal value function:')
    print(improved['value'].reshape((5, 5)))
    print()
    print('The optimal policy:')
    print(improved['policy'])

    ''''''''''''''''''''''''''''''''''''


except ImportError:
    #module not found, skipping
    pass
