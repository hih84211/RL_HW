#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:23:53 2022

@author: wangyiren
"""
import numpy as np
#RL code for a 25*25 cube
initial_value_matrix = np.zeros((5,5))
initial_value_matrix[0][1] = 10
initial_value_matrix[0][3] = 5
initial_value_vector = initial_value_matrix.reshape(25)
reward_action = np.array([[0,-1,-1,0],[10,10,10,10],[0,0,-1,0],[5,5,5,5],[-1,0,-1,0],
                 [0,-1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[-1,0,0,0],
                 [0,-1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[-1,0,0,0],
                 [0,-1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[-1,0,0,0],
                 [0,-1,0,-1],[0,0,0,-1],[0,0,0,-1],[0,0,0,-1],[-1,0,0,-1]])

reward =( reward_action[:,0]+reward_action[:,1]+
         reward_action[:,2]+reward_action[:,3])/4
state_state = np.zeros((25,25))
mapping_tool = np.arange(25)
mapping_tool = mapping_tool.reshape((5,5))
state_number = 0
matrix_long = 5
matrix_width = 5
for i in range(len(mapping_tool)):
    for j in range(len(mapping_tool[0])):
        if (0 <= i-1 and i+1 < matrix_long) and ( 0 <= j-1 and j+1 < matrix_width):
            state_state[state_number][mapping_tool[i-1][j]],\
            state_state[state_number][mapping_tool[i+1][j]],\
            state_state[state_number][mapping_tool[i][j+1]],\
            state_state[state_number][mapping_tool[i][j-1]] = 0.25,0.25,0.25,0.25# not side point
        elif (0 <= i-1 and i+1 < matrix_long) and (0 > j-1):
            state_state[state_number][mapping_tool[i-1][j]],\
            state_state[state_number][mapping_tool[i+1][j]],\
            state_state[state_number][mapping_tool[i][j+1]],\
            state_state[state_number][mapping_tool[i][j]] = 0.25,0.25,0.25,0.25
        elif (0 <= i-1 and i+1 < matrix_long) and (5 <= j+1):
            state_state[state_number][mapping_tool[i-1][j]],\
            state_state[state_number][mapping_tool[i+1][j]],\
            state_state[state_number][mapping_tool[i][j]],\
            state_state[state_number][mapping_tool[i][j-1]] = 0.25,0.25,0.25,0.25
        elif (0 >= i-1) and ( 0 <= j-1 and j+1 < matrix_width):
            state_state[state_number][mapping_tool[i][j]],\
            state_state[state_number][mapping_tool[i+1][j]],\
            state_state[state_number][mapping_tool[i][j+1]],\
            state_state[state_number][mapping_tool[i][j-1]] = 0.25,0.25,0.25,0.25    
        elif (matrix_long <= i+1) and ( 0 <= j-1 and j+1 < matrix_width):
                state_state[state_number][mapping_tool[i-1][j]],\
                state_state[state_number][mapping_tool[i][j]],\
                state_state[state_number][mapping_tool[i][j+1]],\
                state_state[state_number][mapping_tool[i][j-1]] = 0.25,0.25,0.25,0.25 
        elif (0 >= i-1) and ( matrix_width <= j+1):
            state_state[state_number][mapping_tool[i][j]],\
            state_state[state_number][mapping_tool[i+1][j]],\
            state_state[state_number][mapping_tool[i][j-1]] = 0.5,0.25,0.25         
        elif (0 >= i-1) and (0 > j-1):
            state_state[state_number][mapping_tool[i][j]],\
            state_state[state_number][mapping_tool[i+1][j]],\
            state_state[state_number][mapping_tool[i][j+1]] = 0.5,0.25,0.25 
        elif (matrix_long <= i+1) and (matrix_width <= j+1):
            state_state[state_number][mapping_tool[i][j]],\
            state_state[state_number][mapping_tool[i-1][j]],\
            state_state[state_number][mapping_tool[i][j-1]] = 0.5,0.25,0.25
        elif (matrix_long <= i+1) and (0 > j-1):
            state_state[state_number][mapping_tool[i][j]],\
            state_state[state_number][mapping_tool[i-1][j]],\
            state_state[state_number][mapping_tool[i][j+1]] = 0.5,0.25,0.25
        state_number += 1
state_state[1,:] = 0
state_state[3,:] = 0
state_state[1,21] = 1
state_state[3,13] = 1
unit_matrix = np.zeros((25,25))
for i in range(25):
    unit_matrix[i,i] = 1
gamma = 0.9
value_matrix = np.linalg.inv(unit_matrix - gamma*state_state)@reward
# find the value function using the iteration function
count = 0
while(True):
    count +=1 
    temp = initial_value_vector
    initial_value_vector = reward + gamma*state_state@initial_value_vector
    if np.array_equal(temp,initial_value_vector):
        break
optimal_policy = state_state
count_1 = 0
optimal_value_vector = initial_value_vector
while(True):
    state_number = 0
    count_1+=1
    optimal_action_matrix = []
    state_action = np.zeros((25,4))
    for i in range(len(mapping_tool)):
        for j in range(len(mapping_tool[0])):
            #print(i,j)
            optimal_policy[state_number] = 0
            state_action[state_number] = 0
            reward[state_number] = 0
            action_greedy = []
            action_list = []
            compare_list = []
            try:
                compare_list.append((optimal_value_vector[mapping_tool[i+1][j]],'down'))
            except:
                #print('no down')
                pass
            try:
                if not(i==0):
                    compare_list.append((optimal_value_vector[mapping_tool[i-1][j]],'upper'))
                #else:
                    #print('no upper')
            except:
                pass
            try:
                if not(j==0):
                    compare_list.append((optimal_value_vector[mapping_tool[i][j-1]],'left'))
                #else:
                    #print('no left')
            except:
                pass
            try:
                compare_list.append((optimal_value_vector[mapping_tool[i][j+1]],'right'))
            except:
                #print('no right')
                pass
            # center points
            #they have four actions point to different plots, upper, down, left, right
            # state_value(q value),action_name               
            compare_list = sorted(compare_list, key=lambda x: x[0])
            highest_qvalue_tuple = max(compare_list,key=lambda compare_list:compare_list[0])
            temp_idx = compare_list.index(highest_qvalue_tuple)
            for item in compare_list[temp_idx:]:
                action_greedy.append(item)
            for item in action_greedy:
                if item[1] == 'right':
                    optimal_policy[state_number][mapping_tool[i][j+1]] = 1/len(action_greedy)
                    state_action[state_number][0] = 1/len(action_greedy)
                    reward[state_number] += (1/len(action_greedy))*reward_action[state_number][0]
                elif item[1] == 'left':
                    optimal_policy[state_number][mapping_tool[i][j-1]] = 1/len(action_greedy)
                    state_action[state_number][1] = 1/len(action_greedy)
                    reward[state_number] += (1/len(action_greedy))*reward_action[state_number][1]
                elif item[1] == 'upper':
                    optimal_policy[state_number][mapping_tool[i-1][j]] = 1/len(action_greedy) 
                    state_action[state_number][2] = 1/len(action_greedy)
                    reward[state_number] += (1/len(action_greedy))*reward_action[state_number][2]
                elif item[1] == 'down':
                    optimal_policy[state_number][mapping_tool[i+1][j]] = 1/len(action_greedy) 
                    state_action[state_number][3] = 1/len(action_greedy)
                    reward[state_number] += (1/len(action_greedy))*reward_action[state_number][3]
                else:
                    print('error on finding actions')
            state_number+=1
            for item in action_greedy:
                action_list.append(item[1])
            optimal_action_matrix.append(action_list)
    temp = optimal_value_vector
    optimal_value_vector = reward + gamma*optimal_policy@optimal_value_vector
    if np.linalg.norm( temp - optimal_value_vector) < 1e-5:
        break

if __name__ == '__main__':
    print('part one result (synchronous):','(after ',count, 'iterations)')
    print(initial_value_vector.reshape((5,5)))
    print('part two result','(after ',count_1, 'iterations)' ) 
    print(optimal_value_vector.reshape((5,5)))
    print('optimal policy:')
    print(optimal_action_matrix)