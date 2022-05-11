#
# Policy-gradient with Monte Carlo (REINFORCE)
#
#

#
#"Skeleton" file for HW...
#
# Your task: Complete the missing code (see comments below in the code):
#            1. Implement Policy Gradient with Monte Carlo (REINFORCE)
#            2. Test your algorithm using the provided pg_mc_demo.py file
#


from random import Random
import torch 

#
#   Policy-gradient with Monte Carlo
#
def pg_mc(simenv, policy, gamma, alpha, num_episodes, max_episode_len, window_len=100, term_thresh=None, showPlots=False, prng_seed=789):
    '''
    Parameters:
        simenv:  Simulation environment instance
        policy:  Parameterized model for the policy (e.g. neural network)
        gamma :  Future discount factor, between 0 and 1
        alpha :  Learning step size
        num_episodes: Number of episodes to run
        max_episode_len: Maximum allowed length of a single episode
        window_len: Window size for total rewards windowed average (averaged over multiple episodes)
        term_thresh: If windowed average > term_thresh, stop (if None, run until num_episodes)
        showPlots:  If True, show plot of episode lengths during training, default: False
        prng_seed:  Seed for the random number generator        
    '''

    #initialize a few things
    #    
    prng=Random()
    prng.seed(prng_seed)
    simenv.reset(seed=prng_seed)

    #
    #You might need to add some additional initialization code here,
    #  depending on your specific implementation
    #

    ###########################
    #Start episode loop
    #
    episodeLengths=[]
    episodeRewards=[]
    averagedRewards=[]

    for episode in range(num_episodes):
        if episode%100 == 0:
            print('Episode: {}'.format(episode))

        #initial state & action (action according to policy)
        state=simenv.reset()
        
        #
        #You might need to add some code here,
        #  depending on your specific implementation
        #

        #Run episode state-action-reward sequence to end
        #
        episode_length=0
        tot_reward=0
        
        while episode_length < max_episode_len:

        #
        #Fill in the missing algorithm code here!
        # (Note: test your results with the pg_mc_demo.py file)
        #
        
        
        #update stats for later plotting
        episodeLengths.append(episode_length)
        episodeRewards.append(tot_reward)
        avg_tot_reward=sum(episodeRewards[-window_len:])/window_len
        averagedRewards.append(avg_tot_reward)

        if episode%100 == 0:
            print('\tAvg reward: {}'.format(avg_tot_reward))

        #if termination condition was specified, check it now
        if (term_thresh != None) and (avg_tot_reward >= term_thresh): break


    #if plot metrics was requested, do it now
    if showPlots:
        import matplotlib.pyplot as plt
        plt.subplot(311)
        plt.plot(episodeLengths)
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.subplot(312)
        plt.plot(episodeRewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.subplot(313)
        plt.plot(averagedRewards)
        plt.xlabel('Episode')
        plt.ylabel('Avg Total Reward')
        plt.show()
        #cleanup plots
        plt.cla()
        plt.close('all')
