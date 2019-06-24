# -*- coding: utf-8 -*-
import gym

import numpy as np
import pickle
import utils as ut

class policyIterAgent:
    def __init__(self, n_actions=1, n_states=2, roadmap=None, skdtree=None,
                 states=None, rewards=None, gamma=0.9):
        """ """
        self.n_actions = n_actions #env.action_space.shape[0]
        self.n_states  = n_states #len(states)
        self.action_list = range(self.n_actions)
        
        ## self.state_dim = env.observation_space.shape[0]
        ## self.T         = 1./float(self.n_actions)
        self.roadmap   = roadmap
        self.states    = states

        self.gamma     = gamma
        self.rewards   = rewards 
        self.policy    = None
        self.values    = None
        self.T         = self.create_transition_matrix()

        self.skdtree   = skdtree

        

    def create_transition_matrix(self):
        action_prob = 0.8
        T = np.ones((self.n_actions, self.n_actions))*(1.-action_prob)/float(self.n_actions-1)
        for i in range(self.n_actions):
            T[i,i] = action_prob
        return T


    def compute_policy_v(self, policy):
        """ Iteratively evaluate the value-function under policy.
        Alternatively, we could formulate a set of linear equations in iterms of v[s]
        and solve them to find the value function.
        """
        v = np.zeros(self.n_states)
        eps = 1e-4 #10
        while True:
            prev_v = np.copy(v)
            delta  = float("-inf")
            for s in range(self.n_states):
                ## policy_a = policy[s]
                a   = policy[s]
                ids = self.roadmap[s]
                #v[s] = np.amax(np.dot(self.T, (self.rewards[ids] + self.gamma*prev_v[ids])))
                v[s] = np.dot(self.T[a], (self.rewards[ids] + self.gamma*prev_v[ids]))                
                ## v[s] = ut.softmax( [self.T[a].dot(self.rewards[ids] + self.gamma*prev_v[ids]) \
                ##                     for a in range(self.n_actions)] )
                ## delta = max(delta, np.sum(np.fabs( prev_v-v )))
                
            ## if  delta <= eps:
            if  np.sum(np.fabs( prev_v-v )) <= eps:
                # value converged
                break
        return v

    
    def extract_policy(self, v, gamma = 1.0):
        """ Extract the policy given a value-function """
        policy = np.zeros(self.n_states).astype(int)
        ## policy = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            ids  = self.roadmap[s]
            ## policy[s] = np.dot(self.T, self.rewards[ids] + self.gamma*v[ids]) 
            policy[s] = np.argmax( np.dot(self.T, self.rewards[ids] + self.gamma*v[ids]) )
            ## policy[s] = np.array([np.dot(self.T[a], (self.rewards[ids] + self.gamma*v[ids]))
            ##              for a in range(self.n_actions)])
        return policy


    def find_policy(self, error=0.01):

        ## policy = np.random.rand(self.n_states, self.n_actions)
        ## policy /= np.expand_dims(np.sum(policy, axis=1),axis=1)
        policy = np.random.choice(self.n_actions, size=(self.n_states)).astype(int)  # initialize a random policy
        max_iterations = 200000

        last_delta = float("inf")
        for i in range(max_iterations):
            old_policy_v = self.compute_policy_v(policy)
            new_policy = self.extract_policy(old_policy_v, self.gamma)

            delta = np.sum(np.fabs(policy-new_policy))
            if delta==0.0:
                ## if (np.all(policy == new_policy)) or np.sum(np.fabs(policy-new_policy))<15 :
                ## print ('Policy-Iteration converged at step %d.' %(i+1))
                break
            ## elif delta==last_delta and delta<15:
            ##     break
            last_delta = delta
            policy = new_policy
            
        self.policy = policy
        self.values = old_policy_v
        return policy, old_policy_v


    def get_policy(self):
        return self.policy

    def set_rewards(self, rewards):
        self.rewards = rewards

    def act(self, state, e=0.):
        if type(state) is list: state = np.array(state)
        ## s_idx = self.states.index(state)
        ## near_state = self.skdtree.query(state, 1)
        ## s_idx = self.states.index(near_state)
        s_idx,_ = self.skdtree.search(state)
        a_idx = self.policy[s_idx]
        ## values      = self.policy[s_idx]
        ## best_action = np.argmax(values)
        ## values[best_action] += (1.0-e)
        
        ## a_idx = np.random.choice(self.action_list, p=values/sum(values))
        # get nearstates
        s_ids = self.roadmap[s_idx]
        nxt_state = self.states[s_ids[a_idx]]
        
        return nxt_state-state

    def reset(self):
        self.action_list = range(self.n_actions)


    def save(self, filename):
        d = {'policy': self.policy, 'rewards': self.rewards, 'roadmap': self.roadmap,\
             'T': self.T, "n_actions": self.n_actions, "n_states": self.n_states,
             'states': self.states, 'values': self.values}
        pickle.dump( d, open( filename, "wb" ) )
        pickle.dump( self.skdtree, open( filename+'_kdtree', "wb" ) )
        ## print "Saved a file: ", filename

    def load(self, filename):
        d = pickle.load( open( filename, "rb" ) )
        self.policy  = d['policy']
        self.values  = d['values']
        self.rewards = d['rewards']
        self.roadmap = d['roadmap']
        self.T       = d['T']
        self.n_actions = d['n_actions']
        self.n_states  = d['n_states']
        self.states  = d['states'].tolist()
        self.skdtree = pickle.load( open( filename+'_kdtree', "rb") )
        self.reset()
        ## print "Loaded a file: ", filename



