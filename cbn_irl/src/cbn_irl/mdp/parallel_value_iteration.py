# -*- coding: utf-8 -*-
import numpy as np
import pickle
import multiprocessing as mp
from cbn_irl.path_planning import dijkstra_planning as dp

# reference: https://github.com/wulfebw/py_dvi/blob/master/scripts/dvi.py

class valueIterAgent:
    def __init__(self, n_actions=1, n_states=2, roadmap=None, skdtree=None,
                 states=None, poses=None, rewards=None, gamma=0.9, T=None, verbose=False,
                 n_processes=mp.cpu_count()):
        """ """
        self.n_actions = n_actions #env.action_space.shape[0]
        self.n_states  = n_states #len(states)
        self.action_list = range(self.n_actions)
        self.n_processes = n_processes
        
        self.roadmap    = roadmap
        self.states     = states
        self.poses      = poses

        self.gamma     = gamma
        self.rewards   = rewards 
        self.policy    = np.zeros(self.n_states*self.n_actions)
        self.q_mat     = np.zeros(self.n_states*self.n_actions)
        self.values    = np.zeros(self.n_states)
        self.T         = T #self.create_transition_matrix()

        self.skdtree   = skdtree
        self.theta     = None

        self.verbose   = verbose
        self.reset()


    def solve_mdp(self, error=1e-10, init_values=None, return_q_mat=False, **kwargs):

        if init_values is None:
            self.values = mp.Array('f', np.zeros(self.n_states), lock=False)
        else:
            return NotImplementedError
            ## self.values = mp.Array('f', init_values, lock=False)
        self.rewards = mp.Array('f', self.rewards, lock=False)

        # update each state
        diff = float("inf")
        last_diff = None
        count    = 0
        while diff > error:
            count += 1
            diff = self.solve_step()            
            if self.verbose: print diff
            if last_diff == diff:
                if self.verbose: print "error converged to {} after {} iter".format(diff, count)
                break
            else:
                last_diff = diff

        if self.verbose: print "Coverged to {}".format(diff)
        
        self.values = np.frombuffer(self.values, dtype='float32')
        
        if return_q_mat:
            self.q_mat  = np.frombuffer(self.q_mat, dtype='float32').reshape(self.n_states,self.n_actions)        
            return self.values, q_mat
        
        return self.values
       

    def find_policy(self, error=1e-10, init_values=None):

        if init_values is None:
            self.values = mp.Array('f', np.zeros(self.n_states), lock=False)
        else:
            self.values = mp.Array('f', init_values, lock=False)
        self.rewards = mp.Array('f', self.rewards, lock=False)

        # update each state
        diff = float("inf")
        last_diff = None
        count    = 0
        while diff > error:
            count += 1
            diff = self.solve_step()            
            if self.verbose: print diff
            if last_diff == diff:
                break
            else:
                last_diff = diff

        if self.verbose: print "error converged to {} after {} iter".format(diff, count)

        # generate stochastic policy
        self.solve_policy_step()

        self.values  = np.frombuffer(self.values, dtype='float32')
        self.policy  = np.frombuffer(self.policy, dtype='float32').reshape(self.n_states,self.n_actions)        
        self.rewards = np.frombuffer(self.rewards, dtype='float32')

        return self.policy, self.values


    def solve_step(self):

        self.q_mat = mp.Array('f', np.zeros(self.n_states*self.n_actions), lock=False)
        
        processes = []
        queue     = mp.Queue()

        for state_ids in self.segmented_state_ids:
            p = mp.Process(target=self.solve_chunk, args=(state_ids,
                                                          self.mp_roadmap,
                                                          self.mp_T,
                                                          self.rewards,
                                                          self.gamma,
                                                          self.values,
                                                          self.q_mat,
                                                          self.n_actions,
                                                          queue))
            p.start()
            processes.append(p)

        max_residual = float("-inf")

        # update diff
        for p in processes:
            max_residual = max(queue.get(), max_residual)
            p.join()

        return max_residual

    @staticmethod
    def solve_chunk(segmented_state_ids, roadmap, T, rewards, gamma, values, q_mat,
                    n_actions, queue, states=None, env=None):

        diff = float("-inf")
        for s in segmented_state_ids:
            # get idx of near states
            ids   = roadmap[s*n_actions:(s+1)*n_actions]
            v     = np.array(values)[ids]
            ## v     = [values[i] for i in ids]
            ## v     = [rewards[i] + gamma*values[i] for i in ids]

            #max_v = float("-inf")
            for i in range(n_actions):
                q_mat[s*n_actions+i] = rewards[s] + gamma*np.dot(T[s*(n_actions*n_actions) + i*n_actions:\
                                                                   s*(n_actions*n_actions) + (i+1)*n_actions], v)
                ## max_v = max( max_v, rewards[s] + gamma*np.dot(T[s*(n_actions*n_actions) + i*n_actions:\
                ##                                                 s*(n_actions*n_actions) + (i+1)*n_actions], v))
                ## max_v = max( max_v, np.dot(T[s*(n_actions*n_actions) + i*n_actions:\
                ##                              s*(n_actions*n_actions) + (i+1)*n_actions], v))

            max_v = np.amax(q_mat[s*n_actions:(s+1)*n_actions])
            ## max_v += rewards[s]
            ## max_v = np.amax( np.dot(T[s], v))
            ## max_v = np.amax( np.dot(T, v))
            ## max_v = np.amax( np.dot(T, rewards[ids] + gamma*values[ids]) )

            new_diff = abs(values[s] - max_v)
            if new_diff > diff:                    
                diff = new_diff
            values[s] = max_v

        queue.put(diff)


    def solve_policy_step(self):

        self.policy = mp.Array('f', np.zeros(self.n_states*self.n_actions), lock=False)
        
        processes = []
        for state_ids in self.segmented_state_ids:
            p = mp.Process(target=self.solve_policy_chunk, args=(state_ids,
                                                                 self.policy,
                                                                 self.mp_roadmap,
                                                                 self.mp_T,
                                                                 self.rewards,
                                                                 self.gamma,
                                                                 self.values,
                                                                 self.n_actions))
            p.start()
            processes.append(p)

        max_residual = 0

        # update
        for p in processes:
            p.join()

        self.policy  = np.frombuffer(self.policy, dtype='float32').reshape(self.n_states,self.n_actions)
        self.policy -= self.policy.max(axis=1).reshape((self.n_states, 1))  # For numerical stability.

        # Boltzmann
        # Note: Does this better than e-greedy or greedy?
        self.policy  = np.exp(self.policy)/np.exp(self.policy).sum(axis=1).reshape((self.n_states, 1))
        return self.policy


    @staticmethod
    def solve_policy_chunk(segmented_state_ids, policy, roadmap, T, rewards, gamma, values, n_actions):

        for s in segmented_state_ids:
            s = int(s)
            
            # get idx of near states
            ids = roadmap[s*n_actions:(s+1)*n_actions]
            v  = [values[i] for i in ids]
            ## v  = [rewards[i] + gamma*values[i] for i in ids]

            for a in range(n_actions):
                policy[s*n_actions+a] = rewards[s] + gamma*np.dot(T[s*(n_actions*n_actions)+a*n_actions:\
                                                                    s*(n_actions*n_actions)+(a+1)*n_actions], v)
                ## policy[s*n_actions+a] = np.dot(T[s*(n_actions*n_actions)+a*n_actions:\
                ##                                  s*(n_actions*n_actions)+(a+1)*n_actions], v)
                ## policy[s*n_actions+a] = np.dot(T[a], v)
                ## policy[s*n_actions+a] = np.dot(T[a], rewards[ids] + gamma*values[ids])
            

    def get_policy(self):
        return self.policy

    def get_rewards(self):
        return self.rewards

    def get_values(self):
        return self.values

    def set_rewards(self, rewards):
        self.rewards = rewards

    def act(self, state, env=None, e=0.):
        if type(state) is list: state = np.array(state)
        if env is None:
            s_idx = np.argmin(np.linalg.norm(self.states-state, axis=-1))
        else:            
            s_idx, state = dp.get_close_state_idx(state, env, self.states, self.poses, return_state=True)

        if np.random.binomial(1, e) == 1:
            best_action = np.random.choice(range(self.n_actions))
        else:
            values      = self.policy[s_idx]
            best_action = np.random.choice([i for i, value in enumerate(values) if value == np.max(values)])
            ## best_action = np.argmax(values)
        
        # get nearstates
        s_idx     = self.roadmap[s_idx][best_action]        
        nxt_state = np.array(self.states[s_idx])
        
        return nxt_state-state


    def reset(self):
        self.action_list = range(self.n_actions)
        ## if self.rewards is not None:
        ##     self.rewards = mp.Array('f', self.rewards, lock=False)
        if self.roadmap is not None:
            self.mp_roadmap = mp.Array('i', np.reshape(self.roadmap, (self.n_states*self.n_actions)), lock=False)
        if self.T is not None:
            self.mp_T = mp.Array('f', np.reshape(self.T, (self.n_states*self.n_actions*self.n_actions)), lock=False)
        self.segmented_state_ids = segment_state_indices(self.n_states, self.n_processes)

    def save(self, filename):
        
        d = {'policy': self.policy, 'rewards': self.rewards, 'roadmap': self.roadmap,\
             'T': self.T, "n_actions": self.n_actions, "n_states": self.n_states,
             'states': self.states, 'values': self.values, 'theta': self.theta,
             'poses': self.poses}
        pickle.dump( d, open( filename, "wb" ) )
        pickle.dump( self.skdtree, open( filename+'_kdtree', "wb" ) )
        if self.verbose: print "Saved a file: ", filename

    def load(self, filename):
        d = pickle.load( open( filename, "rb" ) )
        self.policy  = d['policy']
        self.values  = d['values']
        self.rewards = d['rewards']
        self.roadmap = d['roadmap']
        self.T       = d['T']
        self.n_actions = d['n_actions']
        self.n_states  = d['n_states']
        self.states  = d['states']
        self.poses   = d['poses']
        self.theta   = d['theta']
        self.skdtree = pickle.load( open( filename+'_kdtree', "rb") )
        self.reset()

        if type(self.states) is not list: self.states = self.states.tolist()
        print "Loaded a file: ", filename



def segment_state_indices(num_states, num_processes):
    # segments the state indices into chunks to distribute between processes
    state_idxs = np.arange(num_states)
    num_uneven_states = num_states % num_processes
    if num_uneven_states == 0:
        segmented_state_idxs = state_idxs.reshape(num_processes, -1)
    else:
        segmented_state_idxs = state_idxs[:num_states - num_uneven_states].reshape(num_processes, -1).tolist()
        segmented_state_idxs[-1] = np.hstack((segmented_state_idxs[-1], state_idxs[-num_uneven_states:])).tolist()
        
    return segmented_state_idxs

