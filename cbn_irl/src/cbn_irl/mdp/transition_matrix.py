import numpy as np
import multiprocessing as mp

import sys

def create_transition_matrix(roadmap, states, vel_limits, action_prob=0.85, stay_enabled=True):

    n_states  = len(states)
    n_actions = len(roadmap[0])
    n_dim     = len(states[0])
    states    = np.array(states)
    segmented_state_ids = segment_state_indices(n_states, num_processes=mp.cpu_count())
    
    mp_roadmap = mp.Array('i', np.reshape(roadmap, (n_states*n_actions)), lock=False)
   
    T = np.ones(n_states*n_actions*n_actions)*(1.-action_prob)/float(n_actions-1)        
    T = mp.Array('f', T, lock=False)

    processes = []    
    for state_ids in segmented_state_ids:
        p = mp.Process(target=get_T_chunk, args=(state_ids,
                                                 mp_roadmap,
                                                 states,
                                                 T,
                                                 n_actions, 
                                                 vel_limits,
                                                 action_prob,
                                                 n_dim,
                                                 stay_enabled))
        p.start()
        processes.append(p)

    # update diff
    for p in processes:
        p.join()

    return np.frombuffer(T, dtype='float32').reshape(n_states,n_actions,n_actions)


def get_T_chunk(state_ids, roadmap, states, T, n_actions, vel_limits, action_prob,
                n_dim, stay_enabled):

    for s in state_ids:

        ids   = roadmap[s*n_actions:(s+1)*n_actions]

        for a_idx in range(n_actions):
            # compute action
            s_idx = ids[a_idx]
            vel   = states[s_idx] - states[s]

            T[s*(n_actions*n_actions) + a_idx*n_actions + a_idx] = action_prob
            ## if any(vel[j] < vel_limits[0][j] for j in range(n_dim)) or\
            ##   any(vel[j] > vel_limits[1][j] for j in range(n_dim)):
            ## if (vel < vel_limits[0][0]).any() or (vel > vel_limits[1][0]).any() or s_idx == s: 
            ##     for k in range(n_actions):
            ##         T[s*(n_actions*n_actions) + k*n_actions + a_idx] = 0.

        for a_idx in range(n_actions):
            if stay_enabled is False:
                T[s*(n_actions*n_actions) + a_idx*n_actions]=0.
            t_sum = sum(T[s*(n_actions*n_actions) + a_idx*n_actions:
                          s*(n_actions*n_actions) + (a_idx+1)*n_actions])

            if t_sum > 0.:
                for k in range(n_actions):
                    T[s*(n_actions*n_actions) + a_idx*n_actions + k] /= t_sum                    
            ## else:
            ##     for as_idx in range(n_actions):
            ##         T[s*(n_actions*n_actions) + a_idx*n_actions + as_idx] = 1./float(n_actions)



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


def create_transition_matrix2(n_states, n_actions):
    action_prob = 0.8
    T = np.ones((n_states, n_actions, n_actions))*(1.-action_prob)/float(n_actions-1)

    for s in xrange(n_states):
        for i in range(n_actions):
            T[s,i,i] = action_prob
    return T






def create_undirect_transition_matrix(roadmap, states, vel_limits, action_prob=0.85,
                                      stay_enabled=True):

    n_states  = len(states)
    n_actions = len(roadmap[0])
    n_dim     = len(states[0])
    states    = np.array(states)
    segmented_state_ids = segment_state_indices(n_states, num_processes=mp.cpu_count())

    mp_roadmap = mp.Array('i', np.reshape(roadmap, (n_states*n_actions)), lock=False)
   
    ## T = np.ones(n_states*n_actions*n_actions)*(1.-action_prob)/float(n_actions-1)        
    T = np.zeros(n_states*n_actions*n_actions)
    T = mp.Array('f', T, lock=False)

    processes = []    
    for state_ids in segmented_state_ids:
        p = mp.Process(target=get_undirect_T_chunk, args=(state_ids,
                                                          mp_roadmap,
                                                          states,
                                                          T,
                                                          n_actions, 
                                                          vel_limits,
                                                          action_prob,
                                                          n_dim,
                                                          stay_enabled))
        p.start()
        processes.append(p)

    # update diff
    for p in processes:
        p.join()

    return np.frombuffer(T, dtype='float32').reshape(n_states,n_actions,n_actions)


def get_undirect_T_chunk(state_ids, roadmap, states, T, n_actions, vel_limits, action_prob,
                         n_dim, stay_enabled):

    for s in state_ids:

        ids = roadmap[s*n_actions:(s+1)*n_actions]

        for a_idx in range(n_actions):           
            # compute action
            s_idx = ids[a_idx]
            vel   = states[s_idx] - states[s]

            if stay_enabled and a_idx>0 and s_idx == ids[0]:
                # disenable repeated actions
                ## T[s*(n_actions*n_actions) + a_idx*n_actions:
                ##   s*(n_actions*n_actions) + (a_idx+1)*n_actions] *= 0.
                break
            else:
                T[s*(n_actions*n_actions) + a_idx*n_actions + a_idx] = action_prob

        for a_idx in range(n_actions):
            if stay_enabled is False:
                T[s*(n_actions*n_actions) + a_idx*n_actions]=0.
            else:
                if s == roadmap[s*n_actions+a_idx]:
                    T[s*(n_actions*n_actions) + a_idx*n_actions]=0.                    

            ## for a2_idx in range(1, n_actions):
            ##     if ids[0]==ids[a2_idx]:
            ##         T[s*(n_actions*n_actions) + a_idx*n_actions+a2_idx:
            ##           s*(n_actions*n_actions) + (a_idx+1)*n_actions] = 0.
                                                    
            t_sum = sum(T[s*(n_actions*n_actions) + a_idx*n_actions:
                          s*(n_actions*n_actions) + (a_idx+1)*n_actions])

            if t_sum > 0.:
                for k in range(n_actions):
                    T[s*(n_actions*n_actions) + a_idx*n_actions + k] /= t_sum                    
            ## else:
            ##     for as_idx in range(n_actions):
            ##         T[s*(n_actions*n_actions) + a_idx*n_actions + as_idx] = 1./float(n_actions)



