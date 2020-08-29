import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import traj_utils as tu
import planning_utils as pu

show_animation = False #True

class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, x_id, cost, pind):
        self.x    = x
        self.x_id = x_id
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.cost) + "," + str(self.pind)


def dijkstra_planning(env, start, goal, road_map, states, jnt_to_pos=False, verbose=False, **kwargs):
    """
    Original dijkstra algorithm
    
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    reso: grid resolution [m]
    rr: robot radius[m]
    demo_traj: a target trajectory
    gamma: distance weight for the target trajectory
    """
    ndim  = len(goal)
    openset, closedset = dict(), dict()
    distFunc = kwargs.get('distFunc', None)

    # Set start node: TODO jnt_to_pos
    start_idx = get_close_state_idx(start, env, states, distFunc=distFunc)
    nstart    = Node(states[start_idx], start_idx, 0., -1)
    openset[start_idx] = nstart

    goal_idx = get_close_state_idx(goal, env, states, distFunc=distFunc)
    ## assert start_idx!=goal_idx, "Start and goal are same on the graph."
    if start_idx==goal_idx: return [],[]
    if verbose: print "Start and goal indices: ", start_idx, goal_idx
    
    #from IPython import embed; embed()
    while True:
        if len(openset) == 0:
            if verbose: print("Cannot find path")
            return None, None
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]

        ## if c_id == (len(road_map) - 1):
        if c_id == goal_idx :
            if verbose: print("goal is found! {}, {}".format(c_id, current.pind))
            ngoal = Node(states[goal_idx], goal_idx, current.cost, current.pind)
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            if n_id == c_id: continue  # need?

            next_state = states[n_id]

            if distFunc is not None:
                d1 = distFunc(current.x, np.array(next_state))  
            elif jnt_to_pos:            
                # find distance from next state
                w  = np.array([3.,3.,2.,2.,1.5,1.,1.]); w /= np.sum(w)            
                d1 = np.linalg.norm((next_state - current.x)*w) #, ord=np.inf)            
            else:
                d1 = np.linalg.norm(np.array(next_state) - current.x) #, ord=np.inf)            
            node = Node(next_state, n_id, current.cost + d1, c_id)
                
            if n_id in closedset:
                continue
            if env.isValid(next_state, check_collision=True) is False:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node

    # generate final course
    rx   = [ngoal.x]
    rids = [ngoal.x_id]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        rids.append(n.x_id)
        pind = n.pind

    return rx[::-1], rids[::-1]


def modified_dijkstra_planning(env, start, goal, road_map, states, poses, demo_traj,
                               gamma=1.,
                               jnt_to_pos=False, verbose=False, **kwargs):
    """
    Modified version of dijkstra algorithm with fitting cost
    
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    reso: grid resolution [m]
    rr: robot radius[m]
    demo_traj: a target trajectory
    gamma: distance weight for the target trajectory
    """
    ndim  = len(goal)

    openset, closedset = dict(), dict()
    distFunc = kwargs.get('distFunc', None)

    # Set start node: TODO jnt_to_pos
    start_idx = get_close_state_idx(start, env, states, poses, distFunc=distFunc)
    nstart    = Node(states[start_idx], start_idx, 0., -1)
    openset[start_idx] = nstart

    demo_idx = 0
    if not(distFunc is not None and ndim==7):
        demo_traj = tu.interp_over_progress(demo_traj,
                                            np.linspace(0,1,len(demo_traj),
                                                        endpoint=True),
                                                        length=100) 
    if type(demo_traj) is list: demo_traj = np.array(demo_traj)

    if jnt_to_pos:
        demo_pose_traj = env.get_poses(demo_traj)

    goal_idx = get_close_state_idx(goal, env, states, poses, distFunc=distFunc)
    assert start_idx!=goal_idx, "Start and goal are same on the graph."
    if verbose:
        print "Start and goal indices: ", start_idx, goal_idx
    
    #from IPython import embed; embed()
    while True:
        if len(openset) == 0:
            if verbose: print("Cannot find path")
            return None, None
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]

        if c_id == goal_idx :
            if verbose: print("goal is found! {}, {}".format(c_id, current.pind))
            ngoal = Node(states[goal_idx], goal_idx, current.cost, current.pind)
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # find the closest point's index in the demo traj ()
        if jnt_to_pos:
            idx_offset = get_close_state_idx(current.x, env,
                                             demo_traj[demo_idx:],
                                             demo_pose_traj[demo_idx:])
        else:
            idx_offset = get_close_state_idx(current.x, env,
                                             demo_traj[demo_idx:],
                                             distFunc=distFunc)
        demo_idx += idx_offset

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            if n_id == c_id: continue  # need?

            next_state = states[n_id]

            if distFunc is not None:
                d1 = distFunc(current.x, np.array(next_state))  
                d2 = get_close_state_idx(next_state, env, demo_traj[demo_idx:],
                                         return_cost=True,
                                         distFunc=distFunc)
                #node = Node(next_state, n_id, current.cost + d1 + gamma*d2, c_id)
                node = Node(next_state, n_id, current.cost + d1*0.01 + gamma*d2, c_id)
            elif jnt_to_pos:            
                # find distance from next state
                w  = np.array([3.,3.,2.,2.,1.5,1.,1.]); w /= np.sum(w)            
                d1 = np.linalg.norm((next_state - current.x)*w) #, ord=np.inf)            
                # find distance from demo traj
                d2 = get_close_state_idx(next_state, env, demo_traj[demo_idx:],
                                         demo_pose_traj[demo_idx:], return_cost=True)
                node = Node(next_state, n_id, current.cost*0.5 + d1*0.01 + gamma*d2, c_id)
            else:
                d1 = np.linalg.norm(next_state - current.x) #, ord=np.inf)            
                d2 = get_close_state_idx(next_state, env, demo_traj[demo_idx:], return_cost=True)
                node = Node(next_state, n_id, current.cost*0.5 + d1*0.01 + gamma*d2, c_id)

            if env.isValid(next_state, check_collision=True) is False:
                continue

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node


    # generate final course
    rx   = [ngoal.x]
    rids = [ngoal.x_id]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        rids.append(n.x_id)
        pind = n.pind

    ## # remove zero? node
    ## rx = rx[:-1]
    ## rids = rids[:-1]

    return rx[::-1], rids[::-1]


def get_close_state_idx(state, env, states, poses=None, w=None, jnt_ratio=0.0, return_state=False,
                        return_cost=False, distFunc=None):
    if w is None:
        w  = np.array([5.,4.,3.,2.,1.5,1.,1.]); w /= np.sum(w)            
        #w = np.array([10.,10.,6.,5.,4.,2.,1.]); w /= np.sum(w)
    if type(states) is list: states = np.array(states)
    if poses is not None and type(poses) is list: poses = np.array(poses)

    if return_cost:
        if poses is None:
            if distFunc is not None:
                return np.min(distFunc(states, state))
            else:
                return np.min(np.linalg.norm(states-state, axis=1))
        else:            
            return np.min(np.linalg.norm((states-state)*w.reshape(1,-1), axis=1)*jnt_ratio+
                          (np.linalg.norm(poses[:,:3]-np.array(env.get_pos(state)), axis=1)*0.9+
                           pu.quat_dist(poses[:,3:], np.array(env.get_quat(state)))*0.1
                          )*(1.-jnt_ratio))
    else:
        if poses is None:
            if distFunc is not None:
                s_idx = np.argmin(distFunc(states, state))
            else:          
                s_idx = np.argmin(np.linalg.norm(states-state, axis=1))
        else:            
            s_idx = np.argmin(np.linalg.norm((states-state)*w.reshape(1,-1), axis=1)*jnt_ratio+
                          (np.linalg.norm(poses[:,:3]-np.array(env.get_pos(state)), axis=1)*0.9+
                           pu.quat_dist(poses[:,3:], np.array(env.get_quat(state)))*0.1
                           )*(1.-jnt_ratio)  )
        if return_state:
            return s_idx, states[s_idx]
        else:
            return s_idx

