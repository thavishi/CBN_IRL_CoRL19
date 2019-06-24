import numpy as np
import copy
from scipy import interpolate
from cbn_irl.utils import gen_traj as gt
from cbn_irl.viz import viz as v

def get_trans_interp_traj(start, goal, traj, std_traj=None, n_steps=100):
    # Get nonlinear part from a reference traj
    length = len(traj)
    traj -= traj[0,:]  # zero start
    traj /= traj[-1,:] # scale
    for i in range(len(traj[0])):
        traj[:,i] = traj[:,i]*(goal[i]-start[i])+start[i]

    times     = np.linspace(0., 1., len(traj))
    new_times = np.linspace(0., 1., n_steps)
    traj = gt.interpolationData(times, traj, new_times)
    if std_traj is not None:
        std_traj = gt.interpolationData(times, std_traj, new_times)

    assert len(traj)>1, "{} length of reference traj".format(len(traj))

    if std_traj is None:            
        return traj
    else:
        return traj, std_traj


def interp_over_progress(traj, progress, length=1000):
    new_progress = np.linspace(0,1.,length, endpoint=True)
    ## traj = gt.interpolationData(progress, traj, new_progress)

    new_traj = []
    for y in np.array(traj).T:
        new_traj.append( interpolate.spline(progress, y, new_progress))
    new_traj = np.array(new_traj).T
    return new_traj
    
## def get_trans_traj(traj, start, goal, n_steps):
##     # Get nonlinear part from a reference traj
##     init_traj = traj
##     init_traj -= init_traj[0,:]  # zero start
##     init_traj /= init_traj[-1,:] # scale
##     for i in range(len(init_traj[0])):
##         init_traj[:,i] = init_traj[:,i]*(goal[i]-start[i])+start[i]   
##     return init_traj


def get_straight_line(start, goal, n_steps=100):

    progress = np.linspace(0,1, n_steps)

    delta = goal-start

    traj = []
    for p in progress:
        traj.append( list(start + delta*p) )

    return traj



def fit_traj_to_graph(trajs, instrs, env, roadmap, states, poses, n=5,
                      length=100, perturbation=False, projection=True,
                      viz=False):
    
    # Convert the raw demonstrations to roadmap based demonstrations
    from path_augmentation import gen_perturbed_data as gpd
    org_trajs = copy.copy(trajs)
    new_trajs = {}; new_idx_trajs = {}; new_instrs = {}
    for key in trajs.keys():

        if perturbation:
            p = gpd.gen_perturbed_data(np.array(trajs[key][0]).T, n=n, dt=0.01)
            p_length = len(p[0])
            
            new_trajs[key]  = []
            new_instrs[key] = []
            for traj in trajs[key]:
                times     = np.linspace(0., 1., len(traj), endpoint=True)
                new_times = np.linspace(0., 1., p_length, endpoint=True)    
                new_trajs[key].append(gt.interpolationData(times, traj, new_times))
            new_instrs[key] = instrs[key]
            
            for traj, instr in zip(trajs[key], instrs[key]):
                p = gpd.gen_perturbed_data(np.array(traj).T, n=n, dt=0.01)
                new_trajs[key]   = np.vstack([new_trajs[key], p])
                new_instrs[key] += [instr]*n
        else:
            new_trajs = trajs
            new_instrs= instrs

        if projection is False:
            new_idx_trajs[key] = None
        elif viz is False:
            new_trajs[key], new_idx_trajs[key] = gt.convertRawTraj2RoadmapTraj(env, new_trajs[key],
                                                                               roadmap,
                                                                               states, poses,
                                                                               len_traj=length,
                                                                               jnt_to_pos=True,
                                                                               gamma=1.)
        else:
            key = "over_level"
            for i in range(len(trajs[key])):
                if i==0: continue
                if i>1 and i<4: continue
                new_traj, new_idx_traj = gt.convertRawTraj2RoadmapTraj(env, new_trajs[key][i:i+1],
                                                                       roadmap, states, poses,
                                                                       len_traj=length,
                                                                       jnt_to_pos=True,
                                                                       gamma=1.)

                v.traj_objs_plot(new_traj[0], env=env, org_traj=trajs[key][i])
                
    return new_trajs, new_idx_trajs, new_instrs
