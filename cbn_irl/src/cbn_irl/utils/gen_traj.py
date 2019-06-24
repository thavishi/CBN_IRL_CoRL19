import gym
import numpy as np
from scipy import interpolate
import copy
from cbn_irl.mdp import value_iteration as vi
import quaternion as qt

def generate_trajectories(env, agent, 
                          n_trajs  = 100,
                          len_traj = 50,
                          rnd_start= False,
                          epsilon  = 0.):
    trajs = []
    while len(trajs) < n_trajs:

        if rnd_start:
            # override start_pos
            print "not implemented"
            
        state, time = env.reset(), 0
        state = np.array(state)
        traj =[state]

        reached = False
        for _ in range(len_traj-1):
            a = agent.act(state, e=epsilon)
            next_state, r, done, _ = env.step(a)
            traj.append(state)

            if done:
                reached = True
                break
            state = next_state

        # exit loop after filling empty 
        if reached:
            while len(traj)<len_traj:
                traj.append(next_state)
                        
            trajs.append(traj)
        print "Generating {}th trajectories".format(len(trajs))
        
    print "{} episodes was successfull, Agent reached the goal".format(len(trajs))
    return trajs


def generate_segments(env, n_trajs, len_traj=None, enable_interp=True):
    """Generate mouse click data"""
    
    import cursor_demo as cd
    import matplotlib.pyplot as plt

    ax = env.ax
    
    cursor = cd.Cursor(ax)
    cid = plt.connect('button_press_event', cursor.mouse_move)
    plt.show()
    plt.disconnect(cid)
    trajs = cursor.trajs

    print "{} episodes was successfull, Agent reached the goal".format(len(trajs))
    if enable_interp is False: return trajs
        
    # interpolation with specific length
    new_trajs = []
    for traj in trajs:
        times     = np.linspace(0., 1., len(traj))
        new_times = np.linspace(0., 1., len_traj)    
        new_trajs.append(interpolationData(times, traj, new_times))
            
    return new_trajs


def collect_mouse_move(env, n_trajs, enable_interp=False):
    
    import cursor_demo as cd
    import matplotlib.pyplot as plt
    
    cursor = cd.Cursor(env.ax, env.fig)
    #cid = plt.connect('motion_notify_event', cursor.mouse_move)
    env.fig.canvas.mpl_connect("motion_notify_event", cursor.mouse_move)    
    plt.show()
    #plt.disconnect(cid)
    trajs = cursor.trajs
    print np.shape(trajs)
    

    print "{} episodes was successfull, Agent reached the goal".format(len(trajs))
    if enable_interp is False: return trajs
        
    # interpolation with specific length
    new_trajs = []
    for traj in trajs:
        times     = np.linspace(0., 1., len(traj))
        new_times = np.linspace(0., 1., len_traj)    
        new_trajs.append(interpolationData(times, traj, new_times))
            
    return new_trajs
    
    
def interpolationData(time_array, traj, new_time_array, enable_spline=False):
    """Interpolate a multidimensional trajectory"""
    traj = np.array(traj).T
    new_traj = []
    for x in traj:

        if enable_spline:
            ## interp_data = interpolate.spline(time_array, x, new_time_array)
            interp = interpolate.splrep(time_array, x, s=0)
            interp_data = interpolate.splev(new_time_array, interp, der=0, ext=1)
        else:
            interp = interpolate.interp1d(time_array, x)
            interp_data = interp(new_time_array)

        if np.isnan(np.max(interp_data)):
            print "Interpolation error by NaN values"
            print "New start time = ", new_time_array[0], " start time = ", time_array[0]
            print "New end time = ", new_time_array[-1], " end time = ", time_array[-1]
            sys.exit()

        new_traj.append(interp_data)

    return np.array(new_traj).T.tolist()


def convertRawTraj2RoadmapTraj(env, trajs, roadmap, states, poses=None, len_traj=None,
                               gamma=10., jnt_to_pos=False, **kwargs):
    """ """
    from cbn_irl.path_planning import dijkstra_planning as dp
    if len(np.shape(trajs))==2: trajs = [trajs]

    distFunc = kwargs.get('distFunc', None)        
        
    new_trajs = []
    idx_trajs = []
    for traj in trajs:
        start = traj[0]
        goal  = traj[-1]
        new_traj, rids = dp.modified_dijkstra_planning(env, start, goal, roadmap,
                                                       states, poses, traj,
                                                       gamma=gamma, distFunc=distFunc,
                                                       verbose=True)
        if new_traj is None:
            print "No new traj"
            continue
        print "dijkstra algorithm: ", np.shape(new_traj)
        
        if len_traj is not None and len(new_traj) > len_traj:
            print "Too long, so resample it"
            sys.exit()

        assert len(new_traj)>0, "traj length is zero. No match node on the roadmap"

        if len_traj is not None:
            while len(new_traj) < len_traj:
                new_traj.append(new_traj[-1])
                rids.append(rids[-1])
        new_trajs.append(new_traj)
        idx_trajs.append(rids)
    
    return new_trajs, idx_trajs


def data_augmentation(env, data_dict, multiplier=3):
    """Augment data"""

    for key in data_dict.keys():
        length = len(data_dict[key])

        for m in range(multiplier):
            for i in range(length):
                noise = np.random.normal(0,1.,(len(data_dict[key][i]),2))
                noise[0] *= 0.
                noise[-1] *= 0.
                new_traj = data_dict[key][i] + noise
                for j, s in enumerate(new_traj):
                    if not env.isValid(s):
                        new_traj[j] = data_dict[key][i][j]
                data_dict[key].append(new_traj)        
    return data_dict

def traj_augmentation(env, trajs, multiplier=3, std=1.):
    """Augment data"""

    nsamples = len(trajs)
    dim    = len(trajs[0][0])

    new_trajs = []
    for m in range(multiplier):
        for i in range(nsamples):
            noise = np.random.normal(0,std,(len(trajs[i]),dim))
            noise[0] *= 0.
            noise[-1] *= 0.
            new_traj = trajs[i] + noise
            for j, s in enumerate(new_traj):
                if not env.isValid(s):
                    new_traj[j] = trajs[i][j]
            new_trajs.append(new_traj)        
    return new_trajs


def get_demonstration_data(start, goal, objs, env_name, n_trajs, len_traj=None, enable_interp=True):
    """Get 2D mouse click trajectories"""

    data_dict = {}
    for i in range(1):
        env = gym.make(env_name)
        env.__init__(start, goal, objs, viz=True)
        env.render()
        print "-------------------------------------------------------"
        print "label ", i
        print "-------------------------------------------------------"
        ## trajs = generate_segments(env, n_trajs=n_trajs, len_traj=len_traj,
        ##                           enable_interp=enable_interp)
        trajs = collect_mouse_move(env, n_trajs, enable_interp=enable_interp)
        assert len(trajs)>0, "No recorded trajectories"
        
        data_dict[i] = trajs 

    # add common goal
    for key in data_dict.keys():
        new_trajs = []
        for i, traj in enumerate(data_dict[key]):
            new_trajs.append(traj+[goal])
        data_dict[key] = new_trajs
    
    return data_dict


def get_synthetic_data(start, goal, objs, labels, env_name, n_actions=10, n_states=1000,
                       gamma=0.95, error=1e-4, tgt_label=[0], viz=False):
    """ synthetic data from new roadmap """
    
    env = gym.make(env_name)
    env.__init__(start, goal, objs, viz=viz)
    env.seed(0)

    roadmap, states, skdtree = prm.get_roadmap(start, goal,
                                               objs,
                                              env.robot_size,
                                              [env.observation_space.low[0],
                                               env.observation_space.high[0]],
                                              [env.observation_space.low[1],
                                               env.observation_space.high[1]],
                                              leafsize=n_actions,
                                              n_sample=n_states)

    reward_fn = reward.make_wall_reward(goal, objs, labels, tgt_label)
    rewards   = reward_fn(range(len(states)))
    agent = vi.valueIterAgent(n_actions, n_states, roadmap, skdtree, states, rewards, gamma)
    _, values = agent.find_policy(error)
    
    trajs = gt.generate_trajectories(env, agent, 
                                     n_trajs  = 100,
                                     len_traj = 200,
                                     ## rnd_start= False,
                                     epsilon  = 0.1)

    return trajs


def get_interpolation_data(poses, dist_func=None, len_traj=100, use_extra_info=False):
    """This provids a 7d (pos+quat) trajectory """
    ## assert len(poses)<len_traj, "interpolation will not work with the short target length"
    
    ndim  = len(poses[0])
    assert ndim>=7, "Need to convert rpy to quaternion"
    
    if type(poses) is list: poses = np.array(poses)

    # A list of distance
    L = []
    for i in range(1, len(poses)):
        if dist_func is None:
            dist = np.linalg.norm(poses[i][:3]-poses[i-1][:3])
        else:
            dist = dist_func(poses[i-1], poses[i])
        L.append(dist)

    # Scaling the list wrt desired length
    L_traj = []
    for i in range(len(L)):
        L_traj.append( max(int(len_traj*float(L[i])/float(sum(L))), 1) )

    # get linear trajectory
    traj = [poses[0]]
    for i in range(1, len(poses)):
        dx = (poses[i][:3]-poses[i-1][:3])/float(L_traj[i-1])

        if use_extra_info:
            dg = (poses[i][7:]-poses[i-1][7:])/float(L_traj[i-1])

        
        for j in range(L_traj[i-1]):
            point = copy.deepcopy(poses[i-1])
            point[:3] += dx*float(j+1)

            q = qt.slerp(poses[i-1][3:7], poses[i][3:7], float(j+1)/float(L_traj[i-1]))
            point[3] = q[0]
            point[4] = q[1]
            point[5] = q[2]
            point[6] = q[3]

            if use_extra_info:
                # gripper info only  #TEMP              
                point[7:] = poses[i-1][7:] + dg*float(j+1)                

            traj.append(point)
    return traj
