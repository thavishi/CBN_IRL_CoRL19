import numpy as np
import multiprocessing as mp

def get_feature_over_states_no_multi(feature_fn, env, states, instr=None):

    n_feat = np.shape(feature_fn(env, env.reset(), instr))[-1]
    n_states  = len(states)
    feat_map = np.zeros((n_states, n_feat))
    
    for i, state in enumerate(states):
        feat_map[i] = feature_fn(env, state, instr)[0]
    return feat_map


def get_features_from_states(env, states, feature_fn):
    import sharedmem    
    n_states  = len(states)
    feat_len  = len(feature_fn(env, states[0]))
    state_ids = np.arange(n_states, dtype='i')

    features = sharedmem.full((n_states, feat_len), 0.)

    # Start multi processing over support states
    with sharedmem.MapReduce() as pool:
        if n_states % sharedmem.cpu_count() == 0:
            chunksize = n_states / sharedmem.cpu_count()
        else:
            chunksize = n_states / sharedmem.cpu_count() + 1

        def work(i):
            s_ids = state_ids[slice (i, i + chunksize)]
            for j, s_id in enumerate(s_ids):
                s = states[s_id] # state id in states
                features[s_id] = feature_fn(env, s)
            
        pool.map(work, range(0, n_states, chunksize))#, reduce=reduce)
    return np.array(features)


def get_feature_over_states(feature_fn, env, states, instr=None):

    n_feat = np.shape(feature_fn(env, env.reset(), instr=instr))[-1]
    n_states  = len(states)
    segmented_state_ids = segment_state_indices(n_states, num_processes=mp.cpu_count())

    feat_map = np.ones(n_states*n_feat)
    feat_map = mp.Array('f', feat_map, lock=False)

    processes = []    
    for state_ids in segmented_state_ids:
        p = mp.Process(target=get_feat_chunk, args=(state_ids,
                                                    states,
                                                    feature_fn,
                                                    env,
                                                    feat_map,
                                                    n_feat,
                                                    instr))
        p.start()
        processes.append(p)

    # update diff
    for p in processes:
        p.join()
    
    return np.frombuffer(feat_map, dtype='float32').reshape(n_states,n_feat)


def get_feature_over_state_actions(feature_fn, env, states, roadmap, instr=None):

    n_feat = np.shape(feature_fn(env, env.reset(), action=[0,0], instr=instr))[-1]
    n_states  = len(states)
    n_actions = len(np.shape(roadmap))
    segmented_state_ids = segment_state_indices(n_states*n_actions, num_processes=mp.cpu_count())

    feat_map = np.ones(n_states*n_actions*n_feat)
    feat_map = mp.Array('f', feat_map, lock=False)

    processes = []    
    for state_ids in segmented_state_ids:
        p = mp.Process(target=get_feat_chunk, args=(state_ids,
                                                    states,
                                                    feature_fn,
                                                    env,
                                                    feat_map,
                                                    n_feat,
                                                    instr))
        p.start()
        processes.append(p)

    # update diff
    for p in processes:
        p.join()
    
    return np.frombuffer(feat_map, dtype='float32').reshape(n_states,n_feat)


def get_feat_chunk(state_ids, states, feature_fn, env, feat_map, n_feat, instr=None):
    for s in state_ids:
        feat_map[s*n_feat:(s+1)*n_feat] = feature_fn(env, states[s], instr)

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


def save_features(env, trajs, instrs, feature_fn, filename='features.pkl'):

    fss_keys = {}
    for key in trajs.keys():
        fss = []
        for i, (traj, instr) in enumerate(zip(trajs[key], instrs[key])):
            env.set_start_state(traj[0])
            env.set_goal_state(traj[-1])            

            fs = []
            for s in traj:
                f = feature_fn(env, s, instr[0]).tolist()
                fs.append(f)
            print 

            fss.append(fs)
        fss_keys[key] = fss
    
    pickle.dump( fss_keys, open( filename, "wb" ) )
    print "Created feature file"
