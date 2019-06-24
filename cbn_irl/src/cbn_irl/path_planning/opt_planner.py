import os
import numpy as np
## from scipy.optimize import fmin_l_bfgs_b
import openravepy
import trajoptpy

import json
import time
import rospkg


def plan(env_xml, manip_name, start, goal, traj=None, c_fn=None, eq_fn=None, ineq_fn=None, n_steps=100,
         lower_limit=None, upper_limit=None):

    env = openravepy.Environment()
    env.StopSimulation()
    env.Load(env_xml)

    trajoptpy.SetInteractive(False)
    robot = env.GetRobots()[0] # get the first robot
    robot.SetDOFValues(start, robot.GetManipulators(manip_name)[0].GetArmIndices())

    ## if lower_limit is not None and upper_limit is not None:
    ##     jnts = robot.GetJoints(robot.GetManipulators(limb+'_arm')[0].GetArmIndices())
    ##     for i, jnt in enumerate(jnts):
    ##         ## from IPython import embed; embed(); sys.exit()        
    ##         jnt.SetLimits(lower_limit[i:i+1], upper_limit[i:i+1])

    if type(goal) is not list: goal = list(goal)
    if traj is None:
        init_type = 'straight_line'
        traj = goal
    else:
        init_type = 'given_traj'
        n_steps = len(traj)
    if type(traj) is not list: traj = traj.tolist()

    # cost and constraints
    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name, 
            "start_fixed" : True # i.e., DOF values at first timestep are fixed based on current robot state
        },
        "costs" : [
        {
            "type" : "joint_vel", # joint-space velocity cost
            "params": {"coeffs" : [1.]} # a list of length one is automatically expanded to a list of length n_dofs
        },
        ],        
        "constraints" : [
        {
            "type" : "joint", # joint-space target
            "params" : {"vals" : goal } # length of vals = # dofs of manip
        },
        ],
        "init_info" : {
            "type" : init_type, 
            "data" : traj,
            "endpoint" : goal
        }
    }

    # set qp
    s     = json.dumps(request)                # convert dictionary into json-formatted string
    prob  = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem
    n_dof = len(start)
    
    # Set cost and constraints
    if c_fn is not None:
        for t in xrange(1,n_steps):    
            prob.AddErrorCost(c_fn, [(t,j) for j in xrange(n_dof)], "ABS", "up%i"%t)

    ## # EQ: an equality constraint. `f(x) == 0` in a valid solution.
    ## if eq_fn is not None:
    ##     for t in xrange(1,n_steps):
    ##         prob.AddConstraint(eq_fn, [(t,j) for j in xrange(n_dof)], "EQ", "up%i"%t)
            
    # INEQ: an inequality constraint. `f(x) <= `0` in a valid solution.
    if ineq_fn is not None:
        for t in xrange(3,n_steps):
            if type(ineq_fn) is list:
                for i in range(len(ineq_fn)):
                    prob.AddConstraint(ineq_fn[i], [(t,j) for j in xrange(n_dof)],
                                       "INEQ", "up%i"%t)
            else:
                prob.AddConstraint(ineq_fn, [(t,j) for j in xrange(n_dof)],
                                   "INEQ", "up%i"%t)

    t_start = time.time()
    result = trajoptpy.OptimizeProblem(prob) # do optimization
    t_elapsed = time.time() - t_start
    print "optimization took %.3f seconds"%t_elapsed

    ## from trajoptpy.check_traj import traj_is_safe
    ## prob.SetRobotActiveDOFs() # set robot DOFs to DOFs in optimization problem
    ## assert traj_is_safe(result.GetTraj(), robot) # Check that trajectory is collision free

    ## if result.GetTraj() is None or len(result.GetTraj())==0:
    ##     from IPython import embed; embed() #; sys.exit()
    ## for cost in result.GetCosts():
    ##     if cost[0].find('joint_vel')<0 and cost[1]>1.0:
    ##         print "Optimization failed with high cost value {}".format(cost)
    ##         return []
    ## for cstr in result.GetConstraints():
    ##     if cstr[1]>0.001:
    ##         print "Optimization failed with high constraint value {}".format(cstr)
    ##         return []
    return result.GetTraj()




def check(env, traj):
    
    is_okay = True
    if len(traj)==0:
        is_okay = False
    else:
        for s in traj:
            if env.isValid(s) is False:
                is_okay = False
                break
    return is_okay
    


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    rospack      = rospkg.RosPack()
    work_path    = os.path.join(os.path.join(rospack.get_path('irl_constraints_learning')), 'src/irl_constraints_learning/path_planning')
    
    env = openravepy.Environment()
    env.StopSimulation()
    env.Load(os.path.join(work_path, "robots/baxter_structure.xml"))

    trajoptpy.SetInteractive(args.interactive) # pause every iteration, until you press 'p'. Press escape to disable further plotting
    #env.SetViewer('qtcoin') # start the viewer (conflicts with matplotlib)

    robot = env.GetRobots()[0] # get the first robot
    ## manip = robot.GetManipulators()[0] # left_arm
    print robot.GetManipulators('left_arm')[0].GetArmIndices()
    
    joint_start = [0,0,0,0,0,0,0]
    robot.SetDOFValues(joint_start, robot.GetManipulators('left_arm')[0].GetArmIndices())

    joint_target = [1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    request = {
        "basic_info" : {
            "n_steps" : 10,
            "manip" : "left_arm", # see below for valid values
            "start_fixed" : True # i.e., DOF values at first timestep are fixed based on current robot state
        },
        "costs" : [
        {
            "type" : "joint_vel", # joint-space velocity cost
            "params": {"coeffs" : [4]} # a list of length one is automatically expanded to a list of length n_dofs
            # also valid: [1.9, 2, 3, 4, 5, 5, 4, 3, 2, 1]
        }
        ],
        "constraints" : [
        {
            "type" : "joint", # joint-space target
            "params" : {"vals" : joint_target } # length of vals = # dofs of manip
        }
        ],
        "init_info" : {
            "type" : "straight_line", # straight line in joint space.
            "endpoint" : joint_target
        }
    }

    s = json.dumps(request) # convert dictionary into json-formatted string
    prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem

    print "--------------------------------------"
    t_start = time.time()
    result = trajoptpy.OptimizeProblem(prob) # do optimization
    t_elapsed = time.time() - t_start
    print result
    print "optimization took %.3f seconds"%t_elapsed

    from trajoptpy.check_traj import traj_is_safe
    prob.SetRobotActiveDOFs() # set robot DOFs to DOFs in optimization problem
    assert traj_is_safe(result.GetTraj(), robot) # Check that trajectory is collision free

    print result.GetTraj()
    print np.shape(result.GetTraj())

    
    s = raw_input("Hit Enter to request grasps ")
