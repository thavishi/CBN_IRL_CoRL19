import copy
import numpy as np
import math

# ROS & Public library
import tf.transformations as tft
from geometry_msgs.msg import Quaternion


# quat_mean: a quaternion(xyzw) that is the center of gaussian distribution
# n:
# stdDev: a vector (4x1) that describes the standard deviations of the distribution
#         along axis(xyz) and angle
# Return n numbers of QuTem quaternions (gaussian distribution).
def quat_QuTem( quat_mean, n, stdDev ):

    # Gaussian random quaternion
    x = (np.array([np.random.normal(0., 1., n)]).T *stdDev[0]*stdDev[0])
    y = (np.array([np.random.normal(0., 1., n)]).T *stdDev[1]*stdDev[1])
    z = (np.array([np.random.normal(0., 1., n)]).T *stdDev[2]*stdDev[2])
    
    mag = np.zeros((n,1))
    for i in xrange(len(x)):
        mag[i,0] = np.sqrt([x[i,0]**2+y[i,0]**2+z[i,0]**2])
        
    axis  = np.hstack([x/mag, y/mag, z/mag])
    ## angle = np.array([np.random.normal(0., stdDev[3]**2.0, n)]).T
    angle = np.zeros([len(x),1])
    for i in xrange(len(x)):
        rnd = 0.0
        while True:
            rnd = np.random.normal(0.,1.)
            if rnd <= np.pi and rnd > -np.pi:
                break
        angle[i,0] = rnd + np.pi
                    
    # Convert the gaussian axis and angle distribution to unit quaternion distribution
    # angle should be limited to positive range...
    s = np.sin(angle / 2.0);
    quat_rnd = np.hstack([axis*s, np.cos(angle/2.0)])
                    
    # Multiplication with mean quat
    q = np.zeros((n,4))
    for i in xrange(len(x)):
        q[i,:] = tft.quaternion_multiply(quat_mean, quat_rnd[i,:])
                        
    return q
                    
# quat_dist: an angular distance between two quaternion
# q1: KDL Quaternion
## def quat_dist(q1, q2):
##     qd = q1.Inverse()*q2
##     l = np.sqrt( qd.x*qd.x + qd.y*qd.y + qd.z*qd.z)    
##     return 2 * atan2(l, qd.w)
    
# quat_dist: an angular distance between two quaternion
# q1: KDL Quaternion
def quat_dist(q1, q2):

    if len(np.shape(q1))>1:
        x1 = -q1[:,0]; y1 = -q1[:,1]; z1 = -q1[:,2]; w1 = q1[:,3]
        [x2, y2, z2, w2] = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        l  = np.linalg.norm(np.array([x,y,z]).T, axis=-1)
        return 2 * np.arctan2(l, w)
    else:
        q1_inv = copy.copy(q1)
        q1_inv[:3] *= -1

        [x1, y1, z1, w1] = q1_inv
        [x2, y2, z2, w2] = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        qd = [x,y,z,w]
        l  = np.linalg.norm(np.array(qd)[:3], axis=-1)
        return 2 * math.atan2(l, w)
        

    ##     q1_inv * q2.reshape()
    
    ## qd = q1.Inverse()*q2
    ## l = np.sqrt( qd.x*qd.x + qd.y*qd.y + qd.z*qd.z)    
    ## return 2 * atan2(l, qd.w)
