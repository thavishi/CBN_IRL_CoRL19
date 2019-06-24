"""

Probablistic Road Map (PRM) Planner

author: Atsushi Sakai (@Atsushi_twi)

"""

import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

# parameter
#N_SAMPLE = 500  # number of sample_points
## N_KNN = 10  # number of edge from one sampled point
MAX_EDGE_LEN = 10.0  # [m] Maximum edge length

show_animation = False #True


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)


class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data, leafsize=10):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data, leafsize=leafsize)

    def search(self, inp, k=1):
        u"""
        Search NN

        inp: input data, single frame or multi frame

        """
        if type(inp) is list: inp = np.array(inp)

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist
        else:
            dist, index = self.tree.query(inp, k=k)
            return index, dist

    def search_in_distance(self, inp, r):
        u"""
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index

def get_roadmap(start, goal, obstacles, rr, xlim=None, ylim=None, knn=5, n_sample=500):
    u"""Generate roadmap"""
    
    ## obkdtree = KDTree(np.vstack((ox, oy)).T)
    obkdtree = KDTree(obstacles)

    if xlim is None:
        xlim = [min(obstacles[:,0]), max(obstacles[:,0])]
        ylim = [min(obstacles[:,1]), max(obstacles[:,1])]

    sample_x, sample_y = sample_points(start[0], start[1], goal[0], goal[1],
                                       rr, obstacles[:,0], obstacles[:,1],
                                       obkdtree, xlim, ylim, n_sample=n_sample)

    road_map, skdtree = generate_roadmap(sample_x, sample_y, rr, obkdtree,
                                         obstacles, knn=knn)

    return road_map, np.array([sample_x, sample_y]).T, skdtree


def PRM_planning(start, goal, obstacles, rr, xlim=None, ylim=None, cost_fn=None,
                 knn=5, n_sample=500, cstr_fn=None):

    road_map, samples, skd_tree = get_roadmap(start, goal, obstacles, rr, xlim, ylim,
                                              knn=knn, n_sample=n_sample)
    
    if show_animation:
        plt.plot(samples[:,0], samples[:,1], ".b")


    rx, ry = dijkstra_planning(
        start[0], start[1], goal[0], goal[1], obstacles[:,0], obstacles[:,1], rr,
        road_map, samples[:,0], samples[:,1], cost_fn=cost_fn, cstr_fn=cstr_fn)

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, okdtree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.sqrt(dx**2 + dy**2)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    nstep = round(d / D)

    for i in range(int(nstep)):
        idxs, dist = okdtree.search(np.matrix([x, y]).T)
        if dist[0] <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    idxs, dist = okdtree.search(np.matrix([gx, gy]).T)
    if dist[0] <= rr:
        return True  # collision

    return False  # OK


## def is_connected(sx, sy, nx, ny, okdtree, ob_states):

##     if len(ob_states) < 2: return True
##     sys.exit()

##     va = objs_in_tree[0][ids[0]] - state
##     vb = objs_in_tree[1][ids[1]] - state
##     direction = np.dot(va,vb)

 
##     # goal point check
##     idxs, dist = okdtree.search(np.matrix([nx, ny]).T, 2)

##     va = [sx-nx,sy-ny]
##     vb = []

##     return False  # OK



def generate_roadmap(sample_x, sample_y, rr, obkdtree, ob_states, knn, leafsize=50):
    """
    Road map generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    rr: Robot Radius[m]
    obkdtree: KDTree object of obstacles
    """

    road_map = []
    nsample = len(sample_x)
    skdtree = KDTree(np.vstack((sample_x, sample_y)).T, leafsize=leafsize)

    for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):

        index, dists = skdtree.search(
            np.matrix([ix, iy]).T, k=leafsize)
        inds = index[0][0]
        edge_id = []
        #  print(index)

        for ii in range(1, len(inds)):
            nx = sample_x[inds[ii]]
            ny = sample_y[inds[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obkdtree):
                ## and\
                ## not is_connected(ix, iy, nx, ny, obkdtree, ob_states):
                edge_id.append(inds[ii])

            if len(edge_id) >= knn:
                break

        assert len(edge_id)==knn, "number of edge is {}  < {} ".format(len(edge_id), knn)
        road_map.append(edge_id)

    ## plot_road_map(road_map, sample_x, sample_y)
    ## plt.show()

    return road_map, skdtree


def dijkstra_planning(sx, sy, gx, gy, ox, oy, rr, road_map, sample_x, sample_y,
                      cost_fn=None, cstr_fn=None):
    """
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    reso: grid resolution [m]
    rr: robot radius[m]
    """

    nstart = Node(sx, sy, 0.0, -1)
    ngoal = Node(gx, gy, 0.0, -1)

    openset, closedset = dict(), dict()
    openset[len(road_map) - 2] = nstart

    while True:
        if len(openset) == 0:
            print("Cannot find path")
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]

        # show graph
        if show_animation and len(closedset.keys()) % 2 == 0:
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            if cost_fn is None:
                dx = sample_x[n_id] - current.x
                dy = sample_y[n_id] - current.y
                d  = math.sqrt(dx**2 + dy**2)
                node = Node(sample_x[n_id], sample_y[n_id],
                            current.cost + d, c_id)
            else:
                cost = cost_fn([sample_x[n_id], sample_y[n_id]]) #-0.01
                node = Node(sample_x[n_id], sample_y[n_id],
                            current.cost + cost, c_id)

            if cstr_fn[0] is not None:
                progress = get_progress([sx,sy], [gx,gy], [sample_x[n_id], sample_y[n_id]])
                if cstr_fn[0]([sample_x[n_id], sample_y[n_id]], progress) is False:
                    continue
                if cstr_fn[1]([sample_x[n_id], sample_y[n_id]], progress) is False:
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
    rx, ry = [ngoal.x], [ngoal.y]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        pind = n.pind

    return rx, ry


def get_progress(s, g, state):
    return sum([abs(s[0]-state[0]), abs(s[1]-state[1])])/(sum([abs(s[0]-state[0]), abs(s[1]-state[1])]) +\
                                                          sum([abs(g[0]-state[0]), abs(g[1]-state[1])]))

def plot_road_map(road_map, sample_x, sample_y):

    for i in range(len(road_map)): #state
        for ii in range(len(road_map[i])): #connected_state
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")


def sample_points(sx, sy, gx, gy, rr, ox, oy, obkdtree, xlim, ylim, n_sample=500):
    maxx = xlim[1]
    maxy = ylim[1]
    minx = xlim[0]
    miny = ylim[0]

    sample_x, sample_y = [], []

    while len(sample_x) <= n_sample-3:
        tx = (random.random() - minx) * (maxx - minx)
        ty = (random.random() - miny) * (maxy - miny)

        index, dist = obkdtree.search(np.matrix([tx, ty]).T)

        if dist[0] >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    robot_size = 5.0  # [m]

    ox = []
    oy = []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")

    rx, ry = PRM_planning(sx, sy, gx, gy, ox, oy, robot_size)

    assert len(rx) != 0, 'Cannot found path'

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()
