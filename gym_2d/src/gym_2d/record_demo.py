from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.widgets import Button
from ds_tools.mousetrajectory_gui import MouseTrajectory
import config2D as conf


rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

'''
 Brings up an empty world environment to draw "human-demonstrated" trajectories
'''

def generate_demo():

    #Create figure/environment to draw trajectories on
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    points,   = ax.plot([], [], 'ro', markersize=2, lw=2)
    ax.set_xlim(conf.base[0], conf.base[1])
    ax.set_ylim(conf.base[0], conf.base[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$',fontsize=15)
    plt.ylabel('$x_2$',fontsize=15)
    plt.title('Draw trajectories to learn a motion policy:',fontsize=15)

    i = 1
    while i <=conf.objects[0][0]:
        ax.add_artist(plt.Rectangle((conf.objects[i][0], conf.objects[i][1]), conf.objects[i][2], conf.objects[i][3], fc='r'))
        i = i+1
    #ax.add_artist(plt.Rectangle((0.50, 0.25), 0.4, 0.1))
    ax.plot(conf.start[0], conf.start[1], 'rd', markersize=10, lw=2, color = 'b')
    ax.plot(conf.goal[0], conf.goal[1], 'rd', markersize=10, lw=2, color = 'g')

    # Add UI buttons for data/figure manipulation
    store_btn  = plt.axes([0.67, 0.05, 0.075, 0.05])
    clear_btn  = plt.axes([0.78, 0.05, 0.075, 0.05])
    snap_btn   = plt.axes([0.15, 0.05, 0.075, 0.05])    
    bstore     = Button(store_btn, 'store')    
    bclear     = Button(clear_btn, 'clear')    
    bsnap      = Button(snap_btn, 'snap')


    # Calling class to draw data on top of environment
    indexing  = 1 # Set to 1 if you the snaps/data to be indexed with current time-stamp
    store_mat = 0 # Set to 1 if you want to store data in .mat structure for MATLAB
    draw_data = MouseTrajectory(points, indexing, store_mat)
    draw_data.connect()
    bstore.on_clicked(draw_data.store_callback)
    bclear.on_clicked(draw_data.clear_callback)
    bsnap.on_clicked(draw_data.snap_callback)


    # Show
    plt.show()
