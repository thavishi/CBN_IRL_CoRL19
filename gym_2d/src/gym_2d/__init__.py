from gym.envs.registration import register

register(
    id='reaching-v0',
    entry_point='gym_2d.envs:ReachingEnv',
)

register(
    id='reaching-v1',
    entry_point='gym_2d.envs:ContWorldEnv',
)

#register(
#    id='mnist-v0',
#    entry_point='gym_cstr.envs:MnistEnv',
#)
