# METE ALP POTUK
# 150190013
# AI HW2 PROBLEM-2
from GridWorldEnvironment import GridWorldEnv
from NaiveRobot import NaiveRobot # a robot that moves randomly (used to build the environment)
from QRobot import QRobot

robot= QRobot(q_table_mode="zeros", given_trajectories=None, robot_speed=1000)
# q table modes => "zeros", "random" | given trajectory modes => None, 1, 2 | robot speed suggestions => 10, 100, 1000
env = GridWorldEnv(robot=robot, mode="jetpack")
# environment modes => "vanilla", "jetpack"
env.run()
