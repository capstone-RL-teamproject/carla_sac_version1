
from WrappedGymEnv import *
import gym



params1 = {
    'carla_port': 2000,
    'map_name': 'Town10HD',
    'window_resolution': [1080, 1080],
    'grid_size': [3, 3],
    'sync': True,
    'no_render': False,
    'display_sensor': True,
    'ego_filter': 'vehicle.tesla.model3',
    'num_vehicles': 50,
    'num_pedestrians': 20,
    'enable_route_planner': True,
    'sensors_to_amount': ['left_rgb', 'front_rgb', 'right_rgb', 'top_rgb', 'lidar', 'radar'],
}
params2 = {
    'carla_port': 2100,
    'map_name': 'Town10HD',
    'window_resolution': [1080, 1080],
    'grid_size': [3, 3],
    'sync': True,
    'no_render': False,
    'display_sensor': True,
    'ego_filter': 'vehicle.tesla.model3',
    'num_vehicles': 50,
    'num_pedestrians': 20,
    'enable_route_planner': True,
    'sensors_to_amount': ['left_rgb', 'front_rgb', 'right_rgb', 'top_rgb', 'lidar', 'radar'],
}





import carla

def function(port):
    client = carla.Client("localhost", port)
    client.set_timeout(60.0)
    return client

if __name__=="__main__":
    client1 = function(2000)
    client2 = function(2100)
    client1.get_world()
    client2.get_world()

    # env = WrappedGymEnv(gym.make("CarlaRlEnv-v0", params=params1),
    #                     action_repeat=4, image_size=64)
    #
    # env2 = WrappedGymEnv(gym.make("CarlaRlEnv-v0", params=params2),
    #                      action_repeat=4, image_size=64)
    #
    #
    # a1,b1,c1 = env.reset()
    # a2,b2,c2 = env2.reset()
    #
    # print("hi")
    # print(f'a1 {a1} b1 {b1} c1 {c1}\n')
    # print(f'a1 {a2} b1 {b2} c1 {c2}\n')
    #
    # print("end")
