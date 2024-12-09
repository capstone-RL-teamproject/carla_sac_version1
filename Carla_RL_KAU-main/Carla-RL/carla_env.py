import atexit
import os
import signal
import sys
import carla
import gym
import time
import random
import numpy as np
import math
from queue import Queue
from threading import Event
from misc import dist_to_roadline, exist_intersection
from setup import setup
from absl import logging
import graphics
import pygame
logging.set_verbosity(logging.INFO)

class CarlaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors,
                 action_type, enable_preview, steps_per_episode, playing=False, timeout=60):
        super(CarlaEnv, self).__init__()

        self.client, self.world, self.frame, self.server = setup(town=town, fps=fps, client_timeout=timeout)
        self.client.set_timeout(5.0)
        self.map = self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()
        self.lincoln = blueprint_library.filter('lincoln')[0]
        self.im_width = im_width
        self.im_height = im_height
        self.repeat_action = repeat_action
        self.action_type = action_type
        self.start_transform_type = start_transform_type
        self.sensors = sensors
        self.actor_list = []
        self.preview_camera = None
        self.steps_per_episode = steps_per_episode
        self.playing = playing
        self.preview_camera_enabled = enable_preview
        
        if self.preview_camera_enabled:
            self.preview_image_Queue = Queue()
        
        self.observation_space = gym.spaces.Dict({
            # 'image' = depth + segmentation, 'vehilce' = spped + steer + distant 
            'image' : gym.spaces.Box(low=0, high= 255, shape=(self.im_width, self.im_height, 6), dtype=np.uint8),
            'vehicle': gym.spaces.Box(low=-1.0, high=np.inf, shape=(3,), dtype=np.float32)

            # 'depth_image': gym.spaces.Box(low=0, high=255, shape=(self.im_width, self.im_height, 3), dtype=np.uint8),
            # 'segmentation_image': gym.spaces.Box(low=0, high=255, shape=(self.im_width, self.im_height, 3), dtype=np.uint8),
            # 'speed': gym.spaces.Box(low=0, high=200, shape=(), dtype=np.float32),
            # 'steer': gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
            # 'distant': gym.spaces.Box(low=0, high=1000, shape=(), dtype=np.float32)
        })
        
    #action space   
    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        if self.action_type == 'continuous':
            return gym.spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]))
        elif self.action_type == 'discrete':
            return gym.spaces.MultiDiscrete([4, 9])
        else:
            raise NotImplementedError()

    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed) 
        return seed

    def reset(self):
        #reset
        self._destroy_agents()
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.actor_list = []
        self.frame_step = 0
        self.out_of_loop = 0
        self.dist_from_start = 0

        self.depth_image = None
        self.segmentation_image = None
        self.depth_event = Event()
        self.segmentation_event = Event()
        
        self.vehicle = None
        self.sensor_depth = None
        self.sensor_segmentation = None

        spawn_start = time.time()
        while True:
            # vehicle spawn
            try:
                self.start_transform = self._get_start_transform()
                self.end_transform = self._get_end_transform()
                self.prev_dist = self.start_transform.location.distance(self.end_transform.location)
                self.curr_loc = self.start_transform.location
                self.vehicle = self.world.spawn_actor(self.lincoln, self.start_transform)
                break
            except Exception as e:
                logging.error(f'Error spawning vehicle: {e}')
                time.sleep(0.01)

            if time.time() > spawn_start + 3:
                raise Exception('Can\'t spawn a car')

        self.actor_list.append(self.vehicle)

        # segmentation and depth camera 
        self.segmentation_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')

        self.segmentation_bp.set_attribute('image_size_x', f'{self.im_width}')
        self.segmentation_bp.set_attribute('image_size_y', f'{self.im_height}')
        self.segmentation_bp.set_attribute('fov', '90')
        self.depth_bp.set_attribute('image_size_x', f'{self.im_width}')
        self.depth_bp.set_attribute('image_size_y', f'{self.im_height}')
        self.depth_bp.set_attribute('fov', '90')

        bound_x = self.vehicle.bounding_box.extent.x
        transform_front = carla.Transform(carla.Location(x=bound_x, z=1.0))
        self.sensor_segmentation = self.world.spawn_actor(self.segmentation_bp, transform_front, attach_to=self.vehicle)
        self.sensor_depth = self.world.spawn_actor(self.depth_bp, transform_front, attach_to=self.vehicle)

        self.sensor_segmentation.listen(self._segmentation_callback)
        self.sensor_depth.listen(self._depth_callback)

        self.actor_list.extend([self.sensor_segmentation, self.sensor_depth])

        if self.preview_camera_enabled:
            self.preview_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
            self.preview_cam.set_attribute('image_size_x', '400')
            self.preview_cam.set_attribute('image_size_y', '400')
            self.preview_cam.set_attribute('fov', '100')
            transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
            self.preview_sensor = self.world.spawn_actor(self.preview_cam, transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.SpringArm)
            self.preview_sensor.listen(self.preview_image_Queue.put)
            self.actor_list.append(self.preview_sensor)

        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        time.sleep(4)
        
        #collision and lane invation list reset
        self.collision_hist = []
        self.lane_invasion_hist = []

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
        self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
        self.colsensor.listen(self._collision_data)
        self.lanesensor.listen(self._lane_invasion_data)
        self.actor_list.append(self.colsensor)
        self.actor_list.append(self.lanesensor)

        self.world.tick()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))
        return self.get_observation()

    def step(self, action):
        total_reward = 0
        for _ in range(self.repeat_action):
            obs, rew, done, info = self._step(action)
            total_reward += rew
            if done:
                break
        return obs, total_reward, done, info

    def _step(self, action):
        self.world.tick()
        self.render()
            
        self.frame_step += 1

        # action space
        if self.action_type == 'continuous':
            action = carla.VehicleControl(throttle=float(action[0]), steer=float(action[1]))
        elif self.action_type == 'discrete':
            if action[0] == 0:
                action = carla.VehicleControl(throttle=0, steer=float((action[1] - 4)/4), brake=1)
            else:
                action = carla.VehicleControl(throttle=float((action[0])/3), steer=float((action[1] - 4)/4), brake=0)
        else:
            raise NotImplementedError()
        logging.debug('{}, {}, {}'.format(action.throttle, action.steer, action.brake))
        self.vehicle.apply_control(action)

        #calculate reward
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        loc = self.vehicle.get_location()
        dist_to_end = loc.distance(self.end_transform.location)

        dist_text = str(dist_to_end)
        self.world.debug.draw_string(location=loc, text=dist_text, life_time=0.01)
        dis_to_left, dis_to_right, sin_diff, cos_diff = dist_to_roadline(self.map, self.vehicle)
        
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        lane_type = waypoint.lane_type

                                         

        done = False
        reward = 0
        info = dict()

        # distance
        reward += (self.prev_dist - dist_to_end) * 3
        
        # 목표 도착시 reward 추가 
        if dist_to_end < 2.0:
            done = True
            reward += 1000
        self.prev_dist = dist_to_end
        
        # collision
        if len(self.collision_hist) != 0:
            done = True
            reward += -120
            self.collision_hist = []
            self.lane_invasion_hist = []
        
        # sidewalk invasion 시 reward
        invasion_lane_type = [carla.LaneType.Sidewalk, carla.LaneType.Shoulder]
        if self.lane_invasion_hist in invasion_lane_type:
            done = True     
            reward += -120
            self.lane_invasion_hist = []
        
        #invasion
        if len(self.lane_invasion_hist) != 0:
            reward += -10
            self.lane_invasion_hist = []

        # speed
        reward += kmh*0.2
        """
        if (kmh >= 10) and (kmh < 20):
            reward += kmh*0.5
        elif (kmh >= 20) and (kmh < 30):
            reward += kmh*0.2
        else :
            reward += -kmh*0.1
        """

        # lane Center
        if not self.playing:
            reward -= math.exp(-dis_to_left)
            reward -= math.exp(-dis_to_right)
        
        self.prev_dist = dist_to_end
        
        if self.frame_step >= self.steps_per_episode:
            done = True

        self.world.debug.draw_arrow(begin=self.start_transform.location, end=self.end_transform.location, life_time=1.0)

        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        obs = self.get_observation()
        

        if done:
            logging.debug("Env lasts {} steps, restarting ... ".format(self.frame_step))
            self._destroy_agents()

        return obs, reward, done, info
    
    def close(self):
        logging.info("Closes the CARLA server with process PID {}".format(self.server.pid))
        self._destroy_agents()
        os.killpg(self.server.pid, signal.SIGKILL)
        atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))
    
    def render(self, mode='human'):
        if self.preview_camera_enabled:
            self._display, self._clock, self._font = graphics.setup(
                width=400,
                height=400,
                render=(mode=="human"),
            )

            preview_img = self.preview_image_Queue.get()
            preview_img = np.array(preview_img.raw_data)
            preview_img = preview_img.reshape((400, 400, -1))
            preview_img = preview_img[:, :, :3]
            graphics.make_dashboard(
                display=self._display,
                font=self._font,
                clock=self._clock,
                observations={"preview_camera": preview_img},
            )

            if mode == "human":
                pygame.display.flip()
            else:
                raise NotImplementedError()

    def _destroy_agents(self):
        for actor in self.actor_list:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []

    def _collision_data(self, event):
        self.collision_hist.append(event)
    
    def _lane_invasion_data(self, event):
        # Change this function to filter lane invasions
        #self.lane_invasion_hist.append(event)

        invasion_types = set([x.type for x in event.crossed_lane_markings])
        
        # Sidewalk invasion
        if carla.LaneType.Sidewalk in invasion_types:
            logging.info("Sidewalk invasion detected!")
            self.lane_invasion_hist.append('sidewalk')


    def _get_start_transform(self):
        if self.start_transform_type == 'random':
            return random.choice(self.map.get_spawn_points())
        if self.start_transform_type == 'fixed':
            start_transform = self.map.get_spawn_points()[70]
            return start_transform
        if self.start_transform_type == 'highway':
            if self.map.name == "Town04":
                for trial in range(10):
                    start_transform = random.choice(self.map.get_spawn_points())
                    start_waypoint = self.map.get_waypoint(start_transform.location)
                    if start_waypoint.road_id in list(range(35, 50)):
                        break
                return start_transform
            else:
                raise NotImplementedError
            
    def _get_end_transform(self):
        indices = [213, 215, 217, 71, 221, 224, 72, 87, 108]
        end_transform = []
        for i in indices:
            end_transform.append(self.map.get_spawn_points()[i])
        return random.choice(end_transform)

    def _depth_callback(self, image):
        self.depth_image = np.array(image.raw_data).reshape((self.im_width, self.im_height, -1))[:, :, :3]
        self.depth_event.set()

    def _segmentation_callback(self, image):
        self.segmentation_image = np.array(image.raw_data).reshape((self.im_width, self.im_height, -1))[:, :, :3]
        self.segmentation_event.set()

    def get_observation(self):
        self.depth_event.clear()
        self.segmentation_event.clear()

        while not (self.depth_event.is_set() and self.segmentation_event.is_set()):
            time.sleep(0.01)
            self.world.tick()

        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        control = self.vehicle.get_control()
        steer = control.steer
        distant = self.vehicle.get_location().distance(self.end_transform.location)
        
        combine_image = np.concatenate((self.depth_image, self.segmentation_image), axis = 2)
        vehicle = np.array([speed,steer,distant])
        observation = {
            # 'depth_image': self.depth_image,
            # 'segmentation_image': self.segmentation_image,
            'image': combine_image,
            'vehicle': vehicle
            # 'speed': speed,
            # 'steer': steer,
            # 'distant': distant
        }
        return observation
