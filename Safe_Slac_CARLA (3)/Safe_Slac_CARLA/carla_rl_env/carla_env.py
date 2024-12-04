import numpy as np
import sys
import pygame
import random
import time
import gym
from gym.spaces import Dict, Discrete, Box, Tuple
import carla

from carla_rl_env.hud import HUD,PIXELS_PER_METER,PIXELS_AHEAD_VEHICLE
from carla_rl_env.Planner import RoutePlanner
from carla_rl_env.sensor import SensorManager
from carla_rl_env.controller import VehiclePIDController
from PIL import Image
from gym.spaces.box import Box
import ray

"""
SAC algorithm for CARLA ENV 


"""



# 센서들 데이터와   hud surface, hud surface global 모두  오프셋 만큼 격자로 렌더링 함
class DisplayManager:
    def __init__(self, grid_size, window_size, display_sensor):
        pygame.init()
        pygame.font.init()
        try:
            if display_sensor:
                self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SHOWN)
            else:
                self.display = pygame.display.set_mode(window_size,pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.HIDDEN)
            self.display.fill(pygame.Color(0, 0, 0))
        except Exception:
            print("display is not correctly created in init")

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []
        self.hud = None

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0] / self.grid_size[1]), int(self.window_size[1] / self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def add_birdeyeview(self, hud):
        self.hud = hud

    def render(self):
        # 사용하는 sensor offset 간격만큼 위치 정해서 display한다
        for s in self.sensor_list:
            if s.surface is not None:
                self.display.blit(s.surface, self.get_display_offset(s.display_pos))

        # surface display
        """
        surface : 로컬뷰 담고 있는 surface (차량 화면 중심)
        surface_global : 전체 맵을 보여주는 글로벌 뷰 
        """
        self.display.blit(self.hud.surface, self.get_display_offset(self.hud.display_pos))
        self.display.blit(self.hud.surface_global, self.get_display_offset(self.hud.display_pos_global))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.flip()

    def clear(self):
        self.sensor_list = []
        self.hud = None



"""
목표 지점 위치 정보 설정 생성및 제거
도착 지점 바꾸려면  transform -> spawn point = random.choice(world.get_map().get_spawn_points())

화면상에서 도착 지점 선택한 위치를(transform info)를 이 클래스에 넘겨야 함 
"""
class TargetPosition(object):
    def __init__(self, transform):

        # resource from outside
        self.transform = transform
        # self created resource
        self.box = None
        self.measure_data = None

        self.set_transform(transform)

    def set_transform(self, transform):
        self.transform = transform
        self.box = carla.BoundingBox(transform.location,carla.Vector3D(1, 1, 1))
        self.measure_data = np.array([
            self.transform.location.x,
            self.transform.location.y,
            self.transform.location.z])

    def destroy_target_pos(self):
        del self.box
        del self.measure_data


@ray.remote#(num_gpus=0.25,num_cpus=1)
class CarlaRlEnv(gym.Env):
    def __init__(self, params):

        # resource from outside
        # parse parameters
        self.carla_port = params['carla_port']
        self.map_name = params['map_name']
        self.window_resolution = params['window_resolution']
        self.grid_size = params['grid_size']
        self.sync = params['sync']
        self.no_render = params['no_render']
        self.display_sensor = params['display_sensor']
        self.ego_filter = params['ego_filter']
        self.num_vehicles = params['num_vehicles']
        self.num_pedestrians = params['num_pedestrians']
        self.enable_route_planner = params['enable_route_planner']
        self.sensors_to_amount = params['sensors_to_amount']
        self.image_size = params["image_size"]
        self.worker_rank = params["worker_rank"]


        # self created resource
        # connet to server
        self.client = carla.Client('localhost', self.carla_port)  #
        self.client.set_timeout(60.0)

        # get world and map
        self.world = self.client.get_world()  #
        self.world = self.client.load_world(self.map_name)

        self.spectator = self.world.get_spectator()  # return spectator actor
        self.map = self.world.get_map()  #
        self.spawn_points = self.map.get_spawn_points()  #
        self.original_settings = self.world.get_settings()  #

        if self.no_render:
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

        if self.sync:
            traffic_manager = self.client.get_trafficmanager(8000+self.worker_rank*100) # actor,traffic manage
            settings = self.world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            self.world.apply_settings(settings)

        self.ego_vehicle = None
        self.display_manager = DisplayManager(self.grid_size, self.window_resolution, self.display_sensor)
        self.display_size = self.display_manager.get_display_size()
        self.vehicle_list = []
        self.sensor_list = []
        self.hud = None
        self.pedestrian_list = []
        self.pedestrian_controller_list = []

        self.target_pos = None  # TargetPosition(carla.Transform())

        self.route_planner_global = RoutePlanner(self.map, 0.5) #(map,resolution)
        self.waypoints = None
        self.waypoints_horizon = None
        self.last_waypoints_len = None

        self.current_step = 0
        self.reward = 0.0
        self.done = False
        self.cost = 0.0
        self.total_step = 0

        # future work :  more camera sensor , radar, gnss, imu sensor
        self.front_camera = None
        self.left_camera = None
        self.right_camera =None
        self.top_camera = None
        self.lidar = None
        self.radar = None
        self.collision = None
        self.lane_invasion = None

        # acc and brake percentage, steering percentage, and reverse flag
        self.action_space = Tuple((Box(np.array([0.0, 0.0, -1.0]), 1.0, shape=(3,), dtype=np.float32),Discrete(2))) # [throttle, brake, action[1]]

        # future work : more camera , radar, gnss,imu sensor fusion
        self.observation_space = Dict({
            'front_camera': Box(0, 255, shape=(self.display_size[0],self.display_size[1], 3), dtype=np.uint8),
            'left_camera': Box(0, 255, shape=(self.display_size[0],self.display_size[1], 3), dtype=np.uint8),
            'right_camera': Box(0, 255, shape=(self.display_size[0], self.display_size[1], 3), dtype=np.uint8),
            'top_camera': Box(0, 255, shape=(self.display_size[0], self.display_size[1], 3), dtype=np.uint8),
            'lidar_image': Box(0, 255, shape=(self.display_size[0], self.display_size[1], 3), dtype=np.uint8),
            'radar_image': Box(0, 255, shape=(self.display_size[0], self.display_size[1], 3), dtype=np.uint8),
            'hud': Box(0, 255, shape=(self.display_size[0], self.display_size[1], 3), dtype=np.uint8),
            'trgt_pos': Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            'wp_hrz': Box(-np.inf, np.inf, shape=(40, 2), dtype=np.float32)
        })
        self.height = self.image_size
        self.width = self.image_size
        self.action_repeat = 4
        self._max_episode_steps = 1000

        self.observation_space = Box(0, 255, (18, self.height, self.width), np.uint8)

        self.ometer_space = Box(-np.inf, np.inf, shape=(40, 2), dtype=np.float32)
        self.tgt_state_space = Box(0, 255, (3, self.height, self.width), np.uint8)


        self.wrap_action_space = Box(-1.0, 1.0, shape=(2,))


    def get_env_shape(self):
        return {
            "env_observation_space_shape":self.observation_space.shape,
            "env_ometer_space_shape":self.ometer_space.shape,
            "env_tgt_state_space_shape":self.tgt_state_space.shape,
            "env_action_space_shape":self.wrap_action_space.shape
        }

    def step(self,action):
        if action[0] > 0:
            throttle = np.clip(action[0], 0.0, 1.0)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-action[0], 0.0, 1.0)

        act_tuple = ([throttle, brake, action[1]], [False])

        for _ in range(self.action_repeat):
            re_ref = self.carla_step(act_tuple) #.remote(act_tuple)

        re = re_ref #ray.get(re_ref)
        re = list(re)
        img_np = re[0]['left_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_1 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = re[0]['front_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_2 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = re[0]['right_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_3 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = re[0]['top_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_4 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = re[0]['lidar_image']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_5 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = re[0]['radar_image']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_6 = np.transpose(img_np_resized, [2, 1, 0])

        src_img = np.concatenate((src_img_1,
                                  src_img_2,
                                  src_img_3,
                                  src_img_4,
                                  src_img_5,
                                  src_img_6), axis=0)

        img_np = re[0]['hud']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        tgt_img = np.transpose(img_np_resized, [2, 0, 1])


        wpsh = re[0]['wp_hrz']

        return src_img, wpsh, tgt_img, re[1], re[2], re[3]

    def carla_step(self, action):


        self.current_step += 1
        self.total_step += 1

        acc = action[0][0]
        brk = action[0][1]
        trn = action[0][2]
        rvs = action[1][0]



        act = carla.VehicleControl(throttle=float(acc), steer=float(trn), brake=float(brk), reverse=bool(rvs))

        self.ego_vehicle.apply_control(act)

        self.world.tick()

        reward, done, cost = self.deal_with_reward_and_done()

        self.hud.update_HUD()

        transform = self.ego_vehicle.get_transform()
        transform.location.z += 10
        transform.rotation.pitch -= 90
        # self.spectator.set_transform(transform)

        img_size = (180, 180)
        observation = {
            'front_camera': self.front_camera.measure_data if self.front_camera is not None else np.zeros(img_size),
            'left_camera': self.left_camera.measure_data if self.left_camera is not None else np.zeros(img_size),
            'right_camera': self.right_camera.measure_data if self.right_camera is not None else np.zeros(img_size),
            'top_camera' : self.top_camera.measure_data if self.top_camera is not None else np.zeros(img_size),
            'lidar_image': self.lidar.measure_data if self.lidar is not None else np.zeros(img_size),
            'radar_image': self.radar.measure_data if self.radar is not None else np.zeros(img_size),
            'hud': self.hud.measure_data,
            'trgt_pos': self.target_pos.measure_data,
            'wp_hrz': self.waypoints_horizon,
        }

        info = {'cost': cost}

        return observation, reward, done, info

    def reset(self):
        self.current_step = 0

        self.remove_all_actors()
        self.world.tick()
        self.create_all_actors()
        self.world.tick()

        self.reward = 0.0
        self.done = False
        self.cost = 0.0
        _, _ = self.update_waypoints_and_horizon()
        img_size = (180, 180)
        observation = {
            'front_camera': self.front_camera.measure_data if self.front_camera is not None else np.zeros(img_size),
            'left_camera': self.left_camera.measure_data if self.left_camera is not None else np.zeros(img_size),
            'right_camera': self.right_camera.measure_data if self.right_camera is not None else np.zeros(img_size),
            'top_camera' : self.top_camera.measure_data if self.top_camera is not None else np.zeros(img_size),
            'lidar_image': self.lidar.measure_data if self.lidar is not None else np.zeros(img_size),
            'radar_image': self.radar.measure_data if self.radar is not None else np.zeros(img_size),
            'hud': self.hud.measure_data,
            'trgt_pos': self.target_pos.measure_data,
            'wp_hrz': self.waypoints_horizon,
        }

        reset_output=observation
        #reset_output=self.env.reset()

        img_np = reset_output['left_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_1 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = reset_output['front_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_2 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = reset_output['right_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_3 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = reset_output['top_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_4 = np.transpose(img_np_resized, [2, 1, 0])


        img_np = reset_output['lidar_image']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_5 = np.transpose(img_np_resized, [2, 1,0])

        img_np = reset_output['radar_image']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_6 = np.transpose(img_np_resized, [2, 1, 0])


        src_img = np.concatenate((src_img_1,
                                  src_img_2,
                                  src_img_3,
                                  src_img_4,
                                  src_img_5,
                                  src_img_6), axis=0)


        img_np = reset_output['hud']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        tgt_img = np.transpose(img_np_resized, [2, 0, 1])

        wpsh = reset_output['wp_hrz']

        return src_img, wpsh, tgt_img
        #return observation

    def update_waypoints_and_horizon(self):
        """
        현재 차량 위치에서 가까운 waypoint 계산해서 일정거리의 경로 계산한다
        실시간으로 경로 추정하고 향후 주행할 경로를 계획하기 위한 함수이다


        :return:
        current_location : 현재 차량위치
        wp : 마지막으로 선택된 현재 가장 가까운 waypoint

        """
        current_location = self.ego_vehicle.get_transform().location # 현재 차량 위치
        num_checked_waypoints = min(10, len(self.waypoints)) # 최대 10개의 경로만 계산한다
        dis = 10000.0
        idx = 1000

        for n_c_wp in range(num_checked_waypoints):
            if current_location.distance(self.waypoints[n_c_wp][0].transform.location) < dis:
                dis = current_location.distance(self.waypoints[n_c_wp][0].transform.location)
                if len(self.waypoints) > 10:
                    wp = self.waypoints[9]
                else:
                    wp = self.waypoints[-1]
                idx = n_c_wp

        for _ in range(idx):
            self.waypoints.pop(0)

        waypoints_horizon = [] # 차량이 따라야할 미래 경로 나타내는 리스트 최대 40개의 경로점만 미리 계산
        if len(self.waypoints) > 40:
            wpsh = self.waypoints[:40]
        else:
            wpsh = self.waypoints + [self.waypoints[-1]] * (40 - len(self.waypoints))
        for wph in wpsh:
            waypoints_horizon.append(np.array([wph[0].transform.location.x, wph[0].transform.location.y]))
        self.waypoints_horizon = np.array(waypoints_horizon)

        return current_location, wp

    # reward function
    def deal_with_reward_and_done(self):
        self.reward = 0.0
        self.cost = 0.0
        current_location, wp = self.update_waypoints_and_horizon() # 현재 차량 위치, 현재 위치에서 가장가까운 마지막 waypoint
        """
        cal_error_2D function 
        주어진 목표지점(waypoint)와 현재 위치(location)사이에서 측면 오차를 계산한다 
        즉 목표지점이 향하는 방향과 현재 위치간의 옆으로 얼마나 벗어나있는지를 계산한다 
        차량이 목표 경로를 따라 주행할수있도록 하기 위해 오차 계산한다 
        """
        def cal_error_2D(waypoint, location):

            # 목표지점과 현재 위치 사이의 벡터 계산
            vec_2D = np.array([
                location.x - waypoint[0].transform.location.x,
                location.y - waypoint[0].transform.location.y])

            # 직선 거리 계산
            lv_2D = np.linalg.norm(np.array(vec_2D))

            # 목표지점의 진행 방향 나타내는 2d 방향 벡터 계산
            omega_2D = np.array([
                np.cos(waypoint[0].transform.rotation.yaw / 180.0 * np.pi),
                np.sin(waypoint[0].transform.rotation.yaw / 180.0 * np.pi)])


            # 방향 벡터 omega와 위치간의 벡터 vec2d간의 외적 계산
            cross = np.cross(omega_2D, vec_2D / lv_2D+1e-6)

            return - lv_2D * cross, omega_2D # 얼마나 벗어나있는지, 목표지점의 방향

        # time elapes
        time_reward = -1.0
        if self.current_step > 1000:
            self.done = True

        # collision
        if self.collision.measure_data:
            collision_cost = 1.0
            self.done = True
            self.collision.measure_data = None
        else:
            collision_cost = 0.0

        # lane invasion
        if self.lane_invasion.measure_data:
            if self.lane_invasion.measure_data == 'Broken' or self.lane_invasion.measure_data == 'BrokenSolid' or self.lane_invasion.measure_data == 'BrokenBroken':
                lane_invasion_cost = 0.0
            else:
                self.done = True
                lane_invasion_cost = 1.0
            self.lane_invasion.measure_data = None
        else:
            lane_invasion_cost = 0.0

        current_wp = self.map.get_waypoint(current_location)

        if current_location.distance(current_wp.transform.location) > (current_wp.lane_width / 2.0 + 0.2):
            self.done = True
            lane_invasion_cost = 1.0

        # # traffic light
        # if self.ego_vehicle.is_at_traffic_light():
        #    self.done = True
        #    cross_red_light_reward = -1.0
        # else:
        #    cross_red_light_reward = 0.0

        # speed limit
        current_velocity = self.ego_vehicle.get_velocity()
        current_speed = np.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2 + current_velocity.z ** 2)  # unit m/s
        current_speed_limit = 20.0  # unit m/s
        road_speed_limit = self.ego_vehicle.get_speed_limit()

        if road_speed_limit is not None:
            current_speed_limit = min(current_speed_limit, road_speed_limit / 3.6)

        distance = current_location.distance(self.target_pos.transform.location)

        # 다 도착한 경우
        if distance < 1.0:
            arriving_reward = 1.0
            self.done = True
        else:
            arriving_reward = 0.0

        lat_err, omg = cal_error_2D(wp, current_location) # 측면으로 얼마나 떨어져 있는지, 목표 경로점 방향

        # 측면으로 많이 벗어나있으면 패널티
        if abs(lat_err) > 1.2 * 4.0:
            off_way_reward = - 10.0 * abs(lat_err)
            # self.done = True
        else:
            off_way_reward = - 0.0 * abs(lat_err)

        v_long = np.dot(np.array([current_velocity.x, current_velocity.y]), omg) # 현재 차량 속도와 목표경로점 방향이 얼마나 일치하는지 본다

        # speed limit보다 큰 경우 패널티
        if v_long > current_speed_limit:
            speed_reward = -5.0 * (current_speed_limit - v_long) ** 2.0

        else:
            speed_reward = v_long


        steer_reward = -self.ego_vehicle.get_control().steer ** 2
        lat_acc_reward = - abs(self.ego_vehicle.get_control().steer) * v_long ** 2

        # waypoints len reward
        waypoints_len_reward = self.last_waypoints_len - len(self.waypoints)
        self.last_waypoints_len = len(self.waypoints)



        self.reward = 0.1 * time_reward + 200.0 * arriving_reward + 2.0 * off_way_reward + 0.1 * speed_reward + 3.0 * steer_reward + 0.5 * lat_acc_reward + 3.0 * waypoints_len_reward

        self.cost = 200.0 * collision_cost + 10.0 * lane_invasion_cost

        return self.reward, self.done, self.cost

    def create_all_actors(self):
        self.target_pos = TargetPosition(carla.Transform(carla.Location(-114.23, 53.82, 0.6), carla.Rotation(0.0, 90.0, 0.0))) # 지도에서 목표지점 위치 알아내서 이곳에 넘겨야함
        self.target_pos.set_transform(random.choice(self.spawn_points))

        # create ego vehicle
        ego_vehicle_bp = random.choice([bp for bp in self.world.get_blueprint_library().filter(self.ego_filter) if int(bp.get_attribute('number_of_wheels')) == 4])
        ego_vehicle_bp.set_attribute('role_name', 'lead_actor')

        self.ego_vehicle = None
        while self.ego_vehicle is None:
           self.ego_vehicle = self.world.try_spawn_actor(ego_vehicle_bp,random.choice(self.spawn_points))
           time.sleep(0.1)

        # while self.ego_vehicle is None:
        #     self.ego_vehicle = self.world.try_spawn_actor(ego_vehicle_bp, carla.Transform(carla.Location(-18.386, 130.21, 0.546),carla.Rotation(0.0, -180.0, 0.0)))
        #     # time.sleep(0.1)

        self.vehicle_list.append(self.ego_vehicle)
        self.world.tick()

        self.waypoints = self.route_planner_global.trace_route(self.ego_vehicle.get_location(),self.target_pos.transform.location) # planner따른 경로 설정

        self.last_waypoints_len = len(self.waypoints)
        if len(self.waypoints) == 0:
            print("planned waypoints length is zero")

        bbe_x = self.ego_vehicle.bounding_box.extent.x
        bbe_y = self.ego_vehicle.bounding_box.extent.y
        bbe_z = self.ego_vehicle.bounding_box.extent.z


        # future work : more camera , gnss, imu sensor , radar

        if 'front_rgb' in self.sensors_to_amount:
            self.front_camera = SensorManager(self.world, 'RGBCamera',
                                              carla.Transform(carla.Location(x=0, z=bbe_z + 1.4), carla.Rotation(yaw=+00)),
                                              self.ego_vehicle,
                                              {},
                                              self.display_size, [0, 1])
            self.sensor_list.append(self.front_camera)
            self.display_manager.add_sensor(self.front_camera)

        if 'left_rgb' in self.sensors_to_amount:
            self.left_camera = SensorManager(self.world, 'RGBCamera',
                                             carla.Transform(carla.Location(x=0, z=bbe_z+1.4),carla.Rotation(yaw=-90)),
                                             self.ego_vehicle,
                                             {},
                                             self.display_size, [0, 0])
            self.sensor_list.append(self.left_camera)
            self.display_manager.add_sensor(self.left_camera)

        if 'right_rgb' in self.sensors_to_amount:
            self.right_camera = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=bbe_z+1.4),carla.Rotation(yaw=+90)),
                                              self.ego_vehicle,
                                              {},
                                              self.display_size, [0, 2])
            self.sensor_list.append(self.right_camera)
            self.display_manager.add_sensor(self.right_camera)

        if 'top_rgb' in self.sensors_to_amount:
            self.top_camera = SensorManager(self.world, 'RGBCamera',
                                            carla.Transform(carla.Location(x=0, z=20), carla.Rotation(pitch=-90)),
                                            self.ego_vehicle,
                                            {},
                                            self.display_size, [2, 1])
            self.sensor_list.append(self.top_camera)
            self.display_manager.add_sensor(self.top_camera)

        if 'lidar' in self.sensors_to_amount:
            self.lidar = SensorManager(self.world, 'LiDAR',
                                       carla.Transform(carla.Location(x=0, z=bbe_z + 1.4)),
                                       self.ego_vehicle,
                                       {'channels': '64', 'range': '25.0',
                                        'points_per_second': '250000', 'rotation_frequency': '20'},
                                       self.display_size, [1, 0])
            self.sensor_list.append(self.lidar)
            self.display_manager.add_sensor(self.lidar)


        if 'radar' in self.sensors_to_amount:
            bound_x = 0.5 + bbe_x
            bound_y = 0.5 + bbe_y
            bound_z = 0.5 + bbe_z
            self.radar = SensorManager(self.world, 'Radar',
                                       carla.Transform(carla.Location(x=bound_x + 0.05, z=bound_z+0.05), carla.Rotation(pitch=5)),
                                       self.ego_vehicle, {'horizontal_fov' : '60', 'vertical_fov' : '30', 'range' : '20.0'},
                                       self.display_size, [1, 2])

            self.sensor_list.append(self.radar)
            self.display_manager.add_sensor(self.radar)

        self.hud = HUD(self.world,
                       PIXELS_PER_METER,
                       PIXELS_AHEAD_VEHICLE,
                       self.display_size, [2, 0], [2, 2],
                       self.ego_vehicle,
                       self.target_pos.transform,
                       self.waypoints)

        self.display_manager.add_birdeyeview(self.hud)

        self.collision = SensorManager(self.world, 'Collision',
                                       carla.Transform(), self.ego_vehicle, {}, None, None)

        self.sensor_list.append(self.collision)

        self.lane_invasion = SensorManager(self.world, 'Lane_invasion',
                                           carla.Transform(), self.ego_vehicle, {}, None, None)

        self.sensor_list.append(self.lane_invasion)


        transform = self.ego_vehicle.get_transform()
        transform.location.z += 50  # 10
        transform.rotation.pitch -= 90
        self.spectator.set_transform(transform)

        # create other vehicles
        vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
        vehicle_bps_4wheel = []
        vehicle_bps_4wheel = vehicle_bps_4wheel + [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]

        for vehicle_bp in vehicle_bps_4wheel:
            vehicle_bp.set_attribute('role_name', 'autopilot')

        for _ in range(self.num_vehicles):
            vehicle_tmp_ref = None
            while vehicle_tmp_ref is None:
                vehicle_tmp_ref = self.world.try_spawn_actor(random.choice(vehicle_bps_4wheel), random.choice(self.spawn_points))

            vehicle_tmp_ref.set_autopilot()
            self.vehicle_list.append(vehicle_tmp_ref)

        # create pedestrians
        pedestrian_bps = self.world.get_blueprint_library().filter('walker.*')

        for pedestrian_bp in pedestrian_bps:

            if pedestrian_bp.has_attribute('is_invincible'):
                pedestrian_bp.set_attribute('is_invincible', 'false')


        for _ in range(self.num_pedestrians):
            pedestrian_tmp_ref = None

            while pedestrian_tmp_ref is None:
                pedestrian_spawn_transform = carla.Transform()
                loc = self.world.get_random_location_from_navigation()

                if (loc != None):
                    pedestrian_spawn_transform.location = loc

                pedestrian_tmp_ref = self.world.try_spawn_actor(random.choice(pedestrian_bps),pedestrian_spawn_transform)

            pedestrian_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            pedestrian_controller_actor = self.world.spawn_actor(pedestrian_controller_bp, carla.Transform(),pedestrian_tmp_ref)
            pedestrian_controller_actor.start()
            pedestrian_controller_actor.go_to_location( self.world.get_random_location_from_navigation())
            pedestrian_controller_actor.set_max_speed(1.0 + random.random())

            self.pedestrian_list.append(pedestrian_tmp_ref)
            self.pedestrian_controller_list.append(pedestrian_controller_actor)


    def remove_all_actors(self):

        if self.waypoints is not None:
            del self.waypoints

        if self.target_pos is not None:
            del self.target_pos

        for s in self.sensor_list:
            s.destroy_sensor()
            del s
        self.sensor_list = []

        for v in self.vehicle_list:
            if v.is_alive:
                v.destroy()
            del v

        if self.ego_vehicle is not None:
            del self.ego_vehicle
        self.vehicle_list = []

        for c in self.pedestrian_controller_list:
            if c.is_alive:
                c.stop()
                c.destroy()
            del c

        self.pedestrian_controller_list = []

        for p in self.pedestrian_list:
            if p.is_alive:
                p.destroy()
            del p
        self.pedestrian_list = []
        time.sleep(0.1)

        if self.hud is not None:
            self.hud.destroy()
            del self.hud

        self.display_manager.clear()

    def display(self):
        self.display_manager.render()

    def pid_sample(self):
        """
        Random policy for safe slac in our env

        :return: PID controller act # [0]brake or throttle, [1] steer
        """
        act = [0.0,0.0]
        pid = VehiclePIDController(self.ego_vehicle,{'K_P': 0.5, 'K_I': 0.2, 'K_D': 0.01, 'dt': 0.1},{'K_P': 0.15, 'K_I': 0.07, 'K_D': 0.05, 'dt': 0.1})

        if len(self.waypoints) > 20:

            control = pid.run_step(20.0,self.waypoints[19][0])

        else:

            control = pid.run_step(20.0,self.waypoints[-1][0])

        control.steer += np.random.normal(0.0,0.05)

        if control.brake > 0:

            act[0] = -control.brake

        else:

            act[0] = control.throttle

        act[1] = control.steer

        return act