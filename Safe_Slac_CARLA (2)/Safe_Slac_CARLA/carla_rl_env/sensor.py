import pygame
import numpy as np
import time
import carla


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


class SensorManager:
    def __init__(self, world, sensor_type, transform, attached,sensor_options, display_size, display_pos):

        # resources from outside
        self.world = world
        self.sensor_type = sensor_type
        self.transform = transform
        self.attached = attached
        self.sensor_options = sensor_options
        self.display_size = display_size
        self.display_pos = display_pos
        # self created resource
        self.surface = None
        self.measure_data = None
        self.sensor = self.init_sensor(sensor_type, transform,attached, sensor_options, display_size)

        self.timer = CustomTimer()
        self.time_processing = 0.0
        self.tics_processing = 0

    def init_sensor(self, sensor_type, transform, attached, sensor_options, display_size):

        """
        radar, gnss, imu sensor 추가 하기

        :param sensor_type:
        :param transform:
        :param attached:
        :param sensor_options:
        :param display_size:
        :return: sensor type에 맞는 sensor 설정 및 센서 생성
        """
        if sensor_type == 'RGBCamera':

            # camera blueprint
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(display_size[0]))
            camera_bp.set_attribute('image_size_y', str(display_size[1]))

            # 이미지 크기, 초점거리 등 option설정
            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)

            img_size = (self.display_size[0], self.display_size[1], 3)# WHC

            self.measure_data = np.zeros((img_size), dtype=np.uint8)
            camera.listen(self.save_rgb_image)
            return camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')


            # lidar_bp.set_attribute('range', '100')

            # 추천값 사용
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit',lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])
            lidar = self.world.spawn_actor(lidar_bp, transform,attach_to=attached)
            img_size = (self.display_size[0], self.display_size[1], 3)
            self.measure_data = np.zeros((img_size), dtype=np.uint8)
            lidar.listen(self.save_lidar_image)
            return lidar

        elif sensor_type == 'Radar':
            radar_bp  = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key,sensor_options[key])

            radar = self.world.spawn_actor(radar_bp,transform,attach_to=attached)
            img_size = (self.display_size[0],self.display_size[1],3)
            self.measure_data = np.zeros((img_size),dtype=np.uint8)
            radar.listen(self.save_radar_image)
            return radar

        elif sensor_type == 'Collision':
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            for key in sensor_options:
                collision_bp.set_attribute(key, sensor_options[key])
            collision = self.world.spawn_actor(collision_bp, transform, attach_to=attached)
            collision.listen(self.save_collision_msg)
            return collision

        elif sensor_type == 'Lane_invasion':
            lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
            for key in sensor_options:
                lane_invasion_bp.set_attribute(key, sensor_options[key])
            lane_invasion = self.world.spawn_actor(lane_invasion_bp, transform, attach_to=attached)
            lane_invasion.listen(self.save_lane_invasion_msg)
            return lane_invasion


        else:
            return None

    def get_sensor(self):
        return self.sensor

    def destroy_sensor(self):
        if self.sensor.is_alive:
            self.sensor.destroy()

        del self.sensor
        del self.surface
        del self.measure_data
        del self.timer
        del self.time_processing
        del self.tics_processing

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw) # Raw 정보
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # numpy 변환
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.measure_data = array.swapaxes(0, 1) # BGR-> RGB
        self.surface = pygame.surfarray.make_surface(self.measure_data)

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def save_lidar_image(self, image):
        """
        3D 포인트 클라우드 데이터를 추출한 후, 2D 평면상에 매핑하고 고도(z) 정보를 기반으로 색상화하여 이미지로 변환.
        이 이미지를 Pygame에서 렌더링 가능한 표면으로 변환.
        라이다 데이터 포인트 클라우드의 2d평면상의 위치 정보 알수가 있다 z축(높이정보) 추출하여 색상으로 변환한다 높이가 높은 물체는 밝은색상
        반대는 어두운 색상  이를 통해 고도 정보를 평면상에서 시각화 할수있다

        :param image:
        :return:
        """
        t_start = self.timer.time()

        lidar_range = 2.0 * float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4')) # numpy 변환

        # (x, y, z ,좌표와 반사율) -> (n,4) 형식의 2차원 배열로 변환
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points_height = np.array(points[:, 2], dtype=np.float32)
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(self.display_size) / lidar_range
        lidar_data += (0.5 * self.display_size[0], 0.5 * self.display_size[1])
        lidar_data = np.fabs(lidar_data)
        lidar_data = lidar_data.astype(np.int32)
        lidar_img_size = (self.display_size[0], self.display_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        # lidar_img[tuple(lidar_data.T)] = (0,255,0)
        points_height = (points_height - (-1.4)) / (-1.0 - (-1.4))
        points_height = np.expand_dims(points_height, axis=1)
        points_height = np.clip(points_height, 0.0, 1.0)

        height_data = points_height * 255.0
        height_data = np.append(height_data, (1.0 - points_height) * 255.0, axis=1)
        height_data = np.append(height_data, np.ones_like(points_height) * 0.0, axis=1)

        lidar_img[tuple(lidar_data.T)] = height_data.astype(np.uint8)

        lidar_img = np.rot90(lidar_img, 3)
        self.measure_data = lidar_img
        self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def save_radar_image(self, radar_data):
        t_start = self.timer.time()

        radar_range = 1.0 * float(self.sensor_options['range'])
        radar_scale = min(self.display_size) / radar_range
        radar_offset = min(self.display_size) / 2.0

        self.surface = pygame.Surface(self.display_size).convert()
        self.surface.fill(pygame.Color(0, 0, 0))

        current_rot = radar_data.transform.rotation

        for detect in radar_data:
            alt = detect.altitude
            azi = detect.azimuth
            dpt = detect.depth
            x = dpt * np.cos(alt) * np.cos(azi)
            y = dpt * np.cos(alt) * np.sin(azi)
            z = dpt * np.sin(alt)

            center_point = pygame.math.Vector2(x * radar_scale, y * radar_scale + radar_offset)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            velocity_limit = 20.0
            norm_velocity = detect.velocity / velocity_limit
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            pygame.draw.circle(self.surface, pygame.Color(r, g, b),center_point, 5)

        self.measure_data = pygame.surfarray.array3d(self.surface)

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def save_collision_msg(self, collision_msg):
        self.measure_data = True

    def save_lane_invasion_msg(self, lane_invasion_msg):
        """
        lane_invasion_msg.crossed_lane_markings에 포함된 차선 유형(실선, 점선 등)을 추출
        crossed_lane_markings는 차량이 침범한 차선의 종류를 포함하고 있으며, x.type을 통해 차선의 유형을 얻을 수 있다.
        set을 사용하는 이유는 차량이 한 번에 여러 차선을 넘을 수 있기 때문에, 중복된 차선 유형을 제거하기 위함.
        list_type[-1]은 마지막 차선의 유형을 가져오며, 이는 차량이 마지막으로 넘은 차선에 대한 정보를 제공한다.

        :param lane_invasion_msg:
        :return:
        """
        list_type = list(set(x.type for x in lane_invasion_msg.crossed_lane_markings))

        self.measure_data = str(list_type[-1])
