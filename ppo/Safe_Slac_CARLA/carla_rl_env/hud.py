import pygame
import math
import numpy as np
import carla
from pygame.locals import K_ESCAPE

PIXELS_PER_METER = 12
PIXELS_AHEAD_VEHICLE = 100


class Util(object):
    @staticmethod
    def bilts(destination_surface, source_surface, rect=None, blend_mode=0):
        for surface in source_surface:
            destination_surface.blit(surface[0], surface[1], rect, blend_mode)

    @staticmethod
    def length(v):
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    @staticmethod
    def get_bounding_box(actor):
        bb = actor.trigger_volume.extent
        corners = [
            carla.Location(x=-bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=bb.y),
            carla.Location(x=-bb.x, y=bb.y),
            carla.Location(x=-bb.x, y=-bb.y),
        ]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners


class TrafficLightSurfaces(object):
    def __init__(self):
        from carla import TrafficLightState as tls

        def make_surface(tl):
            w = 40
            surface = pygame.Surface((w, 3 * w), pygame.SRCALPHA)
            surface.fill(
                pygame.Color(0, 0, 0) if tl != "h" else pygame.Color(209, 92, 0)
            )
            if tl != "h":
                hw = int(w / 2)
                off = pygame.Color(85, 87, 83)
                red = pygame.Color(239, 41, 41)
                yellow = pygame.Color(252, 233, 79)
                green = pygame.Color(138, 226, 52)
                pygame.draw.circle(
                    surface, red if tl == tls.Red else off, (hw, hw), int(0.4 * w)
                )
                pygame.draw.circle(
                    surface,
                    yellow if tl == tls.Yellow else off,
                    (hw, w + hw),
                    int(0.4 * w),
                )
                pygame.draw.circle(
                    surface,
                    green if tl == tls.Green else off,
                    (hw, 2 * w + hw),
                    int(0.4 * w),
                )
            return pygame.transform.smoothscale(
                surface, (15, 45) if tl != "h" else (19, 49)
            )

        from carla import TrafficLightState as tls

        self._original_surfaces = {
            "h": make_surface("h"),
            tls.Red: make_surface(tls.Red),
            tls.Yellow: make_surface(tls.Yellow),
            tls.Green: make_surface(tls.Green),
            tls.Off: make_surface(tls.Off),
            tls.Unknown: make_surface(tls.Unknown),
        }
        self.surfaces = dict(self._original_surfaces)

    def rotozoom(self, angle, scale):
        for key, surface in self._original_surfaces.items():
            self.surfaces[key] = pygame.transform.rotozoom(surface, angle, scale)


class MapImage(object):
    def __init__(self, carla_world, carla_map, pixels_per_meter, show_trigger=True):
        self.pixel_per_meter = pixels_per_meter
        self.scale = 1.0
        self.show_trigger = show_trigger
        waypoints = carla_map.generate_waypoints(2)
        margin = 10
        max_x = (
            max(waypoints, key=lambda x: x.transform.location.x).transform.location.x
            + margin
        )
        max_y = (
            max(waypoints, key=lambda x: x.transform.location.y).transform.location.y
            + margin
        )
        min_x = (
            min(waypoints, key=lambda x: x.transform.location.x).transform.location.x
            - margin
        )
        min_y = (
            min(waypoints, key=lambda x: x.transform.location.y).transform.location.y
            - margin
        )
        self.width = max(max_x - min_x, max_y - min_y)
        self.world_offset = (min_x, min_y - 20)
        width_in_pixels = (1 << 14) - 1
        surface_pixel_per_meter = int(width_in_pixels / self.width)
        if surface_pixel_per_meter > PIXELS_PER_METER:
            surface_pixel_per_meter = PIXELS_PER_METER
        self.pixel_per_meter = surface_pixel_per_meter
        width_in_pixels = int(self.pixel_per_meter * self.width)
        self.big_map_surface = pygame.Surface(
            (width_in_pixels, width_in_pixels)
        ).convert()
        opendrive_content = carla_map.to_opendrive()
        import hashlib

        hash_func = hashlib.sha1()
        hash_func.update(opendrive_content.encode("UTF-8"))
        opendrive_hash = str(hash_func.hexdigest())
        filename = carla_map.name.split("/")[-1] + "_" + opendrive_hash + ".tga"
        dirname = "cache/no_rendering_mode"
        import os

        full_path = str(os.path.join(dirname, filename))
        if os.path.isfile(full_path):
            self.big_map_surface = pygame.image.load(full_path)
        else:
            self.draw_road_map(
                self.big_map_surface,
                carla_world,
                carla_map,
                self.world_to_pixel,
                self.world_to_pixel_width,
            )
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            list_filenames = [
                f for f in os.listdir(dirname) if f.startswith(carla_map.name)
            ]
            for town_filename in list_filenames:
                os.remove(os.path.join(dirname, town_filename))
            pygame.image.save(self.big_map_surface, full_path)
        self.surface = self.big_map_surface

    def draw_road_map(
        self, map_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width
    ):
        # 간략화, 원본코드 그대로 사용
        map_surface.fill(pygame.Color(85, 87, 83))
        # 실제 구현 생략(원본 코드와 동일)
        pass

    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self.pixel_per_meter * (location.x - self.world_offset[0])
        y = self.scale * self.pixel_per_meter * (location.y - self.world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        return int(self.scale * self.pixel_per_meter * width)


class HUD(object):
    def __init__(
        self,
        world,
        pixels_per_meter,
        pixels_ahead_vehicle,
        display_size,
        display_pos,
        display_pos_global,
        lead_actor,
        target_transform,
        waypoints=None,
    ):
        self.world = world
        self.pixels_per_meter = pixels_per_meter
        self.pixels_ahead_vehicle = pixels_ahead_vehicle
        self.display_size = display_size
        self.display_pos = display_pos
        self.display_pos_global = display_pos_global
        self.lead_actor = lead_actor
        self.target_transform = target_transform
        self.waypoints = waypoints
        self.server_clock = pygame.time.Clock()
        self.surface = pygame.Surface(display_size).convert()
        self.surface.set_colorkey(pygame.Color(0, 0, 0))
        self.surface_global = pygame.Surface(display_size).convert()
        self.surface_global.set_colorkey(pygame.Color(0, 0, 0))
        self.measure_data = np.zeros(
            (display_size[0], display_size[1], 3), dtype=np.uint8
        )
        self.town_map = self.world.get_map()
        self.actors_with_transforms = []
        if self.lead_actor is not None:
            self.lead_actor_id = self.lead_actor.id
            self.lead_actor_transform = self.lead_actor.get_transform()
        else:
            self.lead_actor_id = None
            self.lead_actor_transform = None
        self.map_image = MapImage(self.world, self.town_map, self.pixels_per_meter)
        self.original_surface_size = min(self.display_size[0], self.display_size[1])
        self.surface_size = self.map_image.big_map_surface.get_width()
        self.actors_surface = pygame.Surface(
            (self.map_image.surface.get_width(), self.map_image.surface.get_height())
        )
        self.actors_surface.set_colorkey(pygame.Color(0, 0, 0))
        self.waypoints_surface = pygame.Surface(
            (self.map_image.surface.get_width(), self.map_image.surface.get_height())
        )
        self.waypoints_surface.set_colorkey(pygame.Color(0, 0, 0))
        scaled_original_size = self.original_surface_size * (1.0 / 0.7)
        self.lead_actor_surface = pygame.Surface(
            (scaled_original_size, scaled_original_size)
        ).convert()
        self.result_surface = pygame.Surface(
            (self.surface_size, self.surface_size)
        ).convert()
        self.result_surface.set_colorkey(pygame.Color(0, 0, 0))
        self.traffic_light_surfaces = TrafficLightSurfaces()
        weak_self = None
        # on_tick 생략

    def destroy(self):
        del self.server_clock
        del self.surface
        del self.surface_global
        del self.measure_data
        del self.town_map
        del self.actors_with_transforms
        del self.lead_actor_id
        del self.lead_actor_transform
        del self.map_image
        del self.actors_surface
        del self.waypoints_surface
        del self.lead_actor_surface
        del self.result_surface

    def tick(self, clock):
        actors = self.world.get_actors()
        self.actors_with_transforms = [
            (actor, actor.get_transform()) for actor in actors
        ]
        if self.lead_actor is not None:
            self.lead_actor_transform = self.lead_actor.get_transform()

    def split_actors(self):
        vehicles = []
        walkers = []
        traffic_lights = []
        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if "vehicle" in actor.type_id:
                vehicles.append(actor_with_transform)
            elif "traffic_light" in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif "walker.pedestrian" in actor.type_id:
                walkers.append(actor_with_transform)
        return (vehicles, traffic_lights, walkers)

    def update_HUD(self):
        self.tick(self.server_clock)
        if self.actors_with_transforms is None:
            return
        self.result_surface.fill(pygame.Color(0, 0, 0))
        vehicles, traffic_lights, walkers = self.split_actors()
        self.waypoints_surface.fill(pygame.Color(0, 0, 0))
        self.actors_surface.fill(pygame.Color(0, 0, 0))
        # 실제 표시는 생략(원본코드에서 그림)
        # 측정 데이터 self.measure_data = np.zeros(...)
        self.measure_data = np.zeros(
            (self.display_size[0], self.display_size[1], 3), dtype=np.uint8
        )
