import glob
import os
import sys
import numpy as np
import carla
from carla import TrafficLightState as tls

import argparse
import logging
import datetime
import weakref
import math
import random
import hashlib
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_COMMA
from pygame.locals import K_DOWN
from pygame.locals import K_ESCAPE
from pygame.locals import K_F1
from pygame.locals import K_LEFT
from pygame.locals import K_PERIOD
from pygame.locals import K_RIGHT
from pygame.locals import K_SLASH
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_h
from pygame.locals import K_i
from pygame.locals import K_m
from pygame.locals import K_p
from pygame.locals import K_q
from pygame.locals import K_s
from pygame.locals import K_w
COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

COLOR_CYAN = pygame.Color(0,255,255)

COLOR_RED   = pygame.Color(255, 0, 0)
COLOR_GREEN = pygame.Color(0, 255, 0)
COLOR_BLUE  = pygame.Color(0, 0, 255)
COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

# Macro Defines
PIXELS_PER_METER = 12
PIXELS_AHEAD_VEHICLE = 100#150


class Util(object):

    @staticmethod
    def bilts(destination_surface,source_surface,rect=None,blend_mode=0):

        # source surface : ((surface,(0,0),(surface,(0,0),,)
        # Render all the surface in a destination_surface

        for surface in source_surface:
            destination_surface.blit(surface[0],surface[1],rect,blend_mode)







class MapImage(object):
    """
    rendering 2d image from carla world. use cach system(opendrive)
    """
    def __init__(self,carla_world, carla_map, pixels_per_meter):
        self.pixel_per_meter = pixels_per_meter
        self.scale = 1.0

        waypoints = carla_map.generate_waypoints(2) # 2m간격으로 waypoint 생성
        margin = 10

        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin

        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self.world_offset = (min_x, min_y-20)

        # Maximum size of a Pygame surface
        width_in_pixels = (1 << 14) - 1

        # Adapt Pixels per meter to make world fit in surface
        surface_pixel_per_meter = int(width_in_pixels / self.width)
        if surface_pixel_per_meter > PIXELS_PER_METER:
            surface_pixel_per_meter = PIXELS_PER_METER

        self.pixel_per_meter = surface_pixel_per_meter
        width_in_pixels = int(self.pixel_per_meter * self.width)

        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()

        # Load OpenDrive content
        opendrive_content = carla_map.to_opendrive()

        # Get hash based on content
        hash_func = hashlib.sha1()
        hash_func.update(opendrive_content.encode("UTF-8"))
        opendrive_hash = str(hash_func.hexdigest())

        # Build path for saving or loading the cached rendered map
        filename = carla_map.name.split('/')[-1] + "_" + opendrive_hash + ".tga"
        dirname = os.path.join("cache", "no_rendering_mode")
        full_path = str(os.path.join(dirname, filename))

        if os.path.isfile(full_path):
            # Load Image
            self.big_map_surface = pygame.image.load(full_path)
        else:
            # Render map
            self.draw_road_map(
                self.big_map_surface,
                carla_world,
                carla_map,
                self.world_to_pixel,
                self.world_to_pixel_width)

            # If folders path does not exist, create it
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            # Remove files if selected town had a previous version saved
            list_filenames = glob.glob(os.path.join(dirname, carla_map.name) + "*")
            for town_filename in list_filenames:
                os.remove(town_filename)

            # Save rendered map for next executions of same map
            pygame.image.save(self.big_map_surface, full_path)

        self.surface = self.big_map_surface

    def draw_road_map(self,map_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):

        map_surface.fill(COLOR_ALUMINIUM_4)
        precision = 0.05

        def lane_marking_color_to_tango(lane_marking_color):
            tango_color = COLOR_BLACK

            if lane_marking_color == carla.LaneMarkingColor.White:
                tango_color = COLOR_ALUMINIUM_2

            elif lane_marking_color == carla.LaneMarkingColor.Blue:
                tango_color = COLOR_SKY_BLUE_0

            elif lane_marking_color == carla.LaneMarkingColor.Green:
                tango_color = COLOR_CHAMELEON_0

            elif lane_marking_color == carla.LaneMarkingColor.Red:
                tango_color = COLOR_SCARLET_RED_0

            elif lane_marking_color == carla.LaneMarkingColor.Yellow:
                tango_color = COLOR_ORANGE_0

            return tango_color

        def lateral_shift(transform, shift):
            """
            Makes a lateral shift of the forward vector of a transform

            주어진 transform 위치를 측면으로 shift만큼 이동하여 새루운 위치 반환 한다
            측면위치 계산 위한 함수이다
            """
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_solid_line(surface, color, closed, points, width):
            """Draws solid lines in a surface given a set of points, width and color
            연속된 실선 차선"""
            if len(points) >= 2:
                pygame.draw.lines(surface, color, closed, points, width)

        def draw_broken_line(surface, color, closed, points, width):
            """Draws broken lines in a surface given a set of points, width and color
            점선 차선"""
            # Select which lines are going to be rendered from the set of lines
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]

            # Draw selected lines
            for line in broken_lines:
                pygame.draw.lines(surface, color, closed, line, width)


        def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign):
            """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken), it converts them
             as a combination of Broken and Solid lines"""
            margin = 0.0
            marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
            if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):

                return [(lane_marking_type, lane_marking_color, marking_1)]
            else:

                marking_2 = [world_to_pixel(lateral_shift(w.transform,sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]

                if lane_marking_type == carla.LaneMarkingType.SolidBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]

            return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]



        def draw_lane(surface, lane, color):
            """
            Renders a single lane in a surface and with a specified color

            왼쪽 차선과 반대방향의 오른쪽 차선을 이어보면 폐곡선 형성됨 -> polygon

            """
            for side in lane:
                lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
                lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]

                polygon = lane_left_side + [x for x in reversed(lane_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    # 외곽선
                    pygame.draw.polygon(surface, color, polygon, 5)

                    # 내부 채우는 기능
                    pygame.draw.polygon(surface, color, polygon)


        def draw_lane_marking_single_side(surface, waypoints, sign):
            """Draws the lane marking given a set of waypoints and decides whether drawing the right or left side of
            the waypoint based on the sign parameter

            sign : -1 왼 , 1 오

            """

            lane_marking = None

            marking_type = carla.LaneMarkingType.NONE
            previous_marking_type = carla.LaneMarkingType.NONE

            marking_color = carla.LaneMarkingColor.Other
            previous_marking_color = carla.LaneMarkingColor.Other

            markings_list = []
            temp_waypoints = []
            current_lane_marking = carla.LaneMarkingType.NONE

            for sample in waypoints:

                lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

                if lane_marking is None:
                    continue

                marking_type = lane_marking.type
                marking_color = lane_marking.color

                if current_lane_marking != marking_type:
                    # Get the list of lane markings to draw
                    markings = get_lane_markings(
                        previous_marking_type,
                        lane_marking_color_to_tango(previous_marking_color),
                        temp_waypoints,
                        sign)
                    current_lane_marking = marking_type

                    # Append each lane marking in the list
                    for marking in markings:
                        markings_list.append(marking)

                    temp_waypoints = temp_waypoints[-1:]

                else:
                    temp_waypoints.append((sample))
                    previous_marking_type = marking_type
                    previous_marking_color = marking_color

            # Add last marking
            last_markings = get_lane_markings(
                previous_marking_type,
                lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                sign)
            for marking in last_markings:
                markings_list.append(marking)

            # Once the lane markings have been simplified to Solid or Broken lines, we draw them

            # markings[0] : solid,broken... type
            # markings[1] : color
            # markings[2] : waypoint list
            for markings in markings_list:
                if markings[0] == carla.LaneMarkingType.Solid:
                    draw_solid_line(surface, markings[1], False, markings[2], 2)
                elif markings[0] == carla.LaneMarkingType.Broken:
                    draw_broken_line(surface, markings[1], False, markings[2], 2)


        def draw_lane_marking(surface, waypoints):
            """Draws the left and right side of lane markings

            opendrive format에 따라 -1은 왼쪽 차선이고 1은 오른쪽이다
            """
            # Left Side
            draw_lane_marking_single_side(surface, waypoints[0], -1)

            # Right Side
            draw_lane_marking_single_side(surface, waypoints[1], 1)


        def draw_traffic_signs(surface, font_surface, actor, color=COLOR_ALUMINIUM_2, trigger_color=COLOR_PLUM_0):
            """Draw stop traffic signs and its bounding box if enabled"""
            transform = actor.get_transform()

            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = carla.Location(-forward_vector.y, forward_vector.x,
                                         forward_vector.z) * waypoint.lane_width / 2 * 0.7

            line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                    (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]

            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)





        def draw_topology(carla_topology, index):
            """ Draws traffic signs and the roads network with sidewalks, parking and shoulders by generating waypoints"""
            topology = [x[index] for x in carla_topology]
            topology = sorted(topology, key=lambda w: w.transform.location.z)
            set_waypoints = []
            for waypoint in topology:
                waypoints = [waypoint]

                # Generate waypoints of a road id. Stop when road id differs
                nxt = waypoint.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    while nxt.road_id == waypoint.road_id:
                        waypoints.append(nxt)
                        nxt = nxt.next(precision)
                        if len(nxt) > 0:
                            nxt = nxt[0]
                        else:
                            break
                set_waypoints.append(waypoints)

                # Draw Shoulders, Parkings and Sidewalks
                PARKING_COLOR  = pygame.Color(0,0,255)#COLOR_ALUMINIUM_4_5
                SHOULDER_COLOR = COLOR_ALUMINIUM_5
                SIDEWALK_COLOR = COLOR_ALUMINIUM_3

                shoulder = [[], []]
                parking = [[], []]
                sidewalk = [[], []]

                for w in waypoints:
                    # Classify lane types until there are no waypoints by going left
                    l = w.get_left_lane()
                    while l and l.lane_type != carla.LaneType.Driving:

                        if l.lane_type == carla.LaneType.Shoulder:
                            shoulder[0].append(l)

                        if l.lane_type == carla.LaneType.Parking:
                            parking[0].append(l)

                        if l.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[0].append(l)

                        l = l.get_left_lane()

                    # Classify lane types until there are no waypoints by going right
                    r = w.get_right_lane()
                    while r and r.lane_type != carla.LaneType.Driving:

                        if r.lane_type == carla.LaneType.Shoulder:
                            shoulder[1].append(r)

                        if r.lane_type == carla.LaneType.Parking:
                            parking[1].append(r)

                        if r.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[1].append(r)

                        r = r.get_right_lane()

                # Draw classified lane types
                draw_lane(map_surface, shoulder, SHOULDER_COLOR)
                draw_lane(map_surface, parking, PARKING_COLOR)
                draw_lane(map_surface, sidewalk, SIDEWALK_COLOR)

            # Draw Roads
            for waypoints in set_waypoints:
                waypoint = waypoints[0]
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)

                # Draw Lane Markings and Arrows
                if not waypoint.is_junction:
                    draw_lane_marking(map_surface, [waypoints, waypoints])
                    #for n, wp in enumerate(waypoints):
                    #    if ((n + 1) % 400) == 0:
                    #        draw_arrow(map_surface, wp.transform)



        topology = carla_map.get_topology()
        draw_topology(topology, 0)



        actors = carla_world.get_actors()

        # Find and Draw Traffic Signs: Stops and Yields
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont('Arial', font_size, True)

        stops = [actor for actor in actors if 'stop' in actor.type_id]
        yields = [actor for actor in actors if 'yield' in actor.type_id]

        stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        stop_font_surface = pygame.transform.scale(
            stop_font_surface, (stop_font_surface.get_width(), stop_font_surface.get_height() * 2))

        yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
        yield_font_surface = pygame.transform.scale(
            yield_font_surface, (yield_font_surface.get_width(), yield_font_surface.get_height() * 2))

        for ts_stop in stops:
            draw_traffic_signs(map_surface, stop_font_surface, ts_stop, trigger_color=COLOR_SCARLET_RED_1)

        for ts_yield in yields:
            draw_traffic_signs(map_surface, yield_font_surface, ts_yield, trigger_color=COLOR_ORANGE_1)

    def world_to_pixel(self, location, offset=(0, 0)):
        """Converts the world coordinates to pixel coordinates"""
        x = self.scale * self.pixel_per_meter * (location.x - self.world_offset[0])
        y = self.scale * self.pixel_per_meter * (location.y - self.world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return int(self.scale * self.pixel_per_meter * width)



class HUD(object):
    def __init__(self,world,pixels_per_meter,pixels_ahead_vehicle,display_size,display_pos,display_pos_global,lead_actor,target_transform,waypoints=None):

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
        self.surface.set_colorkey(COLOR_BLACK)

        self.surface_global = pygame.Surface(display_size).convert()
        self.surface_global.set_colorkey(COLOR_BLACK)

        self.measure_data = np.zeros((display_size[0],display_size[1],3),dtype=np.uint8)


        # world data
        self.town_map = self.world.get_map()
        self.actors_with_transforms = []

        if self.lead_actor is not None:
            self.lead_actor_id = self.lead_actor.id
            self.lead_actor_transform = self.lead_actor.get_transform()

        else:
            self.lead_actor_id =None
            self.lead_actor_transform =None


        # Create surface
        self.map_image = MapImage(self.world,self.town_map,self.pixels_per_meter)

        self.original_surface_size = min(self.display_size[0],display_size[1])
        self.surface_size = self.map_image.big_map_surface.get_width()

        # render actors
        self.actors_surface = pygame.Surface((self.map_image.surface.get_width(),self.map_image.surface.get_height()))
        self.actors_surface.set_colorkey(COLOR_BLACK)

        self.waypoints_surface = pygame.Surface((self.map_image.surface.get_width(),self.map_image.surface.get_height()))
        self.waypoints_surface.set_colorkey(COLOR_BLACK)

        scaled_original_size = self.original_surface_size * (1.0 / 0.7)
        self.lead_actor_surface = pygame.Surface((scaled_original_size,scaled_original_size)).convert()

        self.result_surface = pygame.Surface((self.surface_size,self.surface_size)).convert()
        self.result_surface.set_colorkey(COLOR_BLACK)


        weak_self = weakref.ref(self)
        self.world.on_tick(lambda timestamp: HUD.on_world_tick(weak_self, timestamp))

    @staticmethod
    def on_world_tick(weak_self,timestamp):
        self = weak_self()
        if not self:
            return
        self.server_clock.tick()

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

    def tick(self,clock):
        actors = self.world.get_actors()

        self.actors_with_transforms = [(actor,actor.get_transform()) for actor in actors]

        if self.lead_actor is not None:
            self.lead_actor_transform = self.lead_actor.get_transform()


    def split_actors(self):
        vehicles = []
        walkers = []

        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0] # (actor,actor.get_transform())

            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)

            elif 'walker.pedestrian' in actor.type_id:
                walkers.append(actor_with_transform)

        return (vehicles, walkers)


    def render_walkers(self,surface,list_walker,world_to_pixel):

        for w in list_walker:
            color = COLOR_RED

            # compute bounding box point
            bb = w[0].bounding_box.extent # actor bounding box point
            corners = [carla.Location(x=-bb.x, y=-bb.y), carla.Location(x=bb.x, y=-bb.y),
                       carla.Location(x=bb.x, y=bb.y), carla.Location(x=-bb.x, y=bb.y)]

            w[1].transform(corners) # transform to global coordinate sys
            corners = [world_to_pixel(p) for p in corners] # for render pygame
            pygame.draw.polygon(surface,color,corners)


    def render_vehicles(self,surface,list_vehicle,world_to_pixel):

        for v in list_vehicle:
            color = COLOR_SCARLET_RED_1

            # 오토바이, 자전거
            if int(v[0].attributes['number_of_wheels'])==2:
                color = COLOR_SCARLET_RED_2

                bb = v[0].bounding_box.extent
                corners = [carla.Location(x=-bb.x, y=-bb.y - 0.1),
                           carla.Location(x=bb.x, y=-bb.y - 0.1),
                           carla.Location(x=bb.x + 0.3, y=0),
                           carla.Location(x=bb.x, y=bb.y + 0.1),
                           carla.Location(x=-bb.x, y=bb.y + 0.1),
                           carla.Location(x=-bb.x, y=-bb.y - 0.1)]
            else:
                if v[0].attributes['role_name'] == 'lead_actor':
                    color = COLOR_BLUE

                # Compute bounding box points
                bb = v[0].bounding_box.extent
                corners = [carla.Location(x=-bb.x, y=-bb.y),
                           carla.Location(x=bb.x, y=-bb.y),
                           carla.Location(x=bb.x + 0.3, y=0),
                           carla.Location(x=bb.x, y=bb.y),
                           carla.Location(x=-bb.x, y=bb.y),
                           carla.Location(x=-bb.x, y=-bb.y)]

            v[1].transform(corners) # transform to global coordinate sys
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)



    def render_points(self, surface, colors, transform, radius_in_pix, world_to_pixel):

        location = world_to_pixel(transform.location)
        pygame.draw.circle(surface, colors[0], location, radius_in_pix, 0)

        location = world_to_pixel(self.lead_actor.get_location())
        pygame.draw.circle(surface, colors[1], location, radius_in_pix, 0)


    def render_actors(self,surface,vehicles,walkers):

        self.render_vehicles(surface,vehicles,self.map_image.world_to_pixel)
        self.render_walkers(surface,walkers,self.map_image.world_to_pixel)
        self.render_points(surface,(COLOR_GREEN,COLOR_BLUE),self.target_transform,10,self.map_image.world_to_pixel)


    def render_waypoints(self,surface,waypoints,world_to_pixel):

        color = COLOR_CYAN
        corners=[]

        for p in waypoints:
            corners.append(carla.Location(x=p[0].transform.location.x,
                                          y=p[0].transform.location.y))

        corners = [world_to_pixel(p) for p in corners]

        for c in corners:
            pygame.draw.circle(surface,color,c,self.pixels_per_meter*1.5)

    def update_HUD(self):

        self.tick(self.server_clock)

        if self.actors_with_transforms is None:
            print("\nself.actors with transform is NONE\n")
            return

        self.result_surface.fill(COLOR_BLACK)

        vehicles,walkers = self.split_actors()

        # render waypoints

        self.waypoints_surface.fill(COLOR_BLACK)
        if self.waypoints is not None:
            self.render_waypoints(self.waypoints_surface,self.waypoints,self.map_image.world_to_pixel)


        # render actor
        self.actors_surface.fill(COLOR_BLACK)
        self.render_actors(self.actors_surface,vehicles,walkers)

        # blits surfaces
        surfaces = ((self.map_image.surface,(0,0)),(self.waypoints_surface,(0,0)),(self.actors_surface,(0,0)))


        if self.lead_actor is None:
            angle =0.0
        else:
            angle = self.lead_actor_transform.rotation.yaw+90.0

        lead_actor_location_screen = self.map_image.world_to_pixel(self.lead_actor_transform.location)

        lead_actor_front = self.lead_actor_transform.get_forward_vector()

        """
        lead actor(main vehicle)을 surface 중심으로 맞추고 lead_front 전방 벡터 방향을 이용하여 위치 조정
        
                center = (self.surface.get_width()/2,
                  self.surface.get_height()/2)
        """
        translation_offset = (lead_actor_location_screen[0]-self.lead_actor_surface.get_width()/2+lead_actor_front.x*self.pixels_ahead_vehicle,
                              lead_actor_location_screen[1]-self.lead_actor_surface.get_height()/2+lead_actor_front.y*self.pixels_ahead_vehicle)


        Util.bilts(self.result_surface,surfaces)

        # lead actor surface setting
        self.lead_actor_surface.fill(COLOR_ALUMINIUM_4)

        # (-translation_offset[0],-translation_offset[1])의 오프셋 적용하여 lead actor surface 중심에 위치 하게 한다
        self.lead_actor_surface.blit(self.result_surface,(-translation_offset[0],-translation_offset[1]))

        # 주변 맵 정보가 차량의 진행 방향에 맞춰서 회전하여 화면 생성하기 위함이다
        rotated_result_surface = pygame.transform.rotate(self.lead_actor_surface,angle).convert()

        center = (self.surface.get_width()/2,
                  self.surface.get_height()/2)

        rotation_pivot = rotated_result_surface.get_rect(center=center)
        self.surface.blit(rotated_result_surface,rotation_pivot)

        self.render_points(self.result_surface, (COLOR_GREEN,COLOR_BLUE),self.target_transform, 100, self.map_image.world_to_pixel)
        self.surface_global.blit(pygame.transform.smoothscale(self.result_surface,self.display_size),(0,0))

        self.measure_data = np.array(pygame.surfarray.array3d(self.surface)).swapaxes(0,1) # HWC

