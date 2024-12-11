import math
import numpy as np
import networkx as nx
from enum import Enum
import carla


class RoadOption(Enum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class RoutePlanner(object):
    def __init__(self, map, resolution):
        self.resolution = resolution
        self.map = map
        self.topology = None
        self.graph = None
        self.id_map = None
        self.road_id_to_edge = None
        self.intersection_node = -1
        self.prev_decision = RoadOption.VOID
        self.build_topology()
        self.build_graph()
        self.find_loose_end()
        self.lane_change_link()

    def build_topology(self):
        self.topology = []
        for segment in self.map.get_topology():
            w1, w2 = segment[0], segment[1]
            l1, l2 = w1.transform.location, w2.transform.location
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            w1.transform.location, w2.transform.location = l1, l2
            seg_dict = dict()
            seg_dict["entry"], seg_dict["exit"] = w1, w2
            seg_dict["entryxyz"], seg_dict["exitxyz"] = (x1, y1, z1), (x2, y2, z2)
            seg_dict["path"] = []
            endloc = w2.transform.location
            if w1.transform.location.distance(endloc) > self.resolution:
                w = w1.next(self.resolution)[0]
                while w.transform.location.distance(endloc) > self.resolution:
                    seg_dict["path"].append(w)
                    w = w.next(self.resolution)[0]
            else:
                seg_dict["path"].append(w1.next(self.resolution)[0])
            self.topology.append(seg_dict)

    def build_graph(self):
        self.graph = nx.DiGraph()
        self.id_map = dict()
        self.road_id_to_edge = dict()
        for segement in self.topology:
            entry_xyz, exit_xyz = segement["entryxyz"], segement["exitxyz"]
            path = segement["path"]
            entry_wp, exit_wp = segement["entry"], segement["exit"]
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = (
                entry_wp.road_id,
                entry_wp.section_id,
                entry_wp.lane_id,
            )
            for vertex in entry_xyz, exit_xyz:
                if vertex not in self.id_map:
                    new_id = len(self.id_map)
                    self.id_map[vertex] = new_id
                    self.graph.add_node(new_id, vertex=vertex)
            n1 = self.id_map[entry_xyz]
            n2 = self.id_map[exit_xyz]
            if road_id not in self.road_id_to_edge:
                self.road_id_to_edge[road_id] = dict()
            if section_id not in self.road_id_to_edge[road_id]:
                self.road_id_to_edge[road_id][section_id] = dict()
            self.road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()

            def vector(l1, l2):
                x = l2.x - l1.x
                y = l2.y - l1.y
                z = l2.z - l1.z
                norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
                return [x / norm, y / norm, z / norm]

            self.graph.add_edge(
                n1,
                n2,
                length=len(path) + 1,
                path=path,
                entry_waypoint=entry_wp,
                exit_waypoint=exit_wp,
                entry_vector=np.array(
                    [entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]
                ),
                exit_vector=np.array(
                    [exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]
                ),
                net_vector=vector(
                    entry_wp.transform.location, exit_wp.transform.location
                ),
                intersection=intersection,
                type=RoadOption.LANEFOLLOW,
            )

    def localize(self, location):
        waypoint = self.map.get_waypoint(location)
        edge = None
        try:
            edge = self.road_id_to_edge[waypoint.road_id][waypoint.section_id][
                waypoint.lane_id
            ]
        except KeyError:
            pass
        return edge

    def find_loose_end(self):
        count_loose_ends = 0
        hop_resolution = self.resolution
        for segment in self.topology:
            end_wp = segment["exit"]
            exit_xyz = segment["exitxyz"]
            road_id, section_id, lane_id = (
                end_wp.road_id,
                end_wp.section_id,
                end_wp.lane_id,
            )
            if (
                road_id in self.road_id_to_edge
                and section_id in self.road_id_to_edge[road_id]
                and lane_id in self.road_id_to_edge[road_id][section_id]
            ):
                pass
            else:
                count_loose_ends += 1
                if road_id not in self.road_id_to_edge:
                    self.road_id_to_edge[road_id] = dict()
                if section_id not in self.road_id_to_edge[road_id]:
                    self.road_id_to_edge[road_id][section_id] = dict()
                n1 = self.id_map[exit_xyz]
                n2 = -1 * count_loose_ends
                self.road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                next_wp = end_wp.next(hop_resolution)
                path = []
                while (
                    next_wp is not None
                    and next_wp
                    and next_wp[0].road_id == road_id
                    and next_wp[0].section_id == section_id
                    and next_wp[0].lane_id == lane_id
                ):
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)
                if path:
                    n2_xyz = (
                        path[-1].transform.location.x,
                        path[-1].transform.location.y,
                        path[-1].transform.location.z,
                    )
                    self.graph.add_node(n2, vertex=n2_xyz)
                    self.graph.add_edge(
                        n1,
                        n2,
                        length=len(path) + 1,
                        path=path,
                        entry_waypoint=end_wp,
                        exit_waypoint=path[-1],
                        entry_vector=None,
                        exit_vector=None,
                        net_vector=None,
                        intersection=end_wp.is_junction,
                        type=RoadOption.LANEFOLLOW,
                    )

    def lane_change_link(self):
        for segment in self.topology:
            left_found, right_found = False, False
            for waypoint in segment["path"]:
                if not segment["entry"].is_junction:
                    if (
                        waypoint.right_lane_marking.lane_change & carla.LaneChange.Right
                        and not right_found
                    ):
                        next_waypoint = waypoint.get_right_lane()
                        if (
                            next_waypoint is not None
                            and next_waypoint.lane_type == carla.LaneType.Driving
                            and waypoint.road_id == next_waypoint.road_id
                        ):
                            next_segment = self.localize(
                                next_waypoint.transform.location
                            )
                            if next_segment is not None:
                                self.graph.add_edge(
                                    self.id_map[segment["entryxyz"]],
                                    next_segment[0],
                                    entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint,
                                    intersection=False,
                                    exit_vector=None,
                                    path=[],
                                    length=0,
                                    type=RoadOption.CHANGELANERIGHT,
                                    change_waypoint=next_waypoint,
                                )
                                right_found = True
                    if (
                        waypoint.left_lane_marking.lane_change & carla.LaneChange.Left
                        and not left_found
                    ):
                        next_waypoint = waypoint.get_left_lane()
                        if (
                            next_waypoint is not None
                            and next_waypoint.lane_type == carla.LaneType.Driving
                            and waypoint.road_id == next_waypoint.road_id
                        ):
                            next_segment = self.localize(
                                next_waypoint.transform.location
                            )
                            if next_segment is not None:
                                self.graph.add_edge(
                                    self.id_map[segment["entryxyz"]],
                                    next_segment[0],
                                    entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint,
                                    intersection=False,
                                    exit_vector=None,
                                    path=[],
                                    length=0,
                                    type=RoadOption.CHANGELANELEFT,
                                    change_waypoint=next_waypoint,
                                )
                                left_found = True
                if left_found and right_found:
                    break

    def distanc_heuristic(self, n1, n2):
        l1 = np.array(self.graph.nodes[n1]["vertex"])
        l2 = np.array(self.graph.nodes[n2]["vertex"])
        return np.linalg.norm(l1 - l2)

    def A_star_search(self, origin, destination):
        start, end = self.localize(origin), self.localize(destination)
        route = nx.astar_path(
            self.graph,
            source=start[0],
            target=end[0],
            heuristic=self.distanc_heuristic,
            weight="length",
        )
        route.append(end[1])
        return route

    def trace_route(self, origin, destination):
        route_trace = []
        route = self.A_star_search(origin, destination)
        current_waypoint = self.map.get_waypoint(origin)
        destination_waypoint = self.map.get_waypoint(destination)
        for i in range(len(route) - 1):
            # 단순화, 원본 로직 그대로
            current_waypoint = destination_waypoint
            route_trace.append((current_waypoint, RoadOption.LANEFOLLOW))
        return route_trace
