"""

# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


A star algorithm with networkx lib

"""

import math
import numpy as np
import networkx as nx
from enum import Enum

import carla

def vector(l1,l2):
    """

    :param l1: location object
    :param l2: location object
    :return: unit vector from l1 and l2

    """
    x = l2.x - l1.x
    y = l2.y - l1.y
    z = l2.z - l1.z

    norm = np.linalg.norm([x,y,z]) + np.finfo(float).eps # 0을 방지하기 위함

    return [x/norm,y/norm,z/norm] # return unit vector among l1,l2


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class RoutePlanner(object):
    def __init__(self,map,resolution):

        self.resolution = resolution
        self.map = map
        self.topology = None
        self.graph = None
        self.id_map = None
        self.road_id_to_edge = None

        self.intersection_node = -1
        self.prev_decision = RoadOption.VOID

        # build graph
        self.build_topology()
        self.build_graph()
        self.find_loose_end()
        self.lane_change_link()


    def build_topology(self):
        """

        entry : calral waypoint
        entryxyz : tuple(x y z)

        exit
        exitxyz

        path : entry와 exit 사이의 waypoint이다 정한 resoultion 만큼 나눈다

        :return: void
        """
        self.topology=[]

        # retrieving waypoints to construct detailed topology

        for segment in self.map.get_topology():
            w1,w2 = segment[0],segment[1]
            l1,l2 = w1.transform.location, w2.transform.location

            # Rounding off floating point -> 단순 보정 단계
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)

            w1.transform.location,w2.transform.location = l1,l2


            seg_dict = dict()
            seg_dict['entry'],seg_dict['exit'] = w1,w2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] =[]

            endloc = w2.transform.location # exit waypoint location

            # w1 부터 endloc까지의 유클리드안 거리가 설정한 해상도보다 크면 해상도만큼 나눈다
            if w1.transform.location.distance(endloc) > self.resolution:
                w = w1.next(self.resolution)[0] # w1에서 resolution 거리만큼의 list waypoint 반환한다

                while w.transform.location.distance(endloc) >self.resolution:
                    seg_dict['path'].append(w)
                    w = w.next(self.resolution)[0]

            else:
                seg_dict['path'].append(w1.next(self.resolution)[0])


            self.topology.append(seg_dict)



    def build_graph(self):
        """
        grahp : networkx DiGraph

        vertex : (x y z) position in world map
        edge :
            entry vector : unit vector tangent at entry point
            exit vector: unit vector tangent at exit point
            net vector : entry to exit unit vector
            intersection : 교차 여부 T/F

        id_map :dict
        road_id_to_edge :dict
        :return: void
        """

        self.graph = nx.DiGraph()
        self.id_map = dict()  # Map with structure {(x,y,z): id, ... }
        self.road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }

        for segement in self.topology:
            entry_xyz,exit_xyz = segement['entryxyz'],segement['exitxyz']

            path = segement['path']

            entry_wp,exit_wp = segement['entry'],segement['exit']

            intersection = entry_wp.is_junction # entry wp가 교차로에 있는지 아닌지 T/F

            road_id,section_id,lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id


            # vertex is (x y z)
            for vertex in entry_xyz,exit_xyz:
                # structure {(x,y,z): id, ... }
                if vertex not in self.id_map:
                    new_id = len(self.id_map) # self.id map 길이를 id로 사용한다 0,1,2,....순차적으로 노드 확장함
                    self.id_map[vertex] =new_id
                    self.graph.add_node(new_id,vertex=vertex)

            # entry id,  exit id
            n1 = self.id_map[entry_xyz]
            n2 = self.id_map[exit_xyz]

            if road_id not in self.road_id_to_edge:
                self.road_id_to_edge[road_id] = dict()

            if section_id not in self.road_id_to_edge[road_id]:
                self.road_id_to_edge[road_id][section_id] =dict()

            self.road_id_to_edge[road_id][section_id][lane_id] = (n1,n2)

            # 객체가 어디를 향하는지 방향을 알아낸다 carla.Vector3D
            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()


            # adding edge with attribute
            #length가 edge의 비용이다
            self.graph.add_edge(
                n1,n2,
                length=len(path)+1,path=path,
                entry_waypoint = entry_wp,exit_waypoint = exit_wp,
                entry_vector = np.array([entry_carla_vector.x,entry_carla_vector.y,entry_carla_vector.z]),
                exit_vector = np.array([exit_carla_vector.x,exit_carla_vector.y,exit_carla_vector.z]),
                net_vector = vector(entry_wp.transform.location,exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW
            )
    def localize(self,location):
        """
        location에서 어떤 도로구간에 속해있는지 찾고 해당하는 도로 정보를 edge로 반환한다

        :param location:
        :return: edge
        """
        waypoint = self.map.get_waypoint(location) # location위치상에 가까운 웨이포인트
        edge = None

        try:
            edge = self.road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            print("EXCEPTION ERROR IN LOCALIZE FUNCTION")
            pass

        return edge

    def find_loose_end(self):
        """
        연결이 끊긴 웨이포인트 이어주는 함수
        :return:
        """
        count_loose_ends = 0
        hop_resolution = self.resolution
        for segment in self.topology:
            end_wp = segment['exit']
            exit_xyz = segment['exitxyz']
            road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id
            if road_id in self.road_id_to_edge and section_id in self.road_id_to_edge[road_id] and lane_id in self.road_id_to_edge[road_id][section_id]:
                pass
            else:
                # unconnect 부분있으면 연결해주고 edge추가 한다
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
                while next_wp is not None and next_wp and next_wp[0].road_id == road_id and next_wp[0].section_id == section_id and next_wp[0].lane_id == lane_id:
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)
                if path:
                    n2_xyz = (path[-1].transform.location.x,
                              path[-1].transform.location.y,
                              path[-1].transform.location.z)
                    self.graph.add_node(n2, vertex=n2_xyz)
                    self.graph.add_edge(
                        n1, n2,
                        length=len(path) + 1, path=path,
                        entry_waypoint=end_wp, exit_waypoint=path[-1],
                        entry_vector=None, exit_vector=None, net_vector=None,
                        intersection=end_wp.is_junction, type=RoadOption.LANEFOLLOW)

    def lane_change_link(self):
        """
        차선 변경 가능한 곳에 비용 0인 연결 edge를 추가 하는 함수이다
        :return:
        """
        for segment in self.topology:
            left_found, right_found = False, False

            for waypoint in segment['path']:
                # 교차로 지점이 아니라면
                if not segment['entry'].is_junction:
                    next_waypoint, next_road_option, next_segment = None, None, None

                    # 오른쪽 차선 사용 가능 조건
                    if waypoint.right_lane_marking.lane_change & carla.LaneChange.Right and not right_found:
                        next_waypoint = waypoint.get_right_lane() # 오른쪽 차선 변경 가능하면 오른쪽 방향의 waypoint 반환한다

                        if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANERIGHT
                            next_segment = self.localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                # length=0 : zero cost link
                                self.graph.add_edge(
                                    self.id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint)
                                right_found = True

                    # 왼쪽 차선 변경 사용 조건
                    if waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and not left_found:
                        next_waypoint = waypoint.get_left_lane()
                        if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANELEFT
                            next_segment = self.localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                self.graph.add_edge(
                                    self.id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint)
                                left_found = True
                if left_found and right_found:
                    break

    def distanc_heuristic(self,n1,n2):
        """
        A*알고리듬에서 유클리드안 거리를 휴리스틱 함수로 사용한다
        :param n1:
        :param n2:
        :return:
        """
        l1 = np.array(self.graph.nodes[n1]['vertex'])
        l2 = np.array(self.graph.nodes[n2]['vertex'])
        return np.linalg.norm(l1-l2)

    def A_star_search(self,origin,destination):
        """
        networkx에서 제공하는 a star algorithm 사용한다
        :param origin: carla.location object of start position
        :param destination: carla.location object of end position
        :return: list of node id Path
        """
        start, end = self.localize(origin), self.localize(destination)
        route = nx.astar_path(
            self.graph, source=start[0], target=end[0],
            heuristic=self.distanc_heuristic, weight='length')

        route.append(end[1]) # 마지막 id 추가
        return route
    def sucessive_last_intersection_edge(self,index,route):
        """
        index부터 교차로의 연속된 부분을 보고 마지막 지점을 반환한다

        :param index:
        :param route:
        :return:
        """
        last_intersection_edge = None
        last_node = None

        route_list = [] # index부터 시작해서 route 끝까지 전반적으로 본다
        for i in range(index,len(route)-1):
            route_list.append((route[i],route[i+1]))

        for node1,node2 in route_list:
            candidate_edge = self.graph.edges[node1,node2]

            # index위치를 last intersection edge로 설정
            if node1 == route[index]:
                last_intersection_edge =candidate_edge

            if candidate_edge['type'] == RoadOption.LANEFOLLOW and candidate_edge['intersection']:
                last_intersection_edge = candidate_edge
                last_node = node2
            else:
                break

        return last_node,last_intersection_edge

    def turn_decision(self,index,route,threshold=math.radians(35)):
        """
        회전을 결정하는 함수 RoadOption 정한다
        :param index:
        :param route:
        :param threshold:
        :return: RoadOption
        """
        decision = None
        # prev cur next node id
        previous_node = route[index-1]
        current_node = route[index]
        next_node = route[index+1]
        next_edge = self.graph.edges[current_node,next_node] # cur to next edge

        if index > 0:
            # 교차로 상태라면 이전 결정 계속 이어간다
            if self.prev_decision != RoadOption.VOID and self.intersection_node >0 and self.intersection_node != previous_node and next_edge['type']==RoadOption.LANEFOLLOW and next_edge['intersection']:
                decision = self.prev_decision

            else:
                # reset intersection node
                self.intersection_node = -1

                current_edge = self.graph.edges[previous_node,current_node] # prev to cur edge

                # cur edge가 교차로 진입전이고 lanefollow라는 조건
                caculate_turn = current_edge['type'] == RoadOption.LANEFOLLOW and not current_edge['intersection'] and next_edge['type'] == RoadOption.LANEFOLLOW and next_edge['intersection']

                if caculate_turn:
                    last_node, tail_edge = self.sucessive_last_intersection_edge(index,route)
                    self.intersection_node = last_node
                    if tail_edge is not None:
                        next_edge = tail_edge

                    cv,nv = current_edge['exit_vector'],next_edge['exit_vector']

                    if cv is None or nv is None:
                        return next_edge['type']

                    cross_list =[]
                    # current node부터 연결있는 노드 본다
                    for neighbor in self.graph.successors(current_node):

                        select_edge = self.graph.edges[current_node,neighbor]
                        if select_edge['type'] == RoadOption.LANEFOLLOW:
                            # current node와 이웃한 노드 중에서 경로상에 없는
                            # 노드를 추가 해서 cross product계산하고 list추가한다
                            if neighbor != route[index+1]:
                                sv = select_edge['net_vector']
                                cross_list.append(np.cross(cv,sv)[2])

                    # carla sim의 좌표계는 왼손법칙(left hand system)이다 따라서 왼손법칙에 따라 양수면 우회전 , 음수면 좌회전이다
                    next_cross = np.cross(cv,nv)[2]

                    # radian angle
                    # deviation작을수록 두 벡터 일치 하는거고 클수록 어긋나는것이다 즉 크면 방향전환 필요한것이다
                    deviation = math.acos(np.clip(
                        np.dot(cv,nv)/(np.linalg.norm(cv)*np.linalg.norm(nv)),-1.0,1.0
                    ))

                    if not cross_list:
                        cross_list.append(0)

                    if deviation < threshold:
                        decision = RoadOption.STRAIGHT

                    # cross list는 인근 주변의 노드의 cross product계산값인데
                    # 이거보다 next cross가 작다는건 더 왼쪽으로 이동해야함을 의미한다
                    elif cross_list and next_cross < min(cross_list):
                        decision = RoadOption.LEFT
                    elif cross_list and next_cross > max(cross_list):
                        decision = RoadOption.RIGHT

                    elif next_cross < 0:
                        decision = RoadOption.LEFT
                    elif next_cross > 0:
                        decision = RoadOption.RIGHT

                else:
                    decision = next_edge['type']

        else:
            decision = next_edge['type']

        self.prev_decision = decision
        return decision




    def closest_in_list(self,current_waypoint,waypoint_list):
        """
        현재 waypoint에서 가장 가까운 거리 index반환
        주어진 현재 위치에서 가장 가까운 지점 찾기 위함이다

        :param current_waypoint:
        :param waypoint_list:
        :return:
        """
        min_distance = float('inf')
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(
                current_waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index


    def trace_route(self, origin,destination):
        """

        :param origin:
        :param destination:
        :return: List [tuple(carla.waypoint, RoadOption)]
        """

        route_trace=[]
        route = self.A_star_search(origin,destination)
        current_waypoint = self.map.get_waypoint(origin)
        destination_waypoint = self.map.get_waypoint(destination)


        for i in range(len(route)-1):
            road_option = self.turn_decision(i,route)
            edge = self.graph.edges[route[i],route[i+1]] # i to i+1 edge
            path =[]

            if edge['type'] != RoadOption.LANEFOLLOW and edge['type'] !=RoadOption.VOID:
                route_trace.append((current_waypoint,road_option)) # (carla.waypoint, RoadOption)
                exit_wp = edge['exit_waypoint'] # waypoint
                n1,n2 = self.road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
                next_edge = self.graph.edges[n1,n2] # n1 to n2 edge

                if next_edge['path']:
                    closest_index = self.closest_in_list(current_waypoint,next_edge['path'])
                    closest_index = min(len(next_edge['path'])-1,closest_index+5)

                    # current waypoint update
                    current_waypoint = next_edge['path'][closest_index]
                else:
                    current_waypoint = next_edge['exit_waypoint']


                route_trace.append((current_waypoint, road_option))




            else:

                path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                closest_index = self.closest_in_list(current_waypoint,path)
                #print(f"\ndebug path is {path}\n\n closest_index is {closest_index}\n\n path[closest_index] is {path[closest_index]}\n\n")
                for waypoint in path[closest_index:]:

                    current_waypoint = waypoint
                    route_trace.append((current_waypoint,road_option))

                    # route 끝 지점 종료 조건
                    if len(route)-i <= 2 and waypoint.transform.location.distance(destination) < 2*self.resolution:
                        break
                    elif len(route)-i <= 2 and current_waypoint.road_id == destination_waypoint.road_id and current_waypoint.section_id == destination_waypoint.section_id and current_waypoint.lane_id == destination_waypoint.lane_id:
                        destination_index = self.closest_in_list(destination_waypoint, path)
                        if closest_index > destination_index:
                            break



        return route_trace