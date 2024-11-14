import carla
import pygame
from carla_rl_env.hud import MapImage

# CARLA 클라이언트 연결
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Town10HD 맵 로드
world = client.load_world('Town10HD')
carla_map = world.get_map()


def draw_spawn_points(surface, spawn_points, world_to_pixel, scale_x, scale_y, selected_indices):
    """스폰 포인트를 미니맵에 그리는 함수"""
    default_color = pygame.Color(255, 0, 0)  # 기본 빨간색
    selected_color = pygame.Color(0, 255, 0)  # 선택된 색 (초록색)

    for index, spawn_point in enumerate(spawn_points):
        # 스폰 포인트의 월드 좌표를 픽셀 좌표로 변환
        pixel_pos = world_to_pixel(spawn_point.location)
        scaled_pixel_pos = [int(pixel_pos[0] * scale_x), int(pixel_pos[1] * scale_y)]

        # 색상 선택
        color = selected_color if index in selected_indices else default_color

        # 스케일이 적용된 좌표에 스폰 포인트 그리기
        pygame.draw.circle(surface, color, scaled_pixel_pos, 5)  # 원의 크기를 5로 설정


def get_clicked_locations(carla_map):
    # Pygame 초기화
    pygame.init()
    screen = pygame.display.set_mode((800, 800))

    # 픽셀 당 미터 설정
    PIXELS_PER_METER = 12
    map_image = MapImage(world, carla_map, PIXELS_PER_METER)

    spawn_points = carla_map.get_spawn_points()

    map_width = map_image.surface.get_width()
    map_height = map_image.surface.get_height()
    scale_x = 800 / map_width
    scale_y = 800 / map_height

    running = True
    selected_indices = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                map_x = mouse_x / scale_x
                map_y = mouse_y / scale_y
                clicked_world_location = map_image.pixel_to_world((map_x, map_y))

                closest_spawn_point = None
                closest_distance = float('inf')

                for index, spawn_point in enumerate(spawn_points):
                    distance = clicked_world_location.distance(spawn_point.location)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_spawn_point = spawn_point
                        closest_index = index

                # 클릭한 위치와 가까운 스폰 포인트가 5m 이내일 때
                if closest_spawn_point and closest_distance < 5.0:
                    if closest_index not in selected_indices:
                        selected_indices.append(closest_index)
                        print(f"선택된 스폰 포인트의 CARLA 좌표: {closest_spawn_point.location}")

                # 두 개의 스폰 포인트가 선택되면 종료
                if len(selected_indices) == 2:
                    running = False

        screen.fill((0, 0, 0))  # 검은색 배경으로 초기화
        scaled_map_surface = pygame.transform.scale(map_image.surface, (800, 800))
        screen.blit(scaled_map_surface, (0, 0))

        # 스폰 포인트 그리기
        draw_spawn_points(screen, spawn_points, map_image.world_to_pixel, scale_x, scale_y, selected_indices)

        # Pygame 화면 업데이트
        pygame.display.flip()

    pygame.quit()

    # 선택된 스폰 포인트의 위치를 리스트로 반환
    selected_locations = [spawn_points[i] for i in selected_indices]
    return selected_locations  # 튜플 대신 리스트로 반환


# 함수 호출 및 결과 출력
locations = get_clicked_locations(carla_map)
print(f"반환된 CARLA 좌표: {locations}")
