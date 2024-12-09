import pygame

# 초기화
pygame.init()

# 창 크기 설정
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Pygame Circle Example")

# 색상 설정 (RGB)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# 원의 위치와 크기
circle_center = (320, 240)  # 화면 중앙
circle_radius = 50           # 원의 반지름

# 실행 상태 플래그
running = True

# 게임 루프
while running:
    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 화면 흰색으로 채우기
    screen.fill(WHITE)

    # 원 그리기
    pygame.draw.circle(screen, RED, circle_center, circle_radius)

    # 화면 업데이트
    pygame.display.flip()

# 종료
pygame.quit()