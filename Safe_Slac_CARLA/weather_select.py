import carla
import tkinter as tk
from tkinter import ttk

# CARLA 클라이언트 연결
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Town10HD 맵 로드
world = client.get_world()
carla_map = world.get_map()

# 날씨 이름 배열
weather_options = [
    "Default", "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
    "MidRainyNoon", "HardRainNoon", "SoftRainNoon", "ClearSunset", "CloudySunset",
    "WetSunset", "WetCloudySunset", "MidRainSunset", "HardRainSunset", "SoftRainSunset",
    "ClearNight", "CloudyNight", "WetNight", "WetCloudyNight", "SoftRainNight", "MidRainyNight",
    "HardRainNight"
]

# 선택된 날씨를 저장할 변수
selected_weather = None

def apply_weather():
    """선택된 날씨를 적용하는 함수"""
    selected_weather_name = selected_weather.get()
    try:
        weather = getattr(carla.WeatherParameters, selected_weather_name)
        world.set_weather(weather)
        print(f"선택된 날씨: {selected_weather_name}")
    except AttributeError:
        print(f"날씨 옵션 '{selected_weather_name}'이 올바르지 않습니다.")

def get_weather_selection(options):
    """Tkinter GUI를 생성하여 날씨 옵션을 선택하게 하는 함수"""
    # Tkinter 루트 윈도우 생성
    root = tk.Tk()
    root.title("Weather Selection")

    # 기본 스타일 설정
    style = ttk.Style()
    style.configure("TButton", font=('Sans', 14), padding=10)
    style.configure("TCombobox", font=('Sans', 14), padding=10)
    style.configure("TLabel", font=('Sans', 16))

    global selected_weather
    selected_weather = tk.StringVar(value=options[0])  # 기본값으로 첫 번째 옵션 선택

    # 라벨 추가
    label = ttk.Label(root, text="Select Weather:", padding=(10, 10))
    label.pack(pady=10)

    # 드롭다운 메뉴 생성
    weather_menu = ttk.Combobox(root, textvariable=selected_weather, values=options, width=30)
    weather_menu.pack(pady=20)

    # 선택 완료 버튼 생성
    select_button = ttk.Button(root, text="Select", command=apply_weather)
    select_button.pack(pady=10)

    # 윈도우 크기 조정
    root.geometry("600x300")

    # Tkinter 메인 루프 시작
    root.mainloop()

# GUI 실행
get_weather_selection(weather_options)
