import ray
import numpy as np
import time

# Ray 초기화 (공유 네임스페이스 지정)
ray.init(namespace="parameter_server_namespace")

# 통합을 담당하는 Actor 정의
@ray.remote
class ParameterServer:
    def __init__(self, expected_workers = 2):
        self.collected_weights = []
        self.expected_workers = expected_workers

    def add_weights(self, weights):
        # 워커로부터 받은 파라미터를 저장
        self.collected_weights.append(weights)
        
    def integrate_parameters(self): # 수집된 파라미터가 있다면 통합 수행
       # if not self.collected_weights:
       #    return None
    
        if len(self.collected_weights) < self.expected_workers: 
            print('Not all weights have been collected yet.') 
            return None
        
        averaged_weights = {}
        
        for key in self.collected_weights[0].keys():
            averaged_weights[key] = np.mean([w[key] for w in self.collected_weights], axis=0)
            #print(f"{key}: {averaged_weights[key]}")

        # 통합 완료 후 초기화
        self.collected_weights = []
        return averaged_weights
    
   # def broadcast_weights(seld, weights): #워커에 업데이트 된 가중치 전송
   #     return [worker.update_weights.remote(weights) for worker in workers]

# ParameterServer Actor 생성 (이름과 수명 지정)
central_server = ParameterServer.options(
    name="ParameterServer", lifetime="detached").remote()

# 통합 및 업데이트 예시 (다른 스크립트에서 호출 가능)
def integrate_and_update():
    # 파라미터 통합 요청
    integrated_weights = ray.get(central_server.integrate_parameters.remote()) #워커만큼 실행되지 않으면 NONE을
    print(integrated_weights)
    if integrated_weights:
        print("Integrated weights have been updated.")
    else:
        print("Waiting for more weights")

# 계속 실행되도록 유지
if __name__ == "__main__":
    print("Central server is ready to collect parameters...")
    print("Current actors:", ray.util.list_named_actors())
    while True:
        integrate_and_update()
        time.sleep(1)