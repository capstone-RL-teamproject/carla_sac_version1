import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnts', type=int, default=1)

    args = parser.parse_args()

    max_cnts = 5
    processes = []

    if args.cnts <= max_cnts:
        for cnt in range(args.cnts):
            print(3000 + cnt*100)
            command = f"./CarlaUE4.sh -carla-port={3000 + cnt*100}"

            # 병렬 실행을 위해 subprocess.Popen 사용
            process = subprocess.Popen(command, shell=True)
            processes.append(process)

        # 모든 프로세스가 끝날 때까지 대기
        for process in processes:
            process.wait()
    else:
        print("cnts is too big")

if __name__ == '__main__':
    main()