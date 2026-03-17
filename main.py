import sys
import math
from collections import defaultdict, deque

def read_input():
    input_lines = []
    while True:
        try:
            line = input()
            if not line:
                break
            input_lines.append(line)
        except EOFError:
            break
    return input_lines

def parse_input(input_lines):
    idx = 0
    
    N = int(input_lines[idx])
    idx += 1
    
    servers = []
    for _ in range(N):
        # 修复了 map() 使用方式
        gi, ki, mi = map(int, input_lines[idx].split())
        servers.append({'gi': gi, 'ki': ki, 'mi': mi})
        idx += 1
    
    M = int(input_lines[idx])
    idx += 1
    
    users = []
    for _ in range(M):
        si, ei, cnti = map(int, input_lines[idx].split())
        users.append({'si': si, 'ei': ei, 'cnti': cnti, 'remaining': cnti, 'tasks': [], 'last_server': None})
        idx += 1
    
    latencies = []
    for _ in range(M):
        row = list(map(int, input_lines[idx].split()))
        latencies.append(row)
        idx += 1
    print(idx)
    print(input_lines)
    a, b = map(int, input_lines[idx-1].split())
    
    return {
        'N': N,
        'servers': servers,
        'M': M,
        'users': users,
        'latencies': latencies,
        'a': a,
        'b': b
    }

def calculate_processing_time(Bj, server_ki, Bj_size):
    """计算处理时间"""
    return math.ceil(f_Bj(Bj, server_ki) * Bj_size)

def f_Bj(Bj, ki):
    """计算推理速度"""
    return ki * Bj

def can_fit_batch(server, Bj, a, b):
    """检查batch是否适合显存"""
    return a * Bj + b <= server['mi']

def schedule_tasks(data):
    users = data['users']
    servers = data['servers']
    latencies = data['latencies']
    a, b = data['a'], data['b']

    # 为每个服务器的每个NPU维护一个队列
    server_queues = [
        {'npus': [{'queue': [], 'current_time': 0} for _ in range(server['gi'])]} 
        for server in servers
    ]

    for user_idx, user in enumerate(users):
        current_time = user['si']
        remaining_samples = user['remaining']
        last_used_server = None
        
        print(f"[DEBUG] Starting scheduling for User {user_idx}, total samples: {remaining_samples}")

        while remaining_samples > 0:
            # 找到最大合法的 batchsize（从大到小尝试）
            max_batch_possible = min(remaining_samples, 1000)
            best_batch = 0
            best_server_idx = -1
            best_npu_idx = -1
            best_finish_time = float('inf')

            # 从大到小尝试 batchsize
            for Bj in range(max_batch_possible, 0, -1):
                if not can_fit_batch(servers[0], Bj, a, b):
                    continue  # 跳过不合法的 batchsize

                # 寻找最适合的服务器/NPU组合
                for server_idx, server in enumerate(servers):
                    if not can_fit_batch(server, Bj, a, b):
                        continue
                    
                    latency = latencies[user_idx][server_idx]
                    arrival_time = current_time + latency
                    
                    processing_time = calculate_processing_time(Bj, server['ki'], Bj)

                    # 查找该服务器中最空闲的 NPU
                    for npu_idx in range(server['gi']):
                        queue = server_queues[server_idx]['npus'][npu_idx]['queue']
                        npu_current_time = server_queues[server_idx]['npus'][npu_idx]['current_time']

                        # 模拟当前队列完成时间
                        last_completion_time = npu_current_time
                        for req in queue:
                            last_completion_time = max(last_completion_time, req['start_time'] + req['processing_time'])

                        start_time = max(arrival_time, last_completion_time)
                        finish_time = start_time + processing_time

                        if finish_time < best_finish_time:
                            best_finish_time = finish_time
                            best_batch = Bj
                            best_server_idx = server_idx
                            best_npu_idx = npu_idx
            
            # 如果找不到合适的服务器和 batchsize
            if best_batch == 0:
                print(f"[WARNING] No valid batch/server found for user {user_idx}, waiting...")
                
                # 等待一小段时间再重试
                current_time += 1
                
                # 防止无限等待
                if current_time > 100000:
                    print(f"[ERROR] Exceeded max time for user {user_idx}, forcing batch=1")
                    best_batch = 1
                    best_server_idx = 0
                    best_npu_idx = 0
                continue
            
            # 分配任务
            best_server = servers[best_server_idx]
            latency = latencies[user_idx][best_server_idx]
            arrival_time = current_time + latency
            
            # 计算开始时间和处理时间
            queue = server_queues[best_server_idx]['npus'][best_npu_idx]['queue']
            npu_current_time = server_queues[best_server_idx]['npus'][best_npu_idx]['current_time']
            
            last_completion_time = npu_current_time
            for req in queue:
                last_completion_time = max(last_completion_time, req['start_time'] + req['processing_time'])
            
            start_time = max(arrival_time, last_completion_time)
            processing_time = calculate_processing_time(best_batch, best_server['ki'], best_batch)
            
            # 添加请求到队列
            server_queues[best_server_idx]['npus'][best_npu_idx]['queue'].append({
                'user_idx': user_idx,
                'start_time': start_time,
                'arrival_time': arrival_time,
                'processing_time': processing_time,
                'batch_size': best_batch
            })
            
            # 更新 NPU 当前时间
            server_queues[best_server_idx]['npus'][best_npu_idx]['current_time'] = start_time + processing_time

            # 更新用户状态
            user['tasks'].append({
                'time': current_time,
                'server': best_server_idx + 1,
                'npu': best_npu_idx + 1,
                'batchsize': best_batch
            })

            remaining_samples -= best_batch
            current_time = start_time + processing_time  # 移动到任务完成时间
            
            # 更新用户的最后使用服务器
            users[user_idx]['last_server'] = best_server_idx
            
            print(f"[DEBUG] Scheduled batch {best_batch} for User {user_idx} on Server {best_server_idx}, NPU {best_npu_idx}, Remaining: {remaining_samples}")
    
    return users

def generate_output(users):
    result = []
    for user in users:
        result.append(str(len(user['tasks'])))
        for task in user['tasks']:
            result.append(f"{task['time']} {task['server']} {task['npu']} {task['batchsize']}")
    return result

def main():
    input_lines = read_input()
    
    # 调试输出：显示读取到的每一行
    # print("[DEBUG] Read input lines:")
    # for i, line in enumerate(input_lines):
    #     print(f"Line {i}: {line}")
    # print()

    data = parse_input(input_lines)
    print("[DEBUG] Read input lines1:")
    scheduled_users = schedule_tasks(data)
    print("[DEBUG] Read input lines2:")
    output = generate_output(scheduled_users)
    print('\n'.join(output))

if __name__ == "__main__":
    main()