import os
import json
import av
def find_video_file(folder_path, target_filename):
    """
    遍历文件夹及其子文件夹，查找名为 target_filename 的视频文件。
    :param folder_path: 文件夹路径
    :param target_filename: 要查找的目标文件名
    :return: 找到的文件路径，若未找到则返回 None
    """
    for root, dirs, files in os.walk(folder_path):
        if target_filename in files:
            # 打印目标文件名
            print(f"找到目标文件: {target_filename}")
            
            # 如果是 bm_teaser.mp4，打印完整路径
            if target_filename == "bm_teaser.mp4":
                print(f"bm_teaser.mp4 的完整路径: {os.path.join(root, target_filename)}")
            
            # 返回完整路径
            return os.path.join(root, target_filename)
    
    # 如果未找到文件，返回 None
    print(f"未找到目标文件: {target_filename}")
    return None

def find_videos_from_json(folder_path, json_file_path):
    """
    从 JSON 文件中读取目标文件名，并在指定文件夹中查找这些文件。
    :param folder_path: 要查找的文件夹路径
    :param json_file_path: JSON 文件的路径
    :return: 一个字典，键为目标文件名，值为找到的文件路径（未找到则为 None）
    """
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 存储结果
    results = {}
    
    # 遍历 JSON 数据，提取目标文件名并查找文件
    for item in data:
        target_filename = item.get("video")
        if target_filename==None:
            print("rrr")
        if target_filename:
            found_path = find_video_file(folder_path, target_filename)
            if found_path==None:
                print("ttt")
            results[target_filename] = found_path
        else:
            results[target_filename] = None
    
    return results

# 示例使用
folder_path = "/root/autodl-tmp/data"  # 替换为你要查找的文件夹路径
json_file_path = "/root/autodl-tmp/qwen_finetune/lmms-finetune-main/data_trian.json"  # 替换为你的 JSON 文件路径
count=0
# 查找文件
results = find_videos_from_json(folder_path, json_file_path)

# 输出结果
for filename, path in results.items():
    container = av.open(path)
    if path:
        #print(count)
        count+=1
    else:
        print(f"未找到文件 '{filename}'")