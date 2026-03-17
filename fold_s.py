import os

def find_files_by_extension(root_dir, ext):
    """
    遍历 root_dir 目录及其子目录，查找所有扩展名为 ext 的文件。

    参数:
        root_dir (str): 要搜索的根目录路径
        ext (str): 文件扩展名，如 '.yaml'

    返回:
        list: 所有匹配文件的完整路径列表
    """
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename=="coco128.yaml":
                full_path = os.path.join(dirpath, filename)
                file_paths.append(full_path)
    return file_paths

# 使用示例
if __name__ == "__main__":
    folder_path = "/root/yolov10"  # 替换为你想搜索的文件夹路径
    extension = ".yaml"
    
    yaml_files = find_files_by_extension(folder_path, extension)
    
    print(f"找到 {len(yaml_files)} 个 {extension} 文件：")
    for path in yaml_files:
        print(path)