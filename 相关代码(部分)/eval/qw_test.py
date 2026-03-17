import cv2

def check_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    print(f"Frame count: {frame_count}, FPS: {fps}, Duration: {duration}")
    cap.release()
    return True

# 测试视频文件
video_path = "/root/autodl-tmp/data/deepmind/video_105.mp4"
check_video(video_path)