import base64
from zhipuai import ZhipuAI

video_path = "/root/autodl-tmp/data/deepmind/video_105.mp4"
with open(video_path, 'rb') as video_file:
    video_base = base64.b64encode(video_file.read()).decode('utf-8')

client = ZhipuAI(api_key="ae178e722a2c71f5c2f17bcf67f16312.hUUtHagstwLo8WCZ") # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4v-plus",  # 填写需要调用的模型名称
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "video_url",
            "video_url": {
                "url" : video_base
            }
          },
          {
            "type": "text",
            "text": "Generate a comprehensive and concise description of the entire video, summarizing its key activities and events without focusing on individual frames."
          }
        ]
      }
    ]
)
print(response.choices[0].message)