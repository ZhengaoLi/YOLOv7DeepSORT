from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array
from moviepy.video.fx.all import speedx  # 确保正确导入 speedx 函数


# 加载两个视频
videoA = VideoFileClip("Result_original.avi")
videoB = VideoFileClip("Result_new.avi")

# 调整视频的高度一致
height = min(videoA.h, videoB.h)  # 选择较小的高度
videoA_resized = videoA.resize(height=height)
videoB_resized = videoB.resize(height=height)

# 横向拼接
combined_video = clips_array([[videoA_resized, videoB_resized]])

# 调整速度 (放慢 1/4 倍)
slowed_video = combined_video.fx(speedx, 0.25)

# 保存最终视频
slowed_video.write_videofile("output_slowed.avi", codec="libx264", audio_codec="aac")
