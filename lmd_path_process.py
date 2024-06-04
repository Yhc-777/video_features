import os

# 指定视频文件夹的路径
video_folder_path = '../MSVD_VTT/MSVD/MSVD_video'
# 指定要保存路径的文本文件的路径
output_file_path = '../MSVD_VTT/MSVD/MSVD_video_paths_list.txt'

# 支持的视频文件扩展名列表
video_extensions = ['.mp4', '.avi', '.mkv']

# 遍历文件夹中的所有文件和子文件夹
for root, dirs, files in os.walk(video_folder_path):
    for file in files:
        # 检查文件扩展名是否在支持的列表中
        if any(file.lower().endswith(ext) for ext in video_extensions):
            # 构造完整的文件路径
            full_path = os.path.join(root, file)
            # 将文件路径写入到文本文件中
            with open(output_file_path, 'a') as f:  # 'a' 表示追加模式
                f.write(full_path + '\n')

print(f"所有视频文件的路径已保存到 {output_file_path}")