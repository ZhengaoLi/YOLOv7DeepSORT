import os

def prepend_folder_name_to_files(current_folder):
    # 获取当前文件夹路径
    # current_folder = os.getcwd()
    folder_name = os.path.basename(current_folder)  # 获取文件夹名称

    # 遍历当前文件夹中的所有文件
    for filename in os.listdir(current_folder):
        # 检查是否为 .jpg 文件
        if filename.endswith(".jpg") or filename.endswith(".txt"):
            # 构造新的文件名
            new_filename = f"{folder_name}_{filename}"
            
            # 重命名文件
            os.rename(
                os.path.join(current_folder, filename),
                os.path.join(current_folder, new_filename)
            )
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    prepend_folder_name_to_files('Dataset1.0/C004')
