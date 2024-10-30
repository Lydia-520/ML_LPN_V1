import os
import re
import matplotlib.pyplot as plt


# 定义函数：生成图表并保存
def plot_and_save(epochs, training_losses, validation_losses, folder_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss in {folder_path}')
    plt.legend()
    plt.grid(True)

    # 保存图表到对应文件夹
    save_path = os.path.join(folder_path, 'loss_plot.png')
    plt.savefig(save_path)
    plt.close()


# 定义函数：解析 train_log.txt 文件并生成图表
def process_train_log(file_path, folder_path):
    epochs = []
    training_losses = []
    validation_losses = []

    # 读取并解析 train_log.txt
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            epoch = int(re.search(r'Epoch (\d+)', line).group(1))
            training_loss = float(re.search(r'Training Loss: ([\d.]+)', line).group(1))
            validation_loss = float(re.search(r'Validation Loss: ([\d.]+)', line).group(1))
            epochs.append(epoch)
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

    # 生成并保存图表
    plot_and_save(epochs, training_losses, validation_losses, folder_path)


# 遍历文件夹寻找 train_log.txt 文件
def find_and_process_logs(root_directory):
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if 'train_log.txt' in filenames:  # 如果文件夹中有 train_log.txt 文件
            file_path = os.path.join(dirpath, 'train_log.txt')
            process_train_log(file_path, dirpath)
            print(f"Generated plot for {file_path}")


# 设置根目录（需要替换为你的实际路径）
root_directory = './save_results/2024-09-18 01:36:17'

# 执行遍历并生成图表
find_and_process_logs(root_directory)

# import matplotlib.pyplot as plt
# import re
#
# # 初始化列表来保存 Epoch, Training Loss 和 Validation Loss
# epochs = []
# training_losses = []
# validation_losses = []
#
# # 读取 train_log.txt 文件
# with open('train_log.txt', 'r') as file:
#     lines = file.readlines()
#     for line in lines:
#         # 使用正则表达式解析每行的 Epoch, Training Loss 和 Validation Loss
#         epoch = int(re.search(r'Epoch (\d+)', line).group(1))
#         training_loss = float(re.search(r'Training Loss: ([\d.]+)', line).group(1))
#         validation_loss = float(re.search(r'Validation Loss: ([\d.]+)', line).group(1))
#
#         # 将解析到的值保存到列表中
#         epochs.append(epoch)
#         training_losses.append(training_loss)
#         validation_losses.append(validation_loss)
#
# # 生成图表
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, training_losses, label='Training Loss')
# plt.plot(epochs, validation_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()
