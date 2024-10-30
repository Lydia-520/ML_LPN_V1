import os
import pandas as pd
import matplotlib.pyplot as plt


# 遍历文件夹寻找 train_log.txt 文件
def find_train_logs(root_dir):
    train_logs = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "train_log.txt" in filenames:
            train_logs.append(os.path.join(dirpath, "train_log.txt"))
    return train_logs


# 读取 train_log.txt 文件最后 20 条数据并计算 Training Loss 和 Validation Loss 的平均值
def get_last_50_avg_losses(log_path):
    # 打开文件并手动处理数据
    data = []
    with open(log_path, 'r') as file:
        for line in file:
            # 假设每行格式类似 "Epoch X, Training Loss: X, Validation Loss: X"
            if "Training Loss" in line and "Validation Loss" in line:
                parts = line.split(',')
                # 提取出 Training Loss 和 Validation Loss 的数值
                try:
                    training_loss = float(parts[1].split(':')[-1].strip())
                    validation_loss = float(parts[2].split(':')[-1].strip())
                    data.append([training_loss, validation_loss])
                except ValueError:
                    continue

    # 将数据转化为 DataFrame
    df = pd.DataFrame(data, columns=["training_loss", "validation_loss"])

    # 如果数据少于 20 条，取全部；否则取最后 20 条
    last_50 = df.tail(50)

    # 计算 Training Loss 和 Validation Loss 的平均值
    training_loss_avg = last_50["training_loss"].mean()
    validation_loss_avg = last_50["validation_loss"].mean()

    return training_loss_avg, validation_loss_avg


# 从文件夹名称中提取最后的数字
def get_folder_number(path):
    folder_name = os.path.basename(os.path.dirname(path))
    # 提取文件夹名称最后的数字
    folder_number = ''.join(filter(str.isdigit, folder_name))
    return int(folder_number)


# 主程序：获取所有文件夹中的平均 loss 值并生成折线图
def main(root_dir):
    train_logs = find_train_logs(root_dir)

    results = []

    for log in train_logs:
        folder_number = get_folder_number(log)
        training_loss_avg, validation_loss_avg = get_last_50_avg_losses(log)
        print(
            f"Folder: {folder_number}, Training Loss Avg: {training_loss_avg}, Validation Loss Avg: {validation_loss_avg}")
        results.append((folder_number, training_loss_avg, validation_loss_avg))

    # 按文件夹编号升序排序
    results.sort(key=lambda x: x[0])

    # 分别提取文件夹编号、训练损失和验证损失
    folder_numbers = [x[0] for x in results]
    training_losses = [x[1] for x in results]
    validation_losses = [x[2] for x in results]

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(folder_numbers, training_losses, marker='o', label='Training Loss', color='blue')
    plt.plot(folder_numbers, validation_losses, marker='o', label='Validation Loss', color='red')

    plt.xlabel('Folder Number')
    plt.ylabel('Loss (Average of Last 50)')
    plt.title('Comparison of Training and Validation Losses (Last 50 epochs)')
    plt.legend()
    plt.grid(True)

    # 保存折线图到上一级目录
    combined_save_path = os.path.join(root_dir, "..", "loss_comparison_plot_50_two.png")
    plt.savefig(combined_save_path)
    plt.show()


# 执行主程序，指定根文件夹
root_directory = "./save_results/2024-09-18 02:11:48"  # 替换为实际的根文件夹路径
main(root_directory)
