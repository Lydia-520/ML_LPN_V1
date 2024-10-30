import os
import matplotlib.pyplot as plt


def plot_and_save_log(log_file, save_dir, index):
    # 读取train_log.txt的内容
    epochs, losses = [], []
    with open(log_file, 'r') as f:
        for line in f:
            # 假设train_log.txt的格式为: "epoch: x, loss: y"
            if "epoch" in line and "loss" in line:
                parts = line.split(',')
                epoch = int(parts[0].split(':')[-1].strip())
                loss = float(parts[1].split(':')[-1].strip())
                epochs.append(epoch)
                losses.append(loss)

    # 生成并保存变化图
    plt.figure()
    plt.plot(epochs, losses, label=f'Training Loss {index}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Train Log - {index}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'train_log_plot_{index}.png'))
    plt.close()


def create_summary_plot(all_plots, summary_dir):
    # 汇总所有变化图到一张图
    total_plots = len(all_plots)
    rows = int(total_plots ** 0.5) + 1
    cols = (total_plots // rows) + (1 if total_plots % rows > 0 else 0)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axs = axs.flatten()  # 扁平化为一维数组，方便逐一添加图像

    for i, plot in enumerate(all_plots):
        img = plt.imread(plot)
        axs[i].imshow(img)
        axs[i].set_title(f'Plot {i + 1}')
        axs[i].axis('off')  # 隐藏坐标轴

    # 删除多余的子图
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # 保存汇总图
    summary_path = os.path.join(summary_dir, 'summary_plot.png')
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()


def process_logs_and_generate_plots(root_dir):
    all_plots = []

    # 遍历所有文件夹
    for root, dirs, files in os.walk(root_dir):
        if 'train_log.txt' in files:
            log_file = os.path.join(root, 'train_log.txt')
            index = os.path.basename(root)  # 使用文件夹名作为索引
            plot_and_save_log(log_file, root, index)
            all_plots.append(os.path.join(root, f'train_log_plot_{index}.png'))

    # 生成汇总图，保存在上一级目录
    if all_plots:
        create_summary_plot(all_plots, root_dir)


# 调用函数，传入根文件夹路径
process_logs_and_generate_plots('./save_results')
