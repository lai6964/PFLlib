import matplotlib.pyplot as plt
from PIL import Image


font2 = {'family': 'Times New Roman',
         'size': 17}

font_axis = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 15}

font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 16}

def crop_white_border(img_path, out_path, threshold=255):
    """
    裁掉四周纯白区域并保存。
    threshold: 把 >= 这个灰度值的像素当成“白”
    """
    img = Image.open(img_path).convert("RGBA")          # 保留透明度
    datas = img.getdata()

    # 把纯白像素设为透明，方便用 getbbox
    new_data = []
    for r, g, b, a in datas:
        if r >= threshold and g >= threshold and b >= threshold:
            new_data.append((255, 255, 255, 0))         # 设为完全透明
        else:
            new_data.append((r, g, b, a))
    img.putdata(new_data)

    bbox = img.getbbox()                                # 得到非透明区域
    if bbox:
        cropped = img.crop(bbox).convert("RGB")         # 转回 RGB
        cropped.save(out_path)
        print(f"已裁剪并保存为: {out_path}")
    else:
        print("整张图都是白的，没有可裁剪区域。")



def read_data(name):
    with open(f'{datadir}/{name}{lastname}', 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        if "Averaged Test Accuracy:" in line:
            num = float(line.split(":")[-1])
            data.append(num)
    return data[:40]


if __name__ == '__main__':
    datadir = "cnn_cifar100_01"
    lastname = "_cifar100.log"

    # datadir = "res_tiny_01"
    # lastname = "_tiny.log"


    # 假设数据文件位于当前目录下，并且每个文件包含多行数据，每行代表一个训练轮次的准确率
    algorithms = ['PerAvg', 'FedPAC', 'FedProto', 'FedALA', 'FedCP', 'FedAS','FedROD', 'FedDFPA']
    colors = ['#b39cd0', '#008bc9', '#ff6f91', '#4d8076', '#d5cabd', '#ffc75f', '#926d00', '#c34a36']
    markers = ['_', '_', '_', '_', '_', '_', '_', '_']

    # 读取数据并绘制图形
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.15, top=0.98)  # 调整子图之间的间距
    for algo, color, marker in zip(algorithms, colors, markers):
        try:
            # 读取数据
            data = read_data(algo)
            print("{}:{}".format(algo,max(data)))

            # 绘制准确率曲线
            plt.plot(data, label=algo, color=color, marker='*', markersize=7, linestyle='--', linewidth=2)
        except FileNotFoundError:
            print(f"File for {algo} not found.")

    # 添加图例、标题和标签
    plt.legend(ncol=2, prop=font_legend)
    ax.set_xlabel('Commincation round $\Gamma$', fontdict=font2)
    ax.set_ylabel('Test Accuracy', fontdict=font2)
    plt.yticks(font=font_axis)
    plt.xticks(font=font_axis)
    ax.grid(True, linestyle='-.')

    # 显示图形
    # plt.savefig(f"{datadir}/{lastname[:-4]}.pdf")
    plt.savefig(f"{datadir}/{lastname[:-4]}.pdf",
                bbox_inches='tight',  # 关键：让 savefig 计算紧凑边界
                pad_inches=0)  # 可选：不留额外边距（默认 0.1）
    plt.show()

    # crop_white_border(f"{datadir}/{lastname[:-4]}.pdf", f"{datadir}/{lastname[:-4]}_crop.pdf")