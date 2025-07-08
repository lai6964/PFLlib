import matplotlib.pyplot as plt


# datadir = "cnn_cifar100_01"
# lastname = "_cifar100.log"

datadir = "res_tiny_01"
lastname = "_tiny.log"


def read_data(name):
    with open(f'{datadir}/{name}{lastname}', 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        if "Averaged Test Accuracy:" in line:
            num = float(line.split(":")[-1])
            data.append(num)
    return data[:40]


# 假设数据文件位于当前目录下，并且每个文件包含多行数据，每行代表一个训练轮次的准确率
algorithms = ['FedPAC', 'PerAvg', 'FedProto', 'FedALA', 'FedCP', 'FedAS', 'FedDFCC','FedROD']
colors = ['blue', 'yellow', 'purple', 'orange', 'gray', 'lightblue', 'red', 'black']
markers = ['_', '_', '_', '_', '_', '_', '_', '_']

# 读取数据并绘制图形
plt.figure(figsize=(10, 5))
for algo, color, marker in zip(algorithms, colors, markers):
    try:
        # 读取数据
        data = read_data(algo)
        print("{}:{}".format(algo,max(data)))

        # 绘制准确率曲线
        plt.plot(data, label=algo, color=color, marker='*')
    except FileNotFoundError:
        print(f"File for {algo} not found.")

# 添加图例、标题和标签
plt.legend()
# plt.title('Accuracy Curve on Cifar100')
plt.title('Accuracy Curve on TinyImagenet')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# 显示图形
plt.savefig(f"{datadir}/{lastname[:-4]}.png")
plt.show()