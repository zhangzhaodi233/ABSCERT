# 依据log计算所有η模型的每个epoch的平均训练时间

def catpe(file_name):
    with open('result/'+file_name, 'r') as f:
        time_sum = 0
        time_count = 0
        for line in f.readlines():
            if 'time' in line:
                tokens = line.split(' ')
                time_sum += float(tokens[-1])
                time_count += 1
        print(time_sum / time_count)

if __name__ == "__main__":
    catpe('mnist_dm_small.log')
    catpe('mnist_dm_medium.log')
    catpe('mnist_dm_large.log')
    catpe('cifar_dm_small.log')
    catpe('cifar_dm_medium.log')
    catpe('cifar_dm_large.log')
