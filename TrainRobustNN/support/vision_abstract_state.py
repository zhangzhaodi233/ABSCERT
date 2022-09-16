from d2l import torch as d2l
import numpy as np
import itertools
from TrainRobustNN.etc import datasets
from TrainRobustNN.etc import conv_models_define
import torch
import os
import matplotlib


def draw_boxplot(x, name):
    x_view = x.reshape(2, -1).cpu().detach().numpy()
    new_x_view = [[], []]
    for t in range(x_view.shape[1]):
        if x_view[1, t] != -1:
            new_x_view[0].append(x_view[0, t])
            new_x_view[1].append(x_view[1, t])
    x_view = np.array(new_x_view)
    
    d2l.set_figsize((30, 5))
    d2l.plt.boxplot(x_view)
    d2l.plt.savefig("output/vision_abstract_state/boxplot/"+name)
    d2l.plt.clf()


def generate_boxplot(test_iter, d, flist):
    for f in flist:
        id = int(f.split('.')[0].split('_')[-1])
        x, y = next(itertools.islice(test_iter, id, None))
        x = datasets.abstract_data(x, 2.0/d)
        draw_boxplot(x, f'{id}.png')


# 绘制修改前后抽象状态的曲线图、散点图
def draw_find_tuning_graph(x1, x2, name):
    x_view1 = x1.reshape(2, -1).cpu().detach().numpy()
    new_x_view1 = []
    x_view2 = x2.reshape(2, -1).cpu().detach().numpy()
    new_x_view2 = []
    for t in range(x_view1.shape[1]):
        if x_view1[1, t] != -1:
            new_x_view1.append((x_view1[0, t] + x_view1[1, t])/2.0)
            new_x_view2.append((x_view2[0, t] + x_view2[1, t])/2.0)
    x_view1 = np.array(new_x_view1)
    x_view2 = np.array(new_x_view2)
    
    d2l.plot(np.arange(len(x_view1)), [x_view1, x_view2], legend=['origin', 'new'], figsize=(30, 5))
    d2l.plt.savefig("output/vision_abstract_state/fine_tuning_graph/曲线图"+name)
    d2l.plt.clf()
    d2l.set_figsize((30, 5))
    d2l.plt.scatter(np.arange(len(x_view1)), x_view1, c='b')
    d2l.plt.scatter(np.arange(len(x_view1)), x_view2, c='r')
    d2l.plt.savefig("output/vision_abstract_state/fine_tuning_graph/散点图"+name)
    d2l.plt.clf()

# 尝试改变抽象状态，看能不能分类成功
def generate_fine_tuning_graph(test_iter, model, d, id):
    x, y = next(itertools.islice(test_iter, id, None))
    x = datasets.abstract_data(x, 2.0/d)
    x, y = x.to('cuda:0'), y.to('cuda:0')
    y_hat = model(x)
    y_true = y.cpu().detach().numpy()
    print(f"y_true: {y_true}")
    print(f"origin_output: {y_hat}")
    x_origin = x.clone()

    y_hat_max = y_hat[0,y_true]
    for h in range(x.shape[2]):
        for w in range(x.shape[3]):
            # 不考虑背景
            if x[0][1][h][w] == -1.0:
                continue

            if x[0][0][h][w] == 1.0:
                continue
            x[0][0][h][w] += d
            x[0][1][h][w] += d
            y_hat = model(x)
            add_ok = False
            if y_hat[0,y_true] > y_hat_max:
                y_hat_max = y_hat[0, y_true]
                add_ok = True
            else:
                x[0][0][h][w] -= d
                x[0][1][h][w] -= d

            if add_ok:
                if x[0][1][h][w] == -1.0 + d or x[0][1][h][w] == -1.0:
                    continue
                x[0][0][h][w] -= 2 * d
                x[0][1][h][w] -= 2 * d
            else:
                if x[0][1][h][w] == -1.0:
                    continue
                x[0][0][h][w] -= d
                x[0][1][h][w] -= d
            y_hat = model(x)
            if y_hat[0, y_true] > y_hat_max:
                y_hat_max = y_hat[0, y_true]
            else:
                if add_ok:
                    x[0][0][h][w] += 2 * d
                    x[0][1][h][w] += 2 * d
                else:
                    x[0][0][h][w] += d
                    x[0][1][h][w] += d
    new_output = model(x)
    print(f"new_output: {new_output}")
    if new_output.argmax(1) == y:
        name = f"{id}_success.png"
    else:
        name = f"{id}_fail.png"
    draw_find_tuning_graph(x_origin, x, name)


def generate_3d_graph(test_iter, d, flist):
    for f in flist:
        id = int(f.split('.')[0].split('_')[-1])
        x, y = next(itertools.islice(test_iter, id, None))
        x = datasets.abstract_data(x, 2.0/d)  # (1, 2, 28, 28)
        rx = np.arange(0, 28, 1)
        ry = np.arange(0, 28, 1)
        rx, ry = np.meshgrid(rx, ry)  # 网格化
        z0, z1 = x[0,0,:,:].detach().numpy(), x[0,1,:,:].detach().numpy()
        z = (z0 + z1) / 2.0
        fig = d2l.plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111,projection='3d')

        min_v, max_v = np.min(z), np.max(z)
        d2l.plt.set_cmap(d2l.plt.get_cmap("seismic", 100))
        color = [d2l.plt.get_cmap("seismic", 100)(int(float(i-min_v)/(max_v-min_v)*100)) for i in z.reshape(-1)]
        im = ax.scatter(ry, rx, z, s=100, c=color, marker='.')
        fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x,pos:round(x*(max_v-min_v)+min_v, 2)))

        d2l.plt.xlabel('X', fontsize=15)
        d2l.plt.ylabel('Y', fontsize=15)
        ax.set_zlabel('Z', fontsize=15)
        d2l.plt.savefig(f'output/vision_abstract_state/3d_graph/{id}.png')
        d2l.plt.clf()
        

def generate_compose_graph(test_iter, d, flist, name):
    print(name)
    d2l.set_figsize((50, 5))
    matrix = np.zeros((20, 784))
    xp = np.arange(0, 20, 1)
    yp = np.arange(0, 784, 1)
    xp, yp = np.meshgrid(xp, yp)
    for f in flist:
        id = int(f.split('.')[0].split('_')[-1])
        x, y = next(itertools.islice(test_iter, id, None))
        x = datasets.abstract_data(x, 2.0/d)  # (1, 2, 28, 28)
        x_view = x.reshape(2, -1).cpu().detach().numpy()  # (2, 784)
        for i in range(x_view.shape[1]):
            matrix[int((x_view[1,i] + 1) * 10), i] += 1

    min_v, max_v = np.min(matrix), np.max(matrix)
    color = [d2l.plt.get_cmap("Blues", 100)(int(float(i-min_v)/(max_v-min_v)*100)) for i in matrix.T.reshape(-1)] 
    d2l.plt.set_cmap(d2l.plt.get_cmap("Blues", 100))
    sc=d2l.plt.scatter(yp, xp, c=color, s=10, marker='.',cmap='Blues')
    d2l.plt.colorbar(sc, format=matplotlib.ticker.FuncFormatter(lambda x,pos:round(x*(max_v-min_v)+min_v, 2)))

    d2l.plt.savefig(f"output/vision_abstract_state/compose_graph/{name}.png")
    d2l.plt.clf()

    m = np.max(matrix, axis=0)
    p = None
    for i, mi in enumerate(m):
        pi = np.where(matrix[:, i] == mi)
        if p is None:
            p = pi[0][-1]
        else:
            p = np.hstack((p, pi[0][-1]))
    p = p.reshape(28,28)
    d2l.plt.imshow(p)
    d2l.plt.savefig(f"vision_abstract_state/compose_graph/{name}_2d.png")
    d2l.plt.clf()



if __name__ == "__main__":
    os.makedirs("output/vision_abstract_state/boxplot", exist_ok=True)
    os.makedirs("output/vision_abstract_state/fine_tuning_graph", exist_ok=True)
    os.makedirs("output/vision_abstract_state/3d_graph", exist_ok=True)
    os.makedirs("output/vision_abstract_state/compose_graph", exist_ok=True)

    dataset = 'mnist'
    model = 'DM_Small'
    in_ch, in_dim = 1, 28
    pretrained_model = "exp_results/mnist_dm_small_0.1.pt"

    _, test_iter = datasets.load_dataset(1, dataset)
    if model == "DM_Small":
        model = conv_models_define.DM_Small(in_ch*2, in_dim)
    elif model == "DM_Medium":
        model = conv_models_define.DM_Medium(in_ch*2, in_dim)
    elif model == "DM_Large":
        model = conv_models_define.DM_Large(in_ch*2, in_dim)
    
    model.load_state_dict(torch.load(pretrained_model)['model_state_dict'])
    model = model.to('cuda:0')
    d = float(pretrained_model.split('_')[-1][:-3])

    generate_fine_tuning_graph(test_iter, model, d, id=6555)

    filelist = os.listdir('counter_examples')
    filelist = sorted(filelist)
    flist = []
    for f in filelist:
        if 'yhat_9_ytrue_8' in f:
            flist.append(f)
    generate_boxplot(test_iter, d, flist)

    generate_3d_graph(test_iter, d, flist)

    flists = []
    names = []
    last = None
    flist = []
    for f in filelist:
        fl = f.split('_')
        if last is None:
            last = fl[0] + '_' + fl[1] + '_' + fl[2] + '_' + fl[3]
            flist.append(f)
            continue
        now = fl[0] + '_' + fl[1] + '_' + fl[2] + '_' + fl[3]
        if now == last:
            flist.append(f)
        else:
            flists.append(flist)
            names.append(last)
            flist = [f]
            last = now
    flists.append(flist)
    names.append(last)

    for i, flist in enumerate(flists):
        generate_compose_graph(test_iter, d, flist, names[i])

    
    flist = []
    filelist = os.listdir('test')
    filelist = sorted(filelist)
    for f in filelist:
        if 'yhat_9_ytrue_9' in f:
            flist.append(f)
    generate_compose_graph(test_iter, d, flist, 'yhat_9_ytrue_9')

