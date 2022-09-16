import torch
from TrainRobustNN.etc.datasets import load_dataset, abstract_data
from d2l import torch as d2l


# load the best model, get the gradient for input and decide whether to generalization or refinement
def refinement(model, model_save_path, batch_size, dataset, interval_num, fnn=False, device=d2l.try_gpu()):
    model.load_state_dict(torch.load(model_save_path+'.pt')['model_state_dict'])
    train_iter, _ = load_dataset(batch_size, dataset)
    model = model.to(device)
    wll, wlu, wul, wuu = 0, 0, 0, 0
    for i, data in enumerate(train_iter):
        if dataset == 'imagenet':
            x = data[0]["data"]
            y = data[0]["label"].squeeze(-1).long().to(x.device)
        else:
            x, y = data
            x, y = x.to(device), y.to(device)

        if fnn:
            if dataset == 'mnist':
                x = x.view(-1, 784)
            elif dataset == 'cifar':
                x = x.view(-1, 3*32*32)

        x = abstract_data(x, interval_num)  # batch_size, channels*2, height, width
        x.requires_grad = True
        loss, logits = model(x, y)
        grad = torch.autograd.grad(loss, x)[0]
        if fnn:
            grad = grad.permute(1,0)
        else:
            grad = grad.permute(1,0,2,3)
        for i in range(grad.shape[0]//2):
            wll += torch.sum(grad[i].clamp(min=0))
            wlu -= torch.sum(grad[i].clamp(max=0))
            wul += torch.sum(grad[i*2].clamp(min=0))
            wuu -= torch.sum(grad[i*2].clamp(max=0))

    # 最简单的方法，一泛化就终止
    if wll > wlu and wul < wuu:
        return False
    return True