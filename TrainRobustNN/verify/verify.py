import torch
from TrainRobustNN.utils.mapping_func import abstract_data
from d2l import torch as d2l


def verify(model, dataset, interval_num, data_iter, fnn=False, device=d2l.try_gpu()):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for data in data_iter:
            if dataset == 'imagenet':
                x = data[0]["data"]
                y = data[0]["label"].squeeze(-1).long().to(x.device)
            else:
                x, y = data
                x, y = x.to(device), y.to(device)

            if fnn:
                if dataset == 'mnist':
                    x_fnn = x.view(-1, 784)
                elif dataset == 'cifar':
                    x_fnn = x.view(-1, 3*32*32)
                x_abstract = abstract_data(x_fnn, interval_num)
            else:
                x_abstract = abstract_data(x, interval_num)
                
            logits = model(x_abstract)
        
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)

        model.train()
        return acc_sum / n