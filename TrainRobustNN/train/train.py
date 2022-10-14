import torch
from torch import nn
from TrainRobustNN.utils.datasets import load_dataset
from TrainRobustNN.utils.mapping_func import abstract_data
import math
import time
from transformers import get_cosine_schedule_with_warmup
from d2l import torch as d2l
from TrainRobustNN.verify.verify import verify


def printlog(s, log_path):
    print(s, file=open(log_path+".log", "a"))

def print_concise_log(s, log_path):
    print(s, file=open(log_path+"_concise.log", "a"))

def train(model, dataset, model_save_path, log_path, fnn=False, interval_num=1, batch_size=256, epochs=10, learning_rate=0.01, optimizer="Adam", weight_decay=0.0001, momentum=0.9, init="xavier", lr_scheduler='cosine', device=d2l.try_gpu()):
    
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init == "gaussian":  # gaussian init，used in alexnet and vgg
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif init == 'kaiming':  # proposed by Kaiming He(MSRA), used in resnet
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            else:  # default xavier init
                nn.init.xavier_uniform_(m.weight)
    
    if hasattr(model, 'conv'):
        model.conv.apply(init_weights)
    model.fc.apply(init_weights)

    train_iter, test_iter = load_dataset(batch_size, dataset)
    last_epoch = -1

    len_train_iter = len(train_iter)
    num_training_steps = len_train_iter * epochs
    if optimizer == "SGD":
        optimizer = torch.optim.SGD([{
            "params": model.parameters(), "lr": learning_rate, "weight_decay": weight_decay, "momentum": momentum
        }])
    else:
        optimizer = torch.optim.Adam([{
            "params": model.parameters(), "lr": learning_rate, "weight_decay": weight_decay
        }])
    if lr_scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, math.ceil(epochs/5), 0.1, -1)  # 下调4次
    else:  # cosine
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(epochs * len_train_iter / 10),
                                                num_training_steps=num_training_steps,
                                                num_cycles=2, last_epoch=last_epoch)

    
    animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs], legend=['train loss', 'train acc', 'test acc'])
    model = model.to(device)
    max_test_acc = 0
    time_sum = 0
    
    for epoch in range(epochs):
        # print current learning rate
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        start = time.time()
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

            x = abstract_data(x, interval_num)
            
            loss, logits = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler != "steplr":
                scheduler.step()
            if i%50 == 0:  # print log every 50 iter
                acc = (logits.argmax(1) == y).float().mean()
                print("### Epochs [{}/{}] --- batch[{}/{}] --- acc {:.4} --- loss {:.4}".format(epoch+1, epochs, i, len(train_iter), acc, loss.item()))
                printlog("### Epochs [{}/{}] --- batch[{}/{}] --- acc {:.4} --- loss {:.4}".format(epoch+1, epochs, i, len(train_iter), acc, loss.item()), log_path)
                animator.add(epoch + (i + 1) / len_train_iter, (loss.detach().cpu(), acc.cpu(), None))

        time_sum += time.time() - start
        if lr_scheduler == "steplr":
            scheduler.step()

        test_acc = verify(model, dataset, interval_num, test_iter, fnn=fnn, device=device)
        print("### Epochs [{}/{}] -- Acc on test {:.4}".format(epoch + 1, epochs, test_acc))
        printlog("### Epochs [{}/{}] -- Acc on test {:.4}".format(epoch + 1, epochs, test_acc), log_path)
        animator.add(epoch + 1, (None, None, test_acc))


        if test_acc > max_test_acc:
            max_test_acc = test_acc
            state_dict = model.state_dict()
            torch.save({'last_epoch': scheduler.last_epoch,
                    'model_state_dict': state_dict},
                    model_save_path + '.pt')
    print("### max test accuracy: {:.4}".format(max_test_acc))
    print("### error: {:.4}%".format((1 - max_test_acc)*100))
    print("### time of per epoch: {:.4}".format(time_sum / epoch))
    printlog("### max test accuracy: {:.4}".format(max_test_acc), log_path)
    printlog("### error: {:.4}%".format((1 - max_test_acc)*100), log_path)
    printlog("### time of per epoch: {:.4}".format(time_sum / epoch), log_path)
    d2l.plt.savefig(f'{model_save_path}.png')
    return max_test_acc