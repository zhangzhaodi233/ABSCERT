from thop import profile
import my_models_define
import torch

# 使用thop计算每个模型的参数量(Params)和计算复杂度(MACs)


if __name__ == "__main__":
    models = [my_models_define.DM_Small(2, 28),
        my_models_define.DM_Medium(2, 28),
        my_models_define.DM_Large(2, 28),
        my_models_define.DM_Small(6, 32),
        my_models_define.DM_Medium(6, 32),
        my_models_define.DM_Large(6, 32),
        my_models_define.AlexNet(6),
        my_models_define.VGG11(6),
        my_models_define.ResNet18(6),
        my_models_define.ResNet34(6),
        my_models_define.ResNet50(6)]

    xs = [torch.randn((1, 2, 28, 28)),
        torch.randn((1, 2, 28, 28)),
        torch.randn((1, 2, 28, 28)),
        torch.randn((1, 6, 32, 32)),
        torch.randn((1, 6, 32, 32)),
        torch.randn((1, 6, 32, 32)),
        torch.randn((1, 6, 224, 224)),
        torch.randn((1, 6, 224, 224)),
        torch.randn((1, 6, 224, 224)),
        torch.randn((1, 6, 224, 224)),
        torch.randn((1, 6, 224, 224))]

    for model,x in zip(models, xs):
        macs, params = profile(model, inputs=(x, ))
        print(macs)
        print(params)