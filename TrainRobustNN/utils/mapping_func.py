import torch
import numpy as np

def abstract_data(x, interval_num):

    step = (1-(-1))/interval_num
    k = torch.div((x - (-1)), step, rounding_mode='floor')
    x_lower = -1 + k * step
    x_lower = torch.clamp(x_lower, -1, 1)
    x_upper = x_lower + step
    x_upper = torch.clamp(x_upper, -1, 1)

    # if x=1，it should not be abstracted to [1,1]
    eq = torch.eq(x_lower, x_upper)
    x_lower = x_lower - step * eq.int()
    x_lower = torch.clamp(x_lower, min=-1)

    x_result = torch.cat((x_upper, x_lower), dim=1)
    return x_result

# def abstract_data(x, interval_num):
#     x_cp = x.clone()
#     x_result = torch.cat((x, x_cp), dim=1)
#     return x_result