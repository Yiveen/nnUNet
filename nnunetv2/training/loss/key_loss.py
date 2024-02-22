import torch
import torch.nn as nn

def huber_loss(pred, target, delta=1.0):
    error = torch.abs(pred - target)
    quadratic = torch.clamp(error, 0.0, delta)
    linear = error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(loss)

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_nonzero, weight_zero=1.0):
        """
        初始化加权MSE损失函数
        :param weight_nonzero: 非零值的权重
        :param weight_zero: 零值的权重
        """
        super(WeightedMSELoss, self).__init__()
        self.weight_nonzero = weight_nonzero
        self.weight_zero = weight_zero

    def forward(self, input, target):
        """
        计算加权MSE损失
        :param input: 预测值
        :param target: 真实值
        :return: 加权MSE损失
        """
        # 计算基本的MSE损失
        basic_loss = huber_loss(input,target)

        # 根据target是否为零来设置权重
        weights = torch.where(target != 0, self.weight_nonzero, self.weight_zero)

        # 应用权重并计算最终损失
        weighted_loss = weights * basic_loss

        # 返回加权损失的均值
        return weighted_loss.mean()