import torch
import torch.nn as nn

from functools import reduce
from operator import mul


def huber_loss(pred, target, delta=1.0):
    error = torch.abs(pred - target)
    quadratic = torch.clamp(error, 0.0, delta)
    linear = error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    # 计算每个样本内部元素的损失和，不再取平均
    return loss  # 假设数据的形状是 [N, C, D, H, W]


class MultiLossFactory(nn.Module):
    def __init__(self):
        super().__init__()

        self.heatmap_loss = WeightedMSELoss(weight_nonzero=1.5)
        self.heatmap_loss_factor = 1

        self.offset_loss = OffsetLoss()
        self.offset_loss_factor = 1

    def forward(self, output, target):
        if self.heatmap_loss:
            heatmap_loss = self.heatmap_loss(output, target)
            heatmap_loss = heatmap_loss * self.heatmap_loss_factor
        else:
            heatmap_loss = None

        if self.offset_loss:
            offset_loss = self.offset_loss(output, target)
            offset_loss = offset_loss * self.offset_loss_factor
        else:
            offset_loss = None

        if self.heatmap_loss:
            heatmap_loss = self.heatmap_loss(output, target)
            heatmap_loss = heatmap_loss * self.heatmap_loss_factor
        else:
            heatmap_loss = None

        # print('heatmap_loss', heatmap_loss)
        # print('offset_loss', offset_loss)
        # print('distribution_loss', distribution_loss)

        loss = heatmap_loss + offset_loss / ((offset_loss / heatmap_loss).detach())

        return loss


class WeightedMSELoss(nn.Module):
    def __init__(self, weight_nonzero, threshold=0.01, weight_zero=0.6):
        """
        初始化加权MSE损失函数
        :param weight_nonzero: 非零值的权重
        :param weight_zero: 零值的权重
        """
        super(WeightedMSELoss, self).__init__()
        self.weight_nonzero = weight_nonzero
        self.weight_zero = weight_zero
        self.threshold = threshold

    def forward(self, input, target):
        """
        计算加权MSE损失
        :param input: 预测值
        :param target: 真实值
        :return: 加权MSE损失
        """
        # 计算基本的MSE损失
        # 应用阈值，修正接近零的预测值
        corrected_predictions = torch.where(input < self.threshold, torch.zeros_like(input), input)

        basic_loss = huber_loss(input, target)
        # print('basic_loss', basic_loss.shape)
        # basic_loss torch.Size([4, 5, 14, 20, 16])
        # weighted_loss torch.Size([4, 5, 14, 20, 16])

        # 根据target是否为零来设置权重
        weights = torch.where(target != 0, self.weight_nonzero, self.weight_zero)

        # 应用权重并计算最终损失
        weighted_loss = weights * basic_loss

        # 对每个样本内部的加权损失求和
        loss_per_sample = torch.sum(weighted_loss, dim=[2, 3, 4])  # 假设数据的形状是 [N, C, D, H, W]
        # print('loss_per_sample', loss_per_sample.shape)

        # 返回批量中所有样本的平均损失
        return torch.mean(loss_per_sample)


class FocalLossForRegression(nn.Module):
    def __init__(self, gamma=2.0, beta=1.0, reduction='mean'):
        """
        初始化 Focal Loss。
        参数:
        - gamma: 调节易分类样本的影响。
        - beta: 平滑 L1 损失的阈值。
        - reduction: 指定损失的缩减方式，可选 'none', 'mean', 'sum'。
        """
        super(FocalLossForRegression, self).__init__()
        self.gamma = gamma

    # def forward(self, inputs, targets):
    #     """
    #     前向传播计算损失。
    #     参数:
    #     - inputs: 模型预测值。
    #     - targets: 真实值。
    #     """
    #     # 计算绝对误差
    #     l1_loss = torch.abs(inputs - targets)

    #     # 计算平滑 L1 损失
    #     condition = l1_loss < self.beta
    #     loss = torch.where(condition, 0.5 * l1_loss ** 2 / self.beta, l1_loss - 0.5 * self.beta)

    #     # 应用 Focal Loss 调节因子
    #     focal_loss = torch.pow(l1_loss, self.gamma) * loss

    #     # 根据 reduction 参数缩减损失
    #     loss_per_sample = torch.sum(focal_loss, dim=[2, 3, 4])  # 假设数据的形状是 [N, C, D, H, W]
    #     # print('loss_per_sample', loss_per_sample.shape)

    #     # 返回批量中所有样本的平均损失
    #     return torch.mean(loss_per_sample)

    def forward(self, inputs, targets):
        """
        前向传播计算损失。
        参数:
        - inputs: 模型预测值。
        - targets: 真实值。
        """
        # gamma=1.0 seems more reliable, but gamma=2 brings about more AP increase
        st = torch.where(torch.ge(targets, 0.01), inputs, 1 - inputs)
        factor = torch.pow(1. - st, self.gamma)
        # print('the factor is \n', factor)
        out = torch.mul((inputs - targets), (inputs - targets)) * factor
        loss_per_sample = torch.sum(out, dim=[2, 3, 4])  # 假设数据的形状是 [N, C, D, H, W]
        # print('loss_per_sample', loss_per_sample.shape)

        # 返回批量中所有样本的平均损失
        return torch.mean(loss_per_sample)


class OffsetLoss(nn.Module):
    def __init__(self):
        super(OffsetLoss, self).__init__()

    def smooth_l1_loss(self, pred, gt, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
        return loss

    def forward(self, pred, gt):
        loss = 0
        batch, channel, _, _, _ = pred.shape
        assert pred.size() == gt.size()
        for b in range(batch):
            for c in range(channel):
                pred_point = self.softargmax(pred[b][c])
                gt_point = self.softargmax(gt[b][c])
                # Sum the loss over all dimensions to get a scalar
                loss += torch.sum(self.smooth_l1_loss(pred_point, gt_point))
        # Average the loss over the batch and channel dimensions
        loss = loss / (batch * channel)
        return loss

    def softargmax(self, tensor_3d):
        depth, height, width = tensor_3d.shape
        max_val = tensor_3d.max()
        max_point = torch.zeros(3)
        soft_argmax_x = torch.zeros((depth, height, width), device=tensor_3d.device)
        soft_argmax_y = torch.zeros((depth, height, width), device=tensor_3d.device)
        soft_argmax_z = torch.zeros((depth, height, width), device=tensor_3d.device)

        for i in range(depth):
            soft_argmax_x[i, :, :] = (i + 1) / depth
        for i in range(height):
            soft_argmax_y[:, i, :] = (i + 1) / height
        for i in range(width):
            soft_argmax_z[:, :, i] = (i + 1) / width

        # Compute softmax
        # tensor_3d_softmax = flat_softmax(tensor_3d)
        tensor_3d_softmax = self.softmax(tensor_3d, max_val)
        # Compute weighted coordinates
        x_coord = torch.sum(tensor_3d_softmax * soft_argmax_x)
        y_coord = torch.sum(tensor_3d_softmax * soft_argmax_y)
        z_coord = torch.sum(tensor_3d_softmax * soft_argmax_z)

        max_point[0] = x_coord.item() * depth - 1
        max_point[1] = y_coord.item() * height - 1
        max_point[2] = z_coord.item() * width - 1

        return max_point

    def softmax(self, tensor, max_val):
        return torch.exp(tensor - max_val) / torch.sum(torch.exp(tensor - max_val))


import torch
import torch.nn as nn
from functools import reduce


class DistributionLoss(nn.Module):
    def __init__(self):
        super(DistributionLoss, self).__init__()

    def forward(self, heatmaps, gt):
        loss = self.js_reg_losses(heatmaps, gt)
        return loss

    def js_reg_losses(self, heatmaps, gt):
        # self.check_tensor(heatmaps, "Heatmaps")
        # self.check_tensor(gt, "Ground Truth")
        loss = 0
        batch, channel, _, _, _ = heatmaps.shape
        assert heatmaps.size() == gt.size()
        for b in range(batch):
            for c in range(channel):
                output = self.normalize_distribution(heatmaps[b][c])
                target = self.normalize_distribution(gt[b][c])
                # print("Heatmaps max:", output.max().item(), "min:", output.min().item())
                # print("GT max:", target.max().item(), "min:", target.min().item())
                assert (output >= 0).all(), "Negative values found in normalized heatmaps"
                assert (target >= 0).all(), "Negative values found in normalized ground truth"

                # Check that sums are close to 1
                assert torch.allclose(output.sum(dim=[-3, -2, -1]), torch.tensor(1.0, dtype=output.dtype),
                                      atol=1e-4), "Sum of normalized heatmaps not close to 1"
                assert torch.allclose(target.sum(dim=[-3, -2, -1]), torch.tensor(1.0, dtype=output.dtype),
                                      atol=1e-4), "Sum of normalized heatmaps not close to 1"

                divergence_loss = self._divergence_reg_losses(output, target, self._js)
                if torch.isnan(divergence_loss).any():
                    print(f"NaN detected in divergence loss at batch {b}, channel {c}")
                loss += divergence_loss
        loss = loss / (batch * channel)
        return loss

    def _js(self, p, q, ndims):
        m = 0.5 * (p + q)
        return 0.5 * (self._kl(p, m, ndims) + self._kl(q, m, ndims))

    # def _kl(self, p, q, ndims):
    #     eps = 1e-24
    #     p = torch.clamp(p, min=eps, max=1-eps)
    #     q = torch.clamp(q, min=eps, max=1-eps)
    #     unsummed_kl = p * ((p / q).log())
    #     kl_values = reduce(lambda t, _: t.sum(-1, keepdim=True), range(ndims), unsummed_kl)
    #     return kl_values

    def _kl(self, p, q, ndims):
        eps = 1e-24
        p = torch.clamp(p, min=eps, max=1 - eps)
        q = torch.clamp(q, min=eps, max=1 - eps)
        log_p = p.log()

        # Apply KL divergence
        kl_div = p * (log_p - q.log())

        # Sum over the spatial dimensions and clamp if necessary
        kl_div = kl_div.sum(dim=list(range(-ndims, 0)), keepdim=True)
        kl_div = torch.clamp(kl_div, min=0, max=1e3)  # Adjust max as needed

        return kl_div.mean()  # Take the mean if needed

    def _divergence_reg_losses(self, heatmaps, gt, divergence):
        ndims = 3  # Assuming 3D spatial dimensions
        divergences = divergence(heatmaps, gt, ndims)
        return divergences.mean()  # Mean over all elements

    def normalize_distribution(self, distribution):
        distribution = torch.clamp(distribution, min=1e-6, max=1e6).to(torch.float64)

        distribution = torch.relu(distribution)  # Redundant but safe
        # distribution = torch.sigmoid(distribution)  # Redundant but safe
        if distribution.max() < 1e-12:
            distribution = distribution * 1e6
            assert (distribution >= 0).all(), "Negative values found after scaling"
        val_sum = distribution.sum(dim=[-3, -2, -1], keepdim=True).clamp(min=1e-24)
        assert val_sum.gt(0).all(), "Non-positive sum found before division"
        normalized_distribution = distribution / val_sum
        assert (normalized_distribution >= 0).all(), "Negative values found after normalization"
        return normalized_distribution

    def check_tensor(self, tensor, name="Tensor"):
        print(f"Checking {name} for NaNs or Infs")
        print("Has NaN:", torch.isnan(tensor).any().item())
        print("Has Inf:", torch.isinf(tensor).any().item())

# def compute_original_dist(self, target, output):
#         b, c, _, h, w = target.shape
#         center = torch.zeros((b, 4, 3), dtype=torch.float32,device='cuda') #Hard-code here
#         center_predict = torch.zeros((b, 4, 3), dtype=torch.float32,device='cuda')

#         for batch_idx in range(b):
#             kernel = target[batch_idx][0]
#             kernel_out = output[batch_idx][0]
#             # print('3',kernel.shape)
#             # print('4',kernel_out.shape)

#             # 使用 torch.where 替换 np.where，条件判断需要调整为 PyTorch 的方式
#             non_zero_indices = torch.where(kernel > 0)

#             if non_zero_indices[0].numel() == 0:  # 使用 numel() 方法检查非零元素数量
#                 continue

#             for i, bias in enumerate([1,2,3,4]): #hard-code here
#                 adjusted_kernel = kernel - bias - 0.7
#                 adjusted_output = kernel_out - bias

#                 mask = (adjusted_kernel > 0) & (adjusted_kernel < 1)
#                 if not torch.any(mask):
#                     print('label has non-zero!!!', bias)
#                     continue
#                 # 使用 torch.nonzero 替代 np.nonzero，且不需要将结果转换为数组
#                 gt_indices = torch.nonzero(mask, as_tuple=False)

#                 max_index = torch.argmax(adjusted_kernel[mask])
#                 peak_coords_abs = gt_indices[max_index]
#                 # print('peak_coords_abs!!!', peak_coords_abs)
#                 # print('peak', peak_coords_abs.shape)
#                 center[batch_idx, i, :] = peak_coords_abs

#                 mask_output = (adjusted_output > 0) & (adjusted_output < 1)
#                 if not torch.any(mask_output):
#                     print('output has non-zero!!!', bias)
#                     continue
#                 gt_indices_output = torch.nonzero(mask_output, as_tuple=False)

#                 max_index_predict = torch.argmax(adjusted_output[mask_output])
#                 peak_coords_abs_predict = gt_indices_output[max_index_predict]
#                 # print(peak_coords_abs_predict.shape)
#                 center_predict[batch_idx, i, :] = peak_coords_abs_predict

#             # 使用 torch.linalg.norm 替代 np.linalg.norm，并在最后计算平均值
#             non_zero_mask = torch.logical_and(center != 0, center_predict != 0)
#             # print('non_zero_mask', non_zero_mask.shape)

#             # 使用掩码过滤出非零元素
#             non_zero_center = center[non_zero_mask]
#             non_zero_center_predict = center_predict[non_zero_mask]

#         # 如果有非零元素，计算它们的范数，否则设定一个默认值（例如0或其他适当的值）
#         if non_zero_center.nelement() > 0:
#             total_distance = torch.linalg.norm(non_zero_center - non_zero_center_predict, dim=-1)
#             # print(total_distance.shape)
#             total_distance = torch.sum(total_distance) / b
#         else:
#             total_distance = torch.tensor(1000.0)  # 或者任何适当的默认值
#         return total_distance
