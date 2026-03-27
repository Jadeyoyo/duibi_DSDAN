import torch
import torch.nn as nn
import numpy as np


class LMMD_loss(nn.Module):
    def __init__(self, class_num=12, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super().__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        l2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma is not None:
            bandwidth = fix_sigma
        else:
            denom = max(n_samples ** 2 - n_samples, 1)
            bandwidth = torch.sum(l2_distance.detach()) / denom
            bandwidth = torch.clamp(bandwidth, min=1e-6)

        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-l2_distance / torch.clamp(bw, min=1e-6)) for bw in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        device = source.device
        batch_size = source.size(0)
        weight_ss, weight_tt, weight_st = self.cal_weight(s_label, t_label, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).to(device)
        weight_tt = torch.from_numpy(weight_tt).to(device)
        weight_st = torch.from_numpy(weight_st).to(device)

        kernels = self.guassian_kernel(source, target,
                                       kernel_mul=self.kernel_mul,
                                       kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        if torch.isnan(kernels).any() or torch.isinf(kernels).any():
            return torch.zeros((), device=device)

        ss = kernels[:batch_size, :batch_size]
        tt = kernels[batch_size:, batch_size:]
        st = kernels[:batch_size, batch_size:]

        if weight_ss.numel() == 1 and weight_ss.item() == 0:
            return torch.zeros((), device=device)

        loss = torch.sum(weight_ss * ss + weight_tt * tt - 2 * weight_st * st)
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.zeros((), device=device)
        return loss

    def convert_to_onehot(self, sca_label, class_num=12):
        return np.eye(class_num, dtype=np.float32)[sca_label]

    def cal_weight(self, s_label, t_label, class_num=12):
        batch_size = s_label.size(0)
        s_sca_label = s_label.detach().cpu().numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0, keepdims=True)
        s_sum[s_sum == 0] = 100.0
        s_vec_label = s_vec_label / s_sum

        t_prob = t_label.detach().cpu().numpy().astype(np.float32)
        t_sca_label = np.argmax(t_prob, axis=1)
        t_sum = np.sum(t_prob, axis=0, keepdims=True)
        t_sum[t_sum == 0] = 100.0
        t_vec_label = t_prob / t_sum

        index = list(set(s_sca_label.tolist()) & set(t_sca_label.tolist()))
        mask_arr = np.zeros((batch_size, class_num), dtype=np.float32)
        if len(index) > 0:
            mask_arr[:, index] = 1.0
            t_vec_label = t_vec_label * mask_arr
            s_vec_label = s_vec_label * mask_arr
            weight_ss = np.matmul(s_vec_label, s_vec_label.T) / len(index)
            weight_tt = np.matmul(t_vec_label, t_vec_label.T) / len(index)
            weight_st = np.matmul(s_vec_label, t_vec_label.T) / len(index)
        else:
            weight_ss = np.array([0], dtype=np.float32)
            weight_tt = np.array([0], dtype=np.float32)
            weight_st = np.array([0], dtype=np.float32)
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
