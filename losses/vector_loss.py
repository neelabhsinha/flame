import torch
import torch.nn as nn


def get_vector(vector):
    v = torch.cat(
        [(torch.cos(vector[:, 1]) * torch.sin(vector[:, 0])).unsqueeze(-1), torch.sin(vector[:, 1])[..., None],
         (torch.cos(vector[:, 1]) * torch.cos(vector[:, 0])).unsqueeze(-1)], dim=-1)
    return v


class VectorDifferenceLoss(nn.Module):
    def __init__(self, data_stats):
        super(VectorDifferenceLoss, self).__init__()
        self.data_stats = data_stats

    def forward(self, target, pred):
        y_p = pred * (self.data_stats['gaze']['max'] - self.data_stats['gaze']['min']) + self.data_stats['gaze']['min']
        y_t = target * (self.data_stats['gaze']['max'] - self.data_stats['gaze']['min']) + self.data_stats['gaze'][
            'min']
        v_p = get_vector(y_p)
        v_t = get_vector(y_t)
        loss = torch.sum(torch.square(v_p - v_t))
        return loss
