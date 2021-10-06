import torch
import torch.nn as nn

# Not used at present
class RegularizedLoss(nn.Module):
    def __init__(self, lambda_l2, lambda_embed, lambda_att):
        super().__init__()
        self.mse_output = nn.MSELoss()
        self.mse_embedding = nn.MSELoss()
        self. lambda_l2 = lambda_l2
        self.lambda_embed = lambda_embed
        self.lambda_att = lambda_att

    def forward(self, target, output, att_w, f_rgb, f_fl):
        l2_loss = self.mse_output(target, output)
        l_embed = self.mse_embedding((f_rgb / torch.norm(f_rgb)), (f_fl / torch.norm(f_fl)))
        l_att = torch.sum(torch.square(att_w))
        l_total = (self.lambda_l2 * l2_loss + self.lambda_embed * l_embed + self.lambda_att * l_att)
        return l_total