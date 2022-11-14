import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        # w1 = torch.flatten(x1)
        # w2 = torch.flatten(x2)
        # return torch.matmul(w1, w2)/(torch.linalg.norm(w1)*torch.linalg.norm(w2))
        return torch.linalg.norm(x1-x2)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        
        losses = torch.max(torch.tensor(0.0), (distance_positive - distance_negative + self.margin))
            

        return losses