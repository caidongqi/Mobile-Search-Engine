from metrics.base import BaseMetric
from torchmetrics import ConfusionMatrix
import torch

class Search(BaseMetric):
    def __init__(self, num_classes, device='cpu', metrics=['precision', 'recall'], target_classes=[], task="multiclass") -> None:
        super().__init__()
        self.matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
        self.metrics = metrics
        self.target_classes = target_classes
        self.pricision = None
        self.recall = None
    
    def update(self, pred, target):
        self.matrix(pred, target)
    
    def compute(self):
        self.result = self.matrix.compute()
        if 'precision' in self.metrics:
            TP_FP = torch.sum(self.result, 0)
            self.pricision = [self.result[idx][idx].item() / TP_FP[idx].item() for idx in self.target_classes]
        
        if 'recall' in self.metrics:
            TP_FN = torch.sum(self.result, 1)
            self.recall = [self.result[idx][idx].item() / TP_FN[idx].item() for idx in self.target_classes]
            
