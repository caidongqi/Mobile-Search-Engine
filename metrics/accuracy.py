from metrics.base import BaseMetric
import torchmetrics

class Accuracy(BaseMetric):
    def __init__(self, num_classes, task="multiclass", average="micro", device="cpu", top_k=1) -> None:
        super().__init__()
        self.acc = torchmetrics.classification.Accuracy(task=task, num_classes=num_classes, average=average, top_k=top_k).to(device)
    
    def update(self, pred, target):
        step_acc = self.acc(pred, target)
        return step_acc
    
    def compute(self):
        self.result = self.acc.compute()

        