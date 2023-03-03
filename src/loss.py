import mindspore
import mindspore.nn as nn
from mindspore import nn, ops
# Generalized Cross Entropy Loss
class GCELoss(nn.Cell):

    def __init__(self, q=0.7, ignore_index=-100):
        super(GCELoss, self).__init__()
        self.q = q
        self.ignore_index = ignore_index
        self.softmax = nn.Softmax(axis=1)
    def construct(self, logits, targets):
        # vanilla cross entropy when q = 0
        if self.q == 0:
            if logits.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(logits.view(-1), targets.float())
            else:
                ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
                loss = ce_loss(logits, targets)
        else:
            # if logits.size(-1) == 1:
            #     pred = torch.sigmoid(logits)
            #     pred = torch.cat((1-pred, pred), dim=-1)
            # else:
            if len(targets) > 1:
                targets = ops.unsqueeze(mindspore.Tensor(targets[0]),dim=0)
            pred = self.softmax(logits)
            ce_loss = nn.NLLLoss(reduction='none')
            loss = -ce_loss(pred,targets)
            loss = (1-loss**self.q) / self.q
        #loss = (loss.view(-1)).sum()
        return loss
