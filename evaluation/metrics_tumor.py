from fastai.vision import *
from sklearn.metrics import jaccard_score

#dynamic iou
def iou(outputs: torch.Tensor, labels: torch.Tensor, channels: list):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return tensor(np.mean(jaccard_score(to_np(outputs_max.view(-1)),to_np(labels_squeezed.view(-1)), average=None, labels=channels)))

