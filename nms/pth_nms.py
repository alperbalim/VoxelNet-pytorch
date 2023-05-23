import torch
import numpy as np
from torchvision.ops import nms

# changed to use with torchvision nms
# torchvision.ops.nms(boxes: Tensor, scores: Tensor, iou_threshold: float)

def pth_nms(dets, thresh):
  """
  dets has to be a tensor

  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  """
  boxes=torch.FloatTensor(dets[:,:4])
  scores=torch.FloatTensor(dets[:,4])
  
  keep = nms(boxes, scores, thresh)

  return keep


