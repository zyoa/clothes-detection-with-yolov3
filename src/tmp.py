import torch
import numpy as np
import torch.nn.functional as F

def pad_to_square(image, pad_value=0):
    _, h, w = image.shape

    # 너비와 높이의 차
    difference = abs(h - w)

    # (top, bottom) padding or (left, right) padding
    if h <= w:
        top = difference // 2
        bottom = difference - difference // 2
        pad = [0, 0, top, bottom]
    else:
        left = difference // 2
        right = difference - difference // 2
        pad = [left, right, 0, 0]

    # Add padding
    image = F.pad(image, pad, mode='constant', value=pad_value)
    return image, pad

def getitem():
    
    image = torch.tensor([
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
    ])
    
    _, h, w = image.shape
    h_factor, w_factor = (h, w)

    # Pad to square resolution
    image, pad = pad_to_square(image)
    _, padded_h, padded_w = image.shape

    # 2. Label
    # -----------------------------------------------------------------------------------
    
    boxes = torch.tensor([[1., 2., 3, 4, 5], [6.,7,8,9,19]])

    # Extract coordinates for unpadded + unscaled image
    x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
    y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
    x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
    y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

    # Adjust for added padding
    x1 += pad[0]
    y1 += pad[2]
    x2 += pad[1]
    y2 += pad[3]

    # Returns (x, y, w, h)
    boxes[:, 1] = ((x1 + x2) / 2) / padded_w
    boxes[:, 2] = ((y1 + y2) / 2) / padded_h
    boxes[:, 3] *= w_factor / padded_w
    boxes[:, 4] *= h_factor / padded_h

    targets = torch.zeros((len(boxes), 6))
    targets[:, 1:] = boxes

    return image, targets

print(getitem())

t = torch.zeros(3, 3, 3)
print(t)

b = torch.tensor([0,1,2])
i = torch.tensor([0, 1, 2])
j = torch.tensor([2, 1, 0])

t[b, i, j] = 1

print(t)


# obj_mask[b, best_ious_idx, gj, gi] = 1