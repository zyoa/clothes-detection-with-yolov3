import torch
import torch.nn as nn
import utils.yolo_calc as yolo_calc

def make_conv(in_channels, out_channels, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            make_conv(channels, channels // 2, kernel_size=1, padding=0),
            make_conv(channels// 2, channels , kernel_size=3)
        )
    
    def forward(self, x):
        return x + self.block(x)


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.body = nn.Sequential(
            make_conv(3, 32, kernel_size=3),
            make_conv(32, 64, kernel_size=3, stride=2),
            ResidualBlock(channels=64),
            make_conv(64, 128, kernel_size=3, stride=2),
            ResidualBlock(channels=128),
            ResidualBlock(channels=128),
            make_conv(128, 256, kernel_size=3, stride=2),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            make_conv(256, 512, kernel_size=3, stride=2),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            make_conv(512, 1024, kernel_size=3, stride=2),
            ResidualBlock(channels=1024),
            ResidualBlock(channels=1024),
            ResidualBlock(channels=1024),
            ResidualBlock(channels=1024),
        )
    
    def forward(self, x):
        return self.body(x)


class YOLODetection(nn.Module):
    def __init__(self, anchors, img_size, n_class):
        super(YOLODetection, self).__init__()
        self.anchors = anchors
        self.n_anchor = len(anchors)
        self.img_size = img_size
        self.n_class = n_class
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.threshold = 0.5
        self.no_obj_weight = 100
        self.is_top_weight = 10
    
    def forward(self, x, target):
        device = torch.device('cuda' if x.is_cuda else 'cpu')
        n_batch = x.size(0)
        n_grid = x.size(2)
        stride = self.img_size / n_grid
        
        # x : [n_batch, n_anchor * (5 + n_class), n_grid, n_grid]
        # --> [n_batch, n_anchor, n_grid, n_grid, 5 + n_class]
        pred = x.view(n_batch, self.n_anchor, self.n_class + 5, n_grid, n_grid)\
                .permute(0, 1, 3, 4, 2).contiguous()
        
        # predicted grid
        pred_cx = torch.sigmoid(pred[..., 0])
        pred_cy = torch.sigmoid(pred[..., 1])
        pred_w = torch.sigmoid(pred[..., 2])
        pred_h = torch.sigmoid(pred[..., 3])
        
        # offsef of grid
        grid_offset = torch.arange(n_grid, dtype=torch.float, device=device).repeat(n_grid, 1)
        grid_x = grid_offset.view([1, 1, n_grid, n_grid])
        grid_y = grid_offset.t().view([1, 1, n_grid, n_grid])
        anchor_w = self.anchors[:, 0].view(1, -1, 1, 1)
        anchor_h = self.anchors[:, 1].view(1, -1, 1, 1)
        
        # [n_batch, n_anchor, n_grid, n_grid, ( [0:4](x, y, w, h), [4]conf, [5:]class) ]
        pred_grid = torch.zeros_like(pred[..., :4], device=device)
        pred_grid[..., 0] = grid_x + pred_cx
        pred_grid[..., 1] = grid_y + pred_cy
        pred_grid[..., 2] = anchor_w * torch.exp(pred_w)
        pred_grid[..., 3] = anchor_h * torch.exp(pred_h)
        pred_conf = torch.sigmoid(pred[..., 4])
        pred_class = torch.sigmoid(pred[..., 5:])
        
        # output form
        output = torch.cat(
            (pred_grid.view(n_batch, -1, 4) * stride,
             pred_conf.view(n_batch, -1, 1),
             pred_class.view(n_batch, -1, self.n_class)),
            -1)
        
        
        # if test phase
        if target is None:
            return output, 0
        
        
        # else train phase
        # select best anchor to target
        t_cx, t_cy = target[:, :2].t() * n_grid
        t_ci, t_cj = t_cx.long(), t_cy.long()
        t_w, t_h = target[:, 2:4].t()
        ious_anchor_target = torch.stack([self.iou_anchor_target(anchor, t_w, t_h) for anchor in self.anchors])
        _, best_iou_idx = ious_anchor_target.max(0)
        
        # set object mask
        batch_i = torch.arange(n_batch, dtype=torch.long, device=device)
        obj_mask = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, dtype=torch.bool, device=device)
        no_obj_mask = torch.ones(n_batch, self.n_anchor, n_grid, n_grid, dtype=torch.bool, device=device)
        
        obj_mask[batch_i, best_iou_idx, t_ci, t_cj] = 1
        no_obj_mask[batch_i, best_iou_idx, t_ci, t_cj] = 0
        
        for i, anchor_ious in enumerate(ious_anchor_target.t()):
            no_obj_mask[i, anchor_ious > self.threshold, t_ci[i], t_cj[i]] = 0
        
        # set target grid
        target_cx = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, dtype=torch.float, device=device)
        target_cy = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, dtype=torch.float, device=device)
        target_w = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, dtype=torch.float, device=device)
        target_h = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, dtype=torch.float, device=device)
        
        target_cx[batch_i, best_iou_idx, t_ci, t_cj] = t_cx - t_cx.floor()
        target_cy[batch_i, best_iou_idx, t_ci, t_cj] = t_cy - t_cy.floor()
        target_w[batch_i, best_iou_idx, t_ci, t_cj] = torch.log(t_w / self.anchors[best_iou_idx][:, 0] + 1e-16)
        target_h[batch_i, best_iou_idx, t_ci, t_cj] = torch.log(t_h / self.anchors[best_iou_idx][:, 1] + 1e-16)
        
        # target class
        target_class = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, self.n_class,
                                   dtype=torch.float, device=device)
        target_class[batch_i, best_iou_idx, t_ci, t_cj] = target[batch_i, 4:]
        
        # target conf
        target_conf = obj_mask.float()
        
        # bounding box loss
        loss_cx = self.mse_loss(pred_cx[obj_mask], target_cx[obj_mask])
        loss_cy = self.mse_loss(pred_cy[obj_mask], target_cy[obj_mask])
        loss_w = self.mse_loss(pred_w[obj_mask], target_w[obj_mask])
        loss_h = self.mse_loss(pred_h[obj_mask], target_h[obj_mask])
        loss_box = loss_cx + loss_cy + loss_w + loss_h
        
        # confidence loss --> obj + no_obj * 100 (weighted)
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], target_conf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], target_conf[no_obj_mask])
        loss_conf = loss_conf_obj + loss_conf_no_obj * self.no_obj_weight
        
        # class loss
        loss_class = self.bce_loss(pred_class[obj_mask], target_class[obj_mask])
        
        # total loss
        loss = loss_box + loss_conf + loss_class
        
        return output, loss
        
        
        # metric...        
        # class_mask = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, dtype=torch.float, device=device)
        # iou_scores = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, dtype=torch.float, device=device)
    
    def iou_anchor_target(self, anchor, t_w, t_h):
        a_w, a_h = anchor
        inter = torch.min(a_w, t_w) * torch.min(a_h, t_h)
        union = (t_w * t_h) + (a_w * a_h) - inter
        return inter / union


class YOLOv3(nn.Module):
    def __init__(self, anchors, img_size=416, n_class=1+51+41):
        super(YOLOv3, self).__init__()
        self.anchors = anchors
        self.last_out_channels = len(anchors) * (4 + 1 + n_class)
        
        self.darknet53 = Darknet53()
        self.detection_block = self.make_detection_block(1024, 512)
        self.yolo_detection = YOLODetection(anchors, img_size, n_class)
    
    def forward(self, x, target=None):
        x = self.darknet53(x)
        x = self.detection_block(x)
        output, loss = self.yolo_detection(x, target)
        
        return output, loss
    
    def make_detection_block(self, in_channels, out_channels):
        return nn.Sequential(
            make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            make_conv(out_channels, out_channels * 2, kernel_size=3),
            make_conv(out_channels * 2, out_channels, kernel_size=1, padding=0),
            make_conv(out_channels, out_channels * 2, kernel_size=3),
            make_conv(out_channels * 2, out_channels, kernel_size=1, padding=0),
            make_conv(out_channels, out_channels * 2, kernel_size=3),
            nn.Conv2d(out_channels * 2, self.last_out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )


if __name__ == '__main__':
    anchors = torch.tensor([[0.3, 0.3], [0.6, 0.3], [0.4, 0.6]])
    model = YOLOv3(anchors=anchors)
    # checkpoint = torch.load('model/my_trained_darknet_171.pt')
    # model.darknet53.body.load_state_dict(checkpoint['darknet53_body_state_dict'])

    x = torch.rand((1, 3, 160, 160))
    y = torch.rand((1, 97))
    print(model(x, y))
    
    checkpoint = {
        'epochs': 0,
        'state_dict': model.state_dict(),
        'optimizer': None,
        'scheduler': None,
        'metric': None
    }
    
    # torch.save(checkpoint, 'model/yolo_0.pt')