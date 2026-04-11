import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

def extract_roi_features(feat_dict, mask, key_i, spatial_scale=None, output_size=(7, 7)):
    batch_size, _, seq_len, mask_h, mask_w = mask.shape
    _, _, feat_c, feat_h, feat_w = feat_dict[key_i].shape
    output_size = (feat_h, feat_w)
    if spatial_scale is None:
        spatial_scale = feat_h / mask_h
    features = feat_dict[key_i].permute(0, 1, 2, 3, 4).reshape(batch_size * seq_len, feat_c, feat_h, feat_w)
    all_boxes = []
    for b in range(batch_size):
        for s in range(seq_len):
            m = mask[b, 0, s]
            if not torch.any(m):
                all_boxes.append(torch.tensor([0, 0, 1, 1]))
                continue
            indices = torch.nonzero(m, as_tuple=False)
            if indices.shape[0] == 0:
                all_boxes.append(torch.tensor([0, 0, 1, 1]))
                continue
            y_min, y_max = (indices[:, 0].min().item(), indices[:, 0].max().item())
            x_min, x_max = (indices[:, 1].min().item(), indices[:, 1].max().item())
            box = torch.tensor([x_min, y_min, x_max + 1, y_max + 1])
            all_boxes.append(box)
    boxes = torch.stack(all_boxes)
    batch_indices = torch.arange(batch_size, device=boxes.device).repeat_interleave(seq_len)
    rois = torch.cat([batch_indices.unsqueeze(1), boxes], dim=1).float().to(feat_dict[key_i].device)
    roi_features = ops.roi_align(features, rois, output_size=output_size, spatial_scale=spatial_scale, aligned=True)
    roi_features = roi_features.reshape(batch_size, seq_len, -1, output_size[0], output_size[1])
    return roi_features

class PDecoder(nn.Module):

    def __init__(self):
        super(PDecoder, self).__init__()
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(72 * 128, 1)

    def forward(self, feat_dict, mask, full_resolution=0):
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 1:
            for key_i in feat_dict.keys():
                org_feat_h, org_feat_w = (feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                feat_dict[key_i] = F.interpolate(feat_dict[key_i].reshape(-1, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4]), size=(int(mask.shape[-2] / 2), int(mask.shape[-1] / 2)), mode='bilinear', align_corners=False)
                feat_dict[key_i] = feat_dict[key_i].reshape(mask.shape[0], -1, feat_dict[key_i].shape[-3], feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
                feat_dict[key_i] = F.interpolate(feat_dict[key_i], size=(org_feat_h, org_feat_w), mode='bilinear', align_corners=False)
        elif full_resolution == 2:
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 3:
            for key_i in feat_dict.keys():
                roi_features = extract_roi_features(feat_dict, mask, key_i)
                feat_dict[key_i] = roi_features
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PDecoderSupElasticity(nn.Module):

    def __init__(self):
        super(PDecoderSupElasticity, self).__init__()
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280 + 3, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640 + 3, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640 + 2, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.mask_convs = nn.ModuleList([nn.ModuleList([nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.Conv2d(16, 1, kernel_size=3, padding=1)]) for _ in range(3)])
        self.shared_fc1 = nn.Linear(72 * 128 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        self.temporal_key_points_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        self.height_values_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        self.height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        self.normed_height_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        self.normed_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        self.normed_sqrt_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        self.gt_retitusion_fc = nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)])

    def forward(self, feat_dict, mask, full_resolution=0):
        for key_i in feat_dict.keys():
            mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
            feat_dict[key_i] = torch.cat([feat_dict[key_i], mask_.unsqueeze(2)], dim=2)
            feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        key_masks_list = []
        for i in range(3):
            mask_feat = F.relu(self.mask_convs[i][0](x))
            mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
            mask = self.mask_convs[i][2](mask_feat)
            key_masks_list.append(mask)
        key_masks = torch.cat(key_masks_list, dim=1)
        x_flat = x.view(x.size(0), -1)
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        temporal_key_points_list = []
        for i in range(3):
            x = F.relu(self.temporal_key_points_fc[i][0](shared_feat))
            x = F.relu(self.temporal_key_points_fc[i][1](x))
            x = self.temporal_key_points_fc[i][2](x)
            temporal_key_points_list.append(x)
        temporal_key_points = torch.cat(temporal_key_points_list, dim=1)
        height_values_list = []
        for i in range(3):
            x = F.relu(self.height_values_fc[i][0](shared_feat))
            x = F.relu(self.height_values_fc[i][1](x))
            x = self.height_values_fc[i][2](x)
            height_values_list.append(x)
        height_values = torch.cat(height_values_list, dim=1)
        height_differences_list = []
        for i in range(2):
            x = F.relu(self.height_differences_fc[i][0](shared_feat))
            x = F.relu(self.height_differences_fc[i][1](x))
            x = self.height_differences_fc[i][2](x)
            height_differences_list.append(x)
        height_differences = torch.cat(height_differences_list, dim=1)
        normed_height_list = []
        for i in range(3):
            x = F.relu(self.normed_height_fc[i][0](shared_feat))
            x = F.relu(self.normed_height_fc[i][1](x))
            x = self.normed_height_fc[i][2](x)
            normed_height_list.append(x)
        normed_height = torch.cat(normed_height_list, dim=1)
        normed_height_differences_list = []
        for i in range(2):
            x = F.relu(self.normed_height_differences_fc[i][0](shared_feat))
            x = F.relu(self.normed_height_differences_fc[i][1](x))
            x = self.normed_height_differences_fc[i][2](x)
            normed_height_differences_list.append(x)
        normed_height_differences = torch.cat(normed_height_differences_list, dim=1)
        normed_sqrt_height_differences_list = []
        for i in range(2):
            x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
            x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
            x = self.normed_sqrt_height_differences_fc[i][2](x)
            normed_sqrt_height_differences_list.append(x)
        normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        x = F.relu(self.gt_retitusion_fc[0](shared_feat))
        x = F.relu(self.gt_retitusion_fc[1](x))
        gt_retitusion = self.gt_retitusion_fc[2](x)
        return (temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks)

class PDecoderSupElasticityv1(nn.Module):

    def __init__(self):
        super(PDecoderSupElasticityv1, self).__init__()
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.mask_convs = nn.ModuleList([nn.ModuleList([nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.Conv2d(16, 1, kernel_size=3, padding=1)]) for _ in range(3)])
        self.shared_fc1 = nn.Linear(72 * 128 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        self.temporal_key_points_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        self.height_values_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        self.height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        self.normed_height_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        self.normed_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        self.normed_sqrt_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        self.gt_retitusion_fc = nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)])

    def forward(self, feat_dict, mask, full_resolution=0):
        for key_i in feat_dict.keys():
            mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
            feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
            feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        key_masks_list = []
        for i in range(3):
            mask_feat = F.relu(self.mask_convs[i][0](x))
            mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
            mask = self.mask_convs[i][2](mask_feat)
            key_masks_list.append(mask)
        key_masks = torch.cat(key_masks_list, dim=1)
        x_flat = x.view(x.size(0), -1)
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        temporal_key_points_list = []
        for i in range(3):
            x = F.relu(self.temporal_key_points_fc[i][0](shared_feat))
            x = F.relu(self.temporal_key_points_fc[i][1](x))
            x = self.temporal_key_points_fc[i][2](x)
            temporal_key_points_list.append(x)
        temporal_key_points = torch.cat(temporal_key_points_list, dim=1)
        height_values_list = []
        for i in range(3):
            x = F.relu(self.height_values_fc[i][0](shared_feat))
            x = F.relu(self.height_values_fc[i][1](x))
            x = self.height_values_fc[i][2](x)
            height_values_list.append(x)
        height_values = torch.cat(height_values_list, dim=1)
        height_differences_list = []
        for i in range(2):
            x = F.relu(self.height_differences_fc[i][0](shared_feat))
            x = F.relu(self.height_differences_fc[i][1](x))
            x = self.height_differences_fc[i][2](x)
            height_differences_list.append(x)
        height_differences = torch.cat(height_differences_list, dim=1)
        normed_height_list = []
        for i in range(3):
            x = F.relu(self.normed_height_fc[i][0](shared_feat))
            x = F.relu(self.normed_height_fc[i][1](x))
            x = self.normed_height_fc[i][2](x)
            normed_height_list.append(x)
        normed_height = torch.cat(normed_height_list, dim=1)
        normed_height_differences_list = []
        for i in range(2):
            x = F.relu(self.normed_height_differences_fc[i][0](shared_feat))
            x = F.relu(self.normed_height_differences_fc[i][1](x))
            x = self.normed_height_differences_fc[i][2](x)
            normed_height_differences_list.append(x)
        normed_height_differences = torch.cat(normed_height_differences_list, dim=1)
        normed_sqrt_height_differences_list = []
        for i in range(2):
            x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
            x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
            x = self.normed_sqrt_height_differences_fc[i][2](x)
            normed_sqrt_height_differences_list.append(x)
        normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        x = F.relu(self.gt_retitusion_fc[0](shared_feat))
        x = F.relu(self.gt_retitusion_fc[1](x))
        gt_retitusion = self.gt_retitusion_fc[2](x)
        return (temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks)

class PDecoderSupElasticityv2(nn.Module):

    def __init__(self, temporal_key_points=True, height_values=True, height_differences=True, normed_height=True, normed_height_differences=True, normed_sqrt_height_differences=True, keytime_masks=True, full_resolution=0):
        super(PDecoderSupElasticityv2, self).__init__()
        self.temporal_key_points = temporal_key_points
        self.height_values = height_values
        self.height_differences = height_differences
        self.normed_height = normed_height
        self.normed_height_differences = normed_height_differences
        self.normed_sqrt_height_differences = normed_sqrt_height_differences
        self.keytime_masks = keytime_masks
        self.full_resolution = full_resolution
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        if self.keytime_masks:
            self.mask_convs = nn.ModuleList([nn.ModuleList([nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.Conv2d(16, 1, kernel_size=3, padding=1)]) for _ in range(3)])
        self.shared_fc1 = nn.Linear(72 * 128 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        if self.temporal_key_points:
            self.temporal_key_points_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.height_values:
            self.height_values_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.height_differences:
            self.height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.normed_height:
            self.normed_height_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.normed_height_differences:
            self.normed_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.normed_sqrt_height_differences:
            self.normed_sqrt_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        self.gt_retitusion_fc = nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)])

    def forward(self, feat_dict, mask):
        if self.full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif self.full_resolution == 1:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].var(dim=1)
        elif self.full_resolution == 2:
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].var(dim=1)
                inf_mask = torch.isinf(feat_dict[key_i])
                feat_dict[key_i] = torch.where(inf_mask, torch.tensor(torch.finfo(feat_dict[key_i].dtype).max, device=feat_dict[key_i].device, dtype=feat_dict[key_i].dtype), feat_dict[key_i])
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        if self.keytime_masks:
            key_masks_list = []
            for i in range(3):
                mask_feat = F.relu(self.mask_convs[i][0](x))
                mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
                mask = self.mask_convs[i][2](mask_feat)
                key_masks_list.append(mask)
            key_masks = torch.cat(key_masks_list, dim=1)
        else:
            key_masks = None
        x_flat = x.reshape(x.size(0), -1)
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        if self.temporal_key_points:
            temporal_key_points_list = []
            for i in range(3):
                x = F.relu(self.temporal_key_points_fc[i][0](shared_feat))
                x = F.relu(self.temporal_key_points_fc[i][1](x))
                x = self.temporal_key_points_fc[i][2](x)
                temporal_key_points_list.append(x)
            temporal_key_points = torch.cat(temporal_key_points_list, dim=1)
        else:
            temporal_key_points = None
        if self.height_values:
            height_values_list = []
            for i in range(3):
                x = F.relu(self.height_values_fc[i][0](shared_feat))
                x = F.relu(self.height_values_fc[i][1](x))
                x = self.height_values_fc[i][2](x)
                height_values_list.append(x)
            height_values = torch.cat(height_values_list, dim=1)
        else:
            height_values = None
        if self.height_differences:
            height_differences_list = []
            for i in range(2):
                x = F.relu(self.height_differences_fc[i][0](shared_feat))
                x = F.relu(self.height_differences_fc[i][1](x))
                x = self.height_differences_fc[i][2](x)
                height_differences_list.append(x)
            height_differences = torch.cat(height_differences_list, dim=1)
        else:
            height_differences = None
        if self.normed_height:
            normed_height_list = []
            for i in range(3):
                x = F.relu(self.normed_height_fc[i][0](shared_feat))
                x = F.relu(self.normed_height_fc[i][1](x))
                x = self.normed_height_fc[i][2](x)
                normed_height_list.append(x)
            normed_height = torch.cat(normed_height_list, dim=1)
        else:
            normed_height = None
        if self.normed_height_differences:
            normed_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_height_differences_fc[i][1](x))
                x = self.normed_height_differences_fc[i][2](x)
                normed_height_differences_list.append(x)
            normed_height_differences = torch.cat(normed_height_differences_list, dim=1)
        else:
            normed_height_differences = None
        if self.normed_sqrt_height_differences:
            normed_sqrt_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
                x = self.normed_sqrt_height_differences_fc[i][2](x)
                normed_sqrt_height_differences_list.append(x)
            normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        else:
            normed_sqrt_height_differences = None
        x = F.relu(self.gt_retitusion_fc[0](shared_feat))
        x = F.relu(self.gt_retitusion_fc[1](x))
        gt_retitusion = self.gt_retitusion_fc[2](x)
        return (temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks)

class PDecoderSupElasticityv2512(nn.Module):

    def __init__(self, temporal_key_points=True, height_values=True, height_differences=True, normed_height=True, normed_height_differences=True, normed_sqrt_height_differences=True, keytime_masks=True, full_resolution=0):
        super(PDecoderSupElasticityv2512, self).__init__()
        self.temporal_key_points = temporal_key_points
        self.height_values = height_values
        self.height_differences = height_differences
        self.normed_height = normed_height
        self.normed_height_differences = normed_height_differences
        self.normed_sqrt_height_differences = normed_sqrt_height_differences
        self.keytime_masks = keytime_masks
        self.full_resolution = full_resolution
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        if self.keytime_masks:
            self.mask_convs = nn.ModuleList([nn.ModuleList([nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.Conv2d(16, 1, kernel_size=3, padding=1)]) for _ in range(3)])
        self.shared_fc1 = nn.Linear(40 * 64 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        if self.temporal_key_points:
            self.temporal_key_points_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.height_values:
            self.height_values_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.height_differences:
            self.height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.normed_height:
            self.normed_height_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.normed_height_differences:
            self.normed_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.normed_sqrt_height_differences:
            self.normed_sqrt_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        self.gt_retitusion_fc = nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)])

    def forward(self, feat_dict, mask):
        if self.full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif self.full_resolution == 1:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].var(dim=1)
        elif self.full_resolution == 2:
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].var(dim=1)
                inf_mask = torch.isinf(feat_dict[key_i])
                feat_dict[key_i] = torch.where(inf_mask, torch.tensor(torch.finfo(feat_dict[key_i].dtype).max, device=feat_dict[key_i].device, dtype=feat_dict[key_i].dtype), feat_dict[key_i])
        x3_up = F.interpolate(feat_dict[3], size=(10, 16), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(20, 32), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(40, 64), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        if self.keytime_masks:
            key_masks_list = []
            for i in range(3):
                mask_feat = F.relu(self.mask_convs[i][0](x))
                mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
                mask = self.mask_convs[i][2](mask_feat)
                key_masks_list.append(mask)
            key_masks = torch.cat(key_masks_list, dim=1)
        else:
            key_masks = None
        x_flat = x.reshape(x.size(0), -1)
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        if self.temporal_key_points:
            temporal_key_points_list = []
            for i in range(3):
                x = F.relu(self.temporal_key_points_fc[i][0](shared_feat))
                x = F.relu(self.temporal_key_points_fc[i][1](x))
                x = self.temporal_key_points_fc[i][2](x)
                temporal_key_points_list.append(x)
            temporal_key_points = torch.cat(temporal_key_points_list, dim=1)
        else:
            temporal_key_points = None
        if self.height_values:
            height_values_list = []
            for i in range(3):
                x = F.relu(self.height_values_fc[i][0](shared_feat))
                x = F.relu(self.height_values_fc[i][1](x))
                x = self.height_values_fc[i][2](x)
                height_values_list.append(x)
            height_values = torch.cat(height_values_list, dim=1)
        else:
            height_values = None
        if self.height_differences:
            height_differences_list = []
            for i in range(2):
                x = F.relu(self.height_differences_fc[i][0](shared_feat))
                x = F.relu(self.height_differences_fc[i][1](x))
                x = self.height_differences_fc[i][2](x)
                height_differences_list.append(x)
            height_differences = torch.cat(height_differences_list, dim=1)
        else:
            height_differences = None
        if self.normed_height:
            normed_height_list = []
            for i in range(3):
                x = F.relu(self.normed_height_fc[i][0](shared_feat))
                x = F.relu(self.normed_height_fc[i][1](x))
                x = self.normed_height_fc[i][2](x)
                normed_height_list.append(x)
            normed_height = torch.cat(normed_height_list, dim=1)
        else:
            normed_height = None
        if self.normed_height_differences:
            normed_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_height_differences_fc[i][1](x))
                x = self.normed_height_differences_fc[i][2](x)
                normed_height_differences_list.append(x)
            normed_height_differences = torch.cat(normed_height_differences_list, dim=1)
        else:
            normed_height_differences = None
        if self.normed_sqrt_height_differences:
            normed_sqrt_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
                x = self.normed_sqrt_height_differences_fc[i][2](x)
                normed_sqrt_height_differences_list.append(x)
            normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        else:
            normed_sqrt_height_differences = None
        x = F.relu(self.gt_retitusion_fc[0](shared_feat))
        x = F.relu(self.gt_retitusion_fc[1](x))
        gt_retitusion = self.gt_retitusion_fc[2](x)
        return (temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks)

class PDecoderSupElasticityv3(nn.Module):

    def __init__(self, temporal_key_points=True, height_values=True, height_differences=True, normed_height=True, normed_height_differences=True, normed_sqrt_height_differences=True, keytime_masks=True, full_resolution=0, retitution=True):
        super(PDecoderSupElasticityv3, self).__init__()
        self.temporal_key_points = temporal_key_points
        self.height_values = height_values
        self.height_differences = height_differences
        self.normed_height = normed_height
        self.normed_height_differences = normed_height_differences
        self.normed_sqrt_height_differences = normed_sqrt_height_differences
        self.keytime_masks = keytime_masks
        self.full_resolution = full_resolution
        self.retitution = retitution
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.temporal_attn = nn.Parameter(torch.randn((64, 72, 128)) * 0.02)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        if self.keytime_masks:
            self.mask_convs = nn.ModuleList([nn.ModuleList([nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.Conv2d(16, 1, kernel_size=3, padding=1)]) for _ in range(3)])
        self.shared_fc1 = nn.Linear(72 * 128 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        if self.temporal_key_points:
            self.temporal_key_points_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.height_values:
            self.height_values_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.height_differences:
            self.height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.normed_height:
            self.normed_height_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.normed_height_differences:
            self.normed_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.normed_sqrt_height_differences:
            self.normed_sqrt_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.retitution:
            self.gt_retitusion_fc = nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)])

    def temporal_weighted_pooling(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T, C, H * W)
        x_perm = x_flat.permute(0, 1, 3, 2)
        attn_flat = self.temporal_attn.reshape(C, H * W)
        attn_perm = attn_flat.t()
        x_norm = F.normalize(x_perm, p=2, dim=-1)
        attn_norm = F.normalize(attn_perm, p=2, dim=-1)
        attn_norm_expanded = attn_norm.unsqueeze(0).unsqueeze(0)
        similarity = torch.sum(x_norm * attn_norm_expanded, dim=-1)
        weights = F.softmax(similarity, dim=1)
        weights_expanded = weights.unsqueeze(-1)
        weighted_sum = torch.sum(weights_expanded * x_perm, dim=1)
        output = weighted_sum.permute(0, 2, 1).reshape(B, C, H, W)
        return output

    def forward(self, feat_dict, mask):
        B = feat_dict[0].shape[0]
        T = feat_dict[0].shape[1]
        if self.full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])
        else:
            assert self.full_resolution == 1
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3])
        x = self.temporal_weighted_pooling(x)
        if self.keytime_masks:
            key_masks_list = []
            for i in range(3):
                mask_feat = F.relu(self.mask_convs[i][0](x))
                mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
                mask = self.mask_convs[i][2](mask_feat)
                key_masks_list.append(mask)
            key_masks = torch.cat(key_masks_list, dim=1)
        else:
            key_masks = None
        x_flat = x.reshape(x.size(0), -1)
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        if self.temporal_key_points:
            temporal_key_points_list = []
            for i in range(3):
                x = F.relu(self.temporal_key_points_fc[i][0](shared_feat))
                x = F.relu(self.temporal_key_points_fc[i][1](x))
                x = self.temporal_key_points_fc[i][2](x)
                temporal_key_points_list.append(x)
            temporal_key_points = torch.cat(temporal_key_points_list, dim=1)
        else:
            temporal_key_points = None
        if self.height_values:
            height_values_list = []
            for i in range(3):
                x = F.relu(self.height_values_fc[i][0](shared_feat))
                x = F.relu(self.height_values_fc[i][1](x))
                x = self.height_values_fc[i][2](x)
                height_values_list.append(x)
            height_values = torch.cat(height_values_list, dim=1)
        else:
            height_values = None
        if self.height_differences:
            height_differences_list = []
            for i in range(2):
                x = F.relu(self.height_differences_fc[i][0](shared_feat))
                x = F.relu(self.height_differences_fc[i][1](x))
                x = self.height_differences_fc[i][2](x)
                height_differences_list.append(x)
            height_differences = torch.cat(height_differences_list, dim=1)
        else:
            height_differences = None
        if self.normed_height:
            normed_height_list = []
            for i in range(3):
                x = F.relu(self.normed_height_fc[i][0](shared_feat))
                x = F.relu(self.normed_height_fc[i][1](x))
                x = self.normed_height_fc[i][2](x)
                normed_height_list.append(x)
            normed_height = torch.cat(normed_height_list, dim=1)
        else:
            normed_height = None
        if self.normed_height_differences:
            normed_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_height_differences_fc[i][1](x))
                x = self.normed_height_differences_fc[i][2](x)
                normed_height_differences_list.append(x)
            normed_height_differences = torch.cat(normed_height_differences_list, dim=1)
        else:
            normed_height_differences = None
        if self.normed_sqrt_height_differences:
            normed_sqrt_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
                x = self.normed_sqrt_height_differences_fc[i][2](x)
                normed_sqrt_height_differences_list.append(x)
            normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        else:
            normed_sqrt_height_differences = None
        if self.retitution:
            x = F.relu(self.gt_retitusion_fc[0](shared_feat))
            x = F.relu(self.gt_retitusion_fc[1](x))
            gt_retitusion = self.gt_retitusion_fc[2](x)
        else:
            gt_retitusion = None
        return (temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks)

class PDecoderSupElasticityv3512(nn.Module):

    def __init__(self, temporal_key_points=True, height_values=True, height_differences=True, normed_height=True, normed_height_differences=True, normed_sqrt_height_differences=True, keytime_masks=True, full_resolution=0):
        super(PDecoderSupElasticityv3512, self).__init__()
        self.temporal_key_points = temporal_key_points
        self.height_values = height_values
        self.height_differences = height_differences
        self.normed_height = normed_height
        self.normed_height_differences = normed_height_differences
        self.normed_sqrt_height_differences = normed_sqrt_height_differences
        self.keytime_masks = keytime_masks
        self.full_resolution = full_resolution
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.temporal_attn = nn.Parameter(torch.randn((64, 40, 64)) * 0.02)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        if self.keytime_masks:
            self.mask_convs = nn.ModuleList([nn.ModuleList([nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.Conv2d(16, 1, kernel_size=3, padding=1)]) for _ in range(3)])
        self.shared_fc1 = nn.Linear(40 * 64 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        if self.temporal_key_points:
            self.temporal_key_points_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.height_values:
            self.height_values_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.height_differences:
            self.height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.normed_height:
            self.normed_height_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.normed_height_differences:
            self.normed_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.normed_sqrt_height_differences:
            self.normed_sqrt_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        self.gt_retitusion_fc = nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)])

    def temporal_weighted_pooling(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T, C, H * W)
        x_perm = x_flat.permute(0, 1, 3, 2)
        attn_flat = self.temporal_attn.reshape(C, H * W)
        attn_perm = attn_flat.t()
        x_norm = F.normalize(x_perm, p=2, dim=-1)
        attn_norm = F.normalize(attn_perm, p=2, dim=-1)
        attn_norm_expanded = attn_norm.unsqueeze(0).unsqueeze(0)
        similarity = torch.sum(x_norm * attn_norm_expanded, dim=-1)
        weights = F.softmax(similarity, dim=1)
        weights_expanded = weights.unsqueeze(-1)
        weighted_sum = torch.sum(weights_expanded * x_perm, dim=1)
        output = weighted_sum.permute(0, 2, 1).reshape(B, C, H, W)
        return output

    def forward(self, feat_dict, mask):
        B = feat_dict[0].shape[0]
        T = feat_dict[0].shape[1]
        if self.full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])
        else:
            assert self.full_resolution == 1
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])
        x3_up = F.interpolate(feat_dict[3], size=(10, 16), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(20, 32), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(40, 64), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3])
        x = self.temporal_weighted_pooling(x)
        if self.keytime_masks:
            key_masks_list = []
            for i in range(3):
                mask_feat = F.relu(self.mask_convs[i][0](x))
                mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
                mask = self.mask_convs[i][2](mask_feat)
                key_masks_list.append(mask)
            key_masks = torch.cat(key_masks_list, dim=1)
        else:
            key_masks = None
        x_flat = x.reshape(x.size(0), -1)
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        if self.temporal_key_points:
            temporal_key_points_list = []
            for i in range(3):
                x = F.relu(self.temporal_key_points_fc[i][0](shared_feat))
                x = F.relu(self.temporal_key_points_fc[i][1](x))
                x = self.temporal_key_points_fc[i][2](x)
                temporal_key_points_list.append(x)
            temporal_key_points = torch.cat(temporal_key_points_list, dim=1)
        else:
            temporal_key_points = None
        if self.height_values:
            height_values_list = []
            for i in range(3):
                x = F.relu(self.height_values_fc[i][0](shared_feat))
                x = F.relu(self.height_values_fc[i][1](x))
                x = self.height_values_fc[i][2](x)
                height_values_list.append(x)
            height_values = torch.cat(height_values_list, dim=1)
        else:
            height_values = None
        if self.height_differences:
            height_differences_list = []
            for i in range(2):
                x = F.relu(self.height_differences_fc[i][0](shared_feat))
                x = F.relu(self.height_differences_fc[i][1](x))
                x = self.height_differences_fc[i][2](x)
                height_differences_list.append(x)
            height_differences = torch.cat(height_differences_list, dim=1)
        else:
            height_differences = None
        if self.normed_height:
            normed_height_list = []
            for i in range(3):
                x = F.relu(self.normed_height_fc[i][0](shared_feat))
                x = F.relu(self.normed_height_fc[i][1](x))
                x = self.normed_height_fc[i][2](x)
                normed_height_list.append(x)
            normed_height = torch.cat(normed_height_list, dim=1)
        else:
            normed_height = None
        if self.normed_height_differences:
            normed_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_height_differences_fc[i][1](x))
                x = self.normed_height_differences_fc[i][2](x)
                normed_height_differences_list.append(x)
            normed_height_differences = torch.cat(normed_height_differences_list, dim=1)
        else:
            normed_height_differences = None
        if self.normed_sqrt_height_differences:
            normed_sqrt_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
                x = self.normed_sqrt_height_differences_fc[i][2](x)
                normed_sqrt_height_differences_list.append(x)
            normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        else:
            normed_sqrt_height_differences = None
        x = F.relu(self.gt_retitusion_fc[0](shared_feat))
        x = F.relu(self.gt_retitusion_fc[1](x))
        gt_retitusion = self.gt_retitusion_fc[2](x)
        return (temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks)

class PDecoderSupElasticityv4(nn.Module):

    def __init__(self, temporal_key_points=True, height_values=True, height_differences=True, normed_height=True, normed_height_differences=True, normed_sqrt_height_differences=True, keytime_masks=True, full_resolution=0, retitution=True):
        super(PDecoderSupElasticityv4, self).__init__()
        self.temporal_key_points = temporal_key_points
        self.height_values = height_values
        self.height_differences = height_differences
        self.normed_height = normed_height
        self.normed_height_differences = normed_height_differences
        self.normed_sqrt_height_differences = normed_sqrt_height_differences
        self.keytime_masks = keytime_masks
        self.full_resolution = full_resolution
        self.retitution = retitution
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.temporal_attn = nn.Parameter(torch.randn((64, 1, 1)) * 0.02)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        if self.keytime_masks:
            self.mask_convs = nn.ModuleList([nn.ModuleList([nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.Conv2d(16, 1, kernel_size=3, padding=1)]) for _ in range(3)])
        self.shared_fc1 = nn.Linear(72 * 128 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        if self.temporal_key_points:
            self.temporal_key_points_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.height_values:
            self.height_values_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.height_differences:
            self.height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.normed_height:
            self.normed_height_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(3)])
        if self.normed_height_differences:
            self.normed_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.normed_sqrt_height_differences:
            self.normed_sqrt_height_differences_fc = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)]) for _ in range(2)])
        if self.retitution:
            self.gt_retitusion_fc = nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 1)])

    def temporal_weighted_pooling(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T, C, H * W)
        x_perm = x_flat.permute(0, 1, 3, 2)
        attn_flat = self.temporal_attn.expand(C, H, W).reshape(C, H * W)
        attn_perm = attn_flat.t()
        x_norm = F.normalize(x_perm, p=2, dim=-1)
        attn_norm = F.normalize(attn_perm, p=2, dim=-1)
        attn_norm_expanded = attn_norm.unsqueeze(0).unsqueeze(0)
        similarity = torch.sum(x_norm * attn_norm_expanded, dim=-1)
        weights = F.softmax(similarity, dim=1)
        weights_expanded = weights.unsqueeze(-1)
        weighted_sum = torch.sum(weights_expanded * x_perm, dim=1)
        output = weighted_sum.permute(0, 2, 1).reshape(B, C, H, W)
        return output

    def forward(self, feat_dict, mask):
        B = feat_dict[0].shape[0]
        T = feat_dict[0].shape[1]
        if self.full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])
        else:
            assert self.full_resolution == 1
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3])
        x = self.temporal_weighted_pooling(x)
        if self.keytime_masks:
            key_masks_list = []
            for i in range(3):
                mask_feat = F.relu(self.mask_convs[i][0](x))
                mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
                mask = self.mask_convs[i][2](mask_feat)
                key_masks_list.append(mask)
            key_masks = torch.cat(key_masks_list, dim=1)
        else:
            key_masks = None
        x_flat = x.reshape(x.size(0), -1)
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        if self.temporal_key_points:
            temporal_key_points_list = []
            for i in range(3):
                x = F.relu(self.temporal_key_points_fc[i][0](shared_feat))
                x = F.relu(self.temporal_key_points_fc[i][1](x))
                x = self.temporal_key_points_fc[i][2](x)
                temporal_key_points_list.append(x)
            temporal_key_points = torch.cat(temporal_key_points_list, dim=1)
        else:
            temporal_key_points = None
        if self.height_values:
            height_values_list = []
            for i in range(3):
                x = F.relu(self.height_values_fc[i][0](shared_feat))
                x = F.relu(self.height_values_fc[i][1](x))
                x = self.height_values_fc[i][2](x)
                height_values_list.append(x)
            height_values = torch.cat(height_values_list, dim=1)
        else:
            height_values = None
        if self.height_differences:
            height_differences_list = []
            for i in range(2):
                x = F.relu(self.height_differences_fc[i][0](shared_feat))
                x = F.relu(self.height_differences_fc[i][1](x))
                x = self.height_differences_fc[i][2](x)
                height_differences_list.append(x)
            height_differences = torch.cat(height_differences_list, dim=1)
        else:
            height_differences = None
        if self.normed_height:
            normed_height_list = []
            for i in range(3):
                x = F.relu(self.normed_height_fc[i][0](shared_feat))
                x = F.relu(self.normed_height_fc[i][1](x))
                x = self.normed_height_fc[i][2](x)
                normed_height_list.append(x)
            normed_height = torch.cat(normed_height_list, dim=1)
        else:
            normed_height = None
        if self.normed_height_differences:
            normed_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_height_differences_fc[i][1](x))
                x = self.normed_height_differences_fc[i][2](x)
                normed_height_differences_list.append(x)
            normed_height_differences = torch.cat(normed_height_differences_list, dim=1)
        else:
            normed_height_differences = None
        if self.normed_sqrt_height_differences:
            normed_sqrt_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
                x = self.normed_sqrt_height_differences_fc[i][2](x)
                normed_sqrt_height_differences_list.append(x)
            normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        else:
            normed_sqrt_height_differences = None
        if self.retitution:
            x = F.relu(self.gt_retitusion_fc[0](shared_feat))
            x = F.relu(self.gt_retitusion_fc[1](x))
            gt_retitusion = self.gt_retitusion_fc[2](x)
        else:
            gt_retitusion = None
        return (temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks)

class PDecoderProbev1(nn.Module):

    def __init__(self):
        super(PDecoderProbev1, self).__init__()
        self.final_fc1 = nn.Linear(320 + 320, 128)
        self.final_fc2 = nn.Linear(128, 64)
        self.final_fc3 = nn.Linear(64, 1)
        self.fc = nn.Linear(1, 1)

    def forward(self, x0, mask=None, full_resolution=0):
        x = F.relu(self.final_fc1(x0))
        x = F.relu(self.final_fc2(x))
        x = F.relu(self.final_fc3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PDecoderProbev2(nn.Module):

    def __init__(self, layer_dims=[320, 320, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320], hidden_dim=128):
        super(PDecoderProbev2, self).__init__()
        self.layer_projections = nn.ModuleList([nn.Linear(dim * 2, hidden_dim) for dim in layer_dims])
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim // 2, 1))

    def forward(self, probe_out_list):
        layer_features = []
        for i, (proj, probe_out) in enumerate(zip(self.layer_projections, probe_out_list)):
            features_0 = probe_out[0]
            features_1 = probe_out[1]
            pooled_0 = torch.mean(features_0, dim=1)
            pooled_1 = torch.mean(features_1, dim=1)
            concatenated = torch.cat([pooled_0, pooled_1], dim=1)
            projected = proj(concatenated)
            layer_features.append(projected)
        aggregated = torch.stack(layer_features, dim=1).mean(dim=1)
        output = self.decoder(aggregated)
        return output

class PDecoderProbev3(nn.Module):

    def __init__(self, layer_dims=[320, 320, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320], hidden_dim=128):
        super(PDecoderProbev3, self).__init__()
        self.layer_mlps_0 = nn.ModuleList([nn.Sequential(nn.Linear(seq_len_0, seq_len_0 // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(seq_len_0 // 2, 1)) for seq_len_0 in [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]])
        self.layer_mlps_1 = nn.ModuleList([nn.Sequential(nn.Linear(seq_len_1, seq_len_1 // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(seq_len_1 // 2, 1)) for seq_len_1 in [9216, 9216, 2304, 2304, 576, 576, 144, 576, 576, 576, 2304, 2304, 2304, 9216, 9216, 9216]])
        self.layer_projections = nn.ModuleList([nn.Linear(dim * 2, hidden_dim) for dim in layer_dims])
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim // 2, 1))

    def forward(self, probe_out_list):
        layer_features = []
        for i, (mlp_0, mlp_1, proj, probe_out) in enumerate(zip(self.layer_mlps_0, self.layer_mlps_1, self.layer_projections, probe_out_list)):
            features_0 = probe_out[0]
            features_1 = probe_out[1]
            features_0_t = features_0.transpose(1, 2)
            features_1_t = features_1.transpose(1, 2)
            reduced_0_t = mlp_0(features_0_t)
            reduced_1_t = mlp_1(features_1_t)
            reduced_0 = reduced_0_t.transpose(1, 2).squeeze(1)
            reduced_1 = reduced_1_t.transpose(1, 2).squeeze(1)
            concatenated = torch.cat([reduced_0, reduced_1], dim=1)
            projected = proj(concatenated)
            layer_features.append(projected)
        aggregated = torch.stack(layer_features, dim=1).mean(dim=1)
        output = self.decoder(aggregated)
        return output

class PDecoderFC1(nn.Module):

    def __init__(self):
        super(PDecoderFC1, self).__init__()
        self.fc1 = nn.Linear(512 + 640 + 320 + 1280, 256)
        self.fc2 = nn.Linear(2560 + 640, 512)
        self.final_fc1 = nn.Linear(256 + 320 + 640, 128)
        self.final_fc2 = nn.Linear(128, 64)
        self.final_fc3 = nn.Linear(64, 1)
        self.fc = nn.Linear(1, 1)

    def forward(self, feat_dict, mask, full_resolution=0):
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 1:
            for key_i in feat_dict.keys():
                org_feat_h, org_feat_w = (feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                feat_dict[key_i] = F.interpolate(feat_dict[key_i].reshape(-1, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4]), size=(int(mask.shape[-2] / 2), int(mask.shape[-1] / 2)), mode='bilinear', align_corners=False)
                feat_dict[key_i] = feat_dict[key_i].reshape(mask.shape[0], -1, feat_dict[key_i].shape[-3], feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
                feat_dict[key_i] = F.interpolate(feat_dict[key_i], size=(org_feat_h, org_feat_w), mode='bilinear', align_corners=False)
        elif full_resolution == 2:
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 3:
            for key_i in feat_dict.keys():
                roi_features = extract_roi_features(feat_dict, mask, key_i)
                feat_dict[key_i] = roi_features
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1
        x2 = torch.cat((feat_dict[2].mean(-1).mean(-1), feat_dict[7].mean(-1).mean(-1), feat_dict[3].mean(-1).mean(-1)), dim=1)
        x2 = F.relu(self.fc2(x2))
        x1 = torch.cat((feat_dict[1].mean(-1).mean(-1), feat_dict[5].mean(-1).mean(-1), feat_dict[6].mean(-1).mean(-1), x2), dim=1)
        x1 = F.relu(self.fc1(x1))
        x0 = torch.cat((feat_dict[0].mean(-1).mean(-1), feat_dict[4].mean(-1).mean(-1), x1), dim=1)
        x = F.relu(self.final_fc1(x0))
        x = F.relu(self.final_fc2(x))
        x = F.relu(self.final_fc3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PDecoderFC2(nn.Module):

    def __init__(self):
        super(PDecoderFC2, self).__init__()
        self.fc1 = nn.Linear(512 + (640 + 320 + 1280) * 36 * 64, 256)
        self.fc2 = nn.Linear((2560 + 640) * 18 * 32, 512)
        self.final_fc1 = nn.Linear(256 + (320 + 640) * 72 * 128, 128)
        self.final_fc2 = nn.Linear(128, 64)
        self.final_fc3 = nn.Linear(64, 1)
        self.fc = nn.Linear(1, 1)

    def forward(self, feat_dict, mask, full_resolution=0):
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 1:
            for key_i in feat_dict.keys():
                org_feat_h, org_feat_w = (feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                feat_dict[key_i] = F.interpolate(feat_dict[key_i].reshape(-1, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4]), size=(int(mask.shape[-2] / 2), int(mask.shape[-1] / 2)), mode='bilinear', align_corners=False)
                feat_dict[key_i] = feat_dict[key_i].reshape(mask.shape[0], -1, feat_dict[key_i].shape[-3], feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
                feat_dict[key_i] = F.interpolate(feat_dict[key_i], size=(org_feat_h, org_feat_w), mode='bilinear', align_corners=False)
        elif full_resolution == 2:
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 3:
            for key_i in feat_dict.keys():
                roi_features = extract_roi_features(feat_dict, mask, key_i)
                feat_dict[key_i] = roi_features
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1
        B = feat_dict[2].shape[0]
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2].reshape(B, -1), feat_dict[7].reshape(B, -1), x3_up.reshape(B, -1)), dim=1)
        x2 = F.relu(self.fc2(x2))
        x1 = torch.cat((feat_dict[1].reshape(B, -1), feat_dict[5].reshape(B, -1), feat_dict[6].reshape(B, -1), x2), dim=1)
        x1 = F.relu(self.fc1(x1))
        x0 = torch.cat((feat_dict[0].reshape(B, -1), feat_dict[4].reshape(B, -1), x1), dim=1)
        x = F.relu(self.final_fc1(x0))
        x = F.relu(self.final_fc2(x))
        x = F.relu(self.final_fc3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PDecoderSigmoid(nn.Module):

    def __init__(self):
        super(PDecoderSigmoid, self).__init__()
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(72 * 128, 1)

    def forward(self, feat_dict, mask, full_resolution=0):
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 1:
            for key_i in feat_dict.keys():
                org_feat_h, org_feat_w = (feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                feat_dict[key_i] = F.interpolate(feat_dict[key_i].reshape(-1, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4]), size=(int(mask.shape[-2] / 2), int(mask.shape[-1] / 2)), mode='bilinear', align_corners=False)
                feat_dict[key_i] = feat_dict[key_i].reshape(mask.shape[0], -1, feat_dict[key_i].shape[-3], feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
                feat_dict[key_i] = F.interpolate(feat_dict[key_i], size=(org_feat_h, org_feat_w), mode='bilinear', align_corners=False)
        elif full_resolution == 2:
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 3:
            for key_i in feat_dict.keys():
                roi_features = extract_roi_features(feat_dict, mask, key_i)
                feat_dict[key_i] = roi_features
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

class PDecoderTemporalGRU(nn.Module):

    def __init__(self):
        super(PDecoderTemporalGRU, self).__init__()
        self.gru_0 = nn.GRU(input_size=320, hidden_size=320, batch_first=True)
        self.gru_1 = nn.GRU(input_size=640, hidden_size=640, batch_first=True)
        self.gru_2 = nn.GRU(input_size=1280, hidden_size=1280, batch_first=True)
        self.gru_3 = nn.GRU(input_size=1280, hidden_size=1280, batch_first=True)
        self.gru_4 = nn.GRU(input_size=640, hidden_size=640, batch_first=True)
        self.gru_5 = nn.GRU(input_size=1280, hidden_size=1280, batch_first=True)
        self.gru_6 = nn.GRU(input_size=320, hidden_size=320, batch_first=True)
        self.gru_7 = nn.GRU(input_size=640, hidden_size=640, batch_first=True)
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(72 * 128, 1)

    def _apply_gru_to_feature(self, feature, key, mask_):
        B, T, C, H, W = feature.shape
        masked_feature = feature * mask_.unsqueeze(2)
        reshaped_feature = masked_feature.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        if key == 0:
            _, gru_output = self.gru_0(reshaped_feature)
        elif key == 1:
            _, gru_output = self.gru_1(reshaped_feature)
        elif key == 2:
            _, gru_output = self.gru_2(reshaped_feature)
        elif key == 3:
            _, gru_output = self.gru_3(reshaped_feature)
        elif key == 4:
            _, gru_output = self.gru_4(reshaped_feature)
        elif key == 5:
            _, gru_output = self.gru_5(reshaped_feature)
        elif key == 6:
            _, gru_output = self.gru_6(reshaped_feature)
        elif key == 7:
            _, gru_output = self.gru_7(reshaped_feature)
        last_output = gru_output[-1]
        output_feature = last_output.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return output_feature

    def forward(self, feat_dict, mask, full_resolution=0):
        if full_resolution == 0:
            processed_feat_dict = {}
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                processed_feat_dict[key_i] = self._apply_gru_to_feature(feat_dict[key_i], key_i, mask_)
            feat_dict = processed_feat_dict
        else:
            assert 0 == 1
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PDecoderTemporalConcat(nn.Module):

    def __init__(self):
        super(PDecoderTemporalConcat, self).__init__()
        self.conv1 = nn.Conv2d(512 + (640 + 320 + 1280) * 16, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d((2560 + 640) * 16, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + (320 + 640) * 16, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(72 * 128, 1)

    def forward(self, feat_dict, mask, full_resolution=0):
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].reshape(feat_dict[key_i].shape[0], -1, feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
        else:
            assert 0 == 1
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PDecoderCondMask(nn.Module):

    def __init__(self):
        super(PDecoderCondMask, self).__init__()
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280 + 3, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640 + 3, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640 + 2, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(72 * 128, 1)

    def forward(self, feat_dict, mask, full_resolution=0):
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = torch.cat([feat_dict[key_i], mask_.unsqueeze(2)], dim=2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PDecoderAdv(nn.Module):

    def __init__(self):
        super(PDecoderAdv, self).__init__()
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(72 * 128, 12)

    def forward(self, feat_dict, mask, full_resolution=0):
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                mask_ = F.interpolate(mask[:, 0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PDecoderTemporalConv(nn.Module):

    def __init__(self):
        super(PDecoderTemporalConv, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels=2949120, out_channels=2949120, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(72 * 128, 1)

    def forward(self, feat_dict, mask, full_resolution=0):
        for key_i in feat_dict.keys():
            batch, time, channels, height, width = feat_dict[key_i].shape
            feat_temporal = feat_dict[key_i].permute(0, 2, 3, 4, 1)
            feat_temporal = feat_temporal.reshape(batch, -1, time)
            feat_temporal = self.temporal_conv(feat_temporal)
            feat_temporal = feat_temporal.view(batch, channels, height, width, time)
            feat_dict[key_i] = feat_temporal.permute(0, 4, 1, 2, 3)
            mask_ = F.interpolate(mask[:, 0].float(), size=(height, width), mode='nearest')
            feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
            feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
