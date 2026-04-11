import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops



def extract_roi_features(feat_dict, mask, key_i, spatial_scale=None, output_size=(7, 7)):
    """
    Extract features inside the bbox given by the mask using ROI Align.
    
    Args:
        feat_dict (dict): Dictionary containing features at different levels
        mask (torch.Tensor): Binary mask of shape [batch_size, 1, seq_len, height, width]
        key_i (str): Key for the specific feature level in feat_dict
        spatial_scale (float, optional): Scale factor to map from mask to feature coordinates
        output_size (tuple, optional): Output size of the ROI features
    
    Returns:
        torch.Tensor: ROI aligned features
    """
    batch_size, _, seq_len, mask_h, mask_w = mask.shape
    _, _, feat_c, feat_h, feat_w = feat_dict[key_i].shape

    output_size = (feat_h, feat_w)
    
    # If spatial_scale is not provided, calculate it based on relative sizes
    if spatial_scale is None:
        spatial_scale = feat_h / mask_h
    
    # Rearrange feature tensor to match torchvision's ROIAlign input format
    # from [batch_size, seq_len, channels, height, width] to [batch_size * seq_len, channels, height, width]
    features = feat_dict[key_i].permute(0, 1, 2, 3, 4).reshape(batch_size * seq_len, feat_c, feat_h, feat_w)
    
    # Get bounding boxes from masks
    all_boxes = []
    for b in range(batch_size):
        for s in range(seq_len):
            # Get binary mask for this instance
            m = mask[b, 0, s]
            
            # Skip if mask is empty
            if not torch.any(m):
                # Add a dummy box if mask is empty
                all_boxes.append(torch.tensor([0, 0, 1, 1]))
                continue
            
            # Find bounding box coordinates from mask
            indices = torch.nonzero(m, as_tuple=False)
            if indices.shape[0] == 0:
                all_boxes.append(torch.tensor([0, 0, 1, 1]))
                continue
                
            y_min, y_max = indices[:, 0].min().item(), indices[:, 0].max().item()
            x_min, x_max = indices[:, 1].min().item(), indices[:, 1].max().item()
            
            # Convert to [x1, y1, x2, y2] format and add batch index
            box = torch.tensor([x_min, y_min, x_max + 1, y_max + 1])
            all_boxes.append(box)
    
    # Stack all boxes and add batch indices
    boxes = torch.stack(all_boxes)
    batch_indices = torch.arange(batch_size, device=boxes.device).repeat_interleave(seq_len)
    rois = torch.cat([batch_indices.unsqueeze(1), boxes], dim=1).float().to(feat_dict[key_i].device)
    
    # Apply ROI Align
    # Note: torchvision's ROIAlign expects boxes in absolute coordinates, not normalized
    roi_features = ops.roi_align(
        features,
        rois,
        output_size=output_size,
        spatial_scale=spatial_scale,
        aligned=True
    )
    
    # Reshape back to [batch_size, seq_len, channels, roi_h, roi_w]
    roi_features = roi_features.reshape(batch_size, seq_len, -1, output_size[0], output_size[1])
    
    return roi_features





class PDecoder(nn.Module):
    def __init__(self):
        super(PDecoder, self).__init__()

        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)


        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(72 * 128, 1)

    def forward(self, feat_dict, mask, full_resolution=0):

        # ipdb.set_trace()
        # full_resolution = 1 means interpolate the feature to the original mask size
        # full_resolution = 2 means no mask feature pooling
        # full_resolution = 3 means ROI align to pool the feature

        
        # average pooling for temporal dimension
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 1:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                org_feat_h, org_feat_w = feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]
                feat_dict[key_i] = F.interpolate(feat_dict[key_i].reshape(-1, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4]), size=(int(mask.shape[-2]/2), int(mask.shape[-1]/2)), mode='bilinear', align_corners=False)
                feat_dict[key_i] = feat_dict[key_i].reshape(mask.shape[0], -1, feat_dict[key_i].shape[-3], feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                # feat_dict[key_i] = feat_dict[key_i] * mask.permute(0,2,1,3,4)
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
                feat_dict[key_i] = F.interpolate(feat_dict[key_i], size=(org_feat_h, org_feat_w), mode='bilinear', align_corners=False)
        elif full_resolution == 2:
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 3: # roi align
            for key_i in feat_dict.keys():
                roi_features = extract_roi_features(feat_dict, mask, key_i)
                feat_dict[key_i] = roi_features
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1



        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x
    


# cond the mask on the feat
class PDecoderSupElasticity(nn.Module):
    def __init__(self):
        super(PDecoderSupElasticity, self).__init__()

        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280 + 3, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640 + 3, 512, kernel_size=3, padding=1)

        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640 + 2, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # 3 separate groups of conv layers to predict 3 keytime_masks
        self.mask_convs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.Conv2d(16, 1, kernel_size=3, padding=1)
            ]) for _ in range(3)
        ])
        
        # Shared feature extractor for all other predictions
        self.shared_fc1 = nn.Linear(72 * 128 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        
        # Specific FC layers for each prediction dimension
        # temporal_key_points (3 dimensions with 3 FC layers each)
        self.temporal_key_points_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(3)
        ])
        
        # height_values (3 dimensions with 3 FC layers each)
        self.height_values_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(3)
        ])
        
        # height_differences (2 dimensions with 3 FC layers each)
        self.height_differences_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(2)
        ])
        
        # normed_height (3 dimensions with 3 FC layers each)
        self.normed_height_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(3)
        ])
        
        # normed_height_differences (2 dimensions with 3 FC layers each)
        self.normed_height_differences_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(2)
        ])
        
        # normed_sqrt_height_differences (2 dimensions with 3 FC layers each)
        self.normed_sqrt_height_differences_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(2)
        ])
        
        # gt_retitusion (1 dimension with 3 FC layers)
        self.gt_retitusion_fc = nn.ModuleList([
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        ])

    def forward(self, feat_dict, mask, full_resolution=0):
        # average pooling for temporal dimension
        # if full_resolution == 0:
        for key_i in feat_dict.keys():
            mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
            feat_dict[key_i] = torch.cat([feat_dict[key_i], mask_.unsqueeze(2)], dim=2)
            feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        # else:
        #     assert 0 == 1

        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        
        # Process each mask separately using its own conv layers
        key_masks_list = []
        for i in range(3):
            mask_feat = F.relu(self.mask_convs[i][0](x))
            mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
            # mask = torch.sigmoid(self.mask_convs[i][2](mask_feat))  # Single channel mask
            mask = self.mask_convs[i][2](mask_feat)  # Single channel mask
            key_masks_list.append(mask)
        key_masks = torch.cat(key_masks_list, dim=1)  # Concatenate on channel dimension
        
        # Flatten and process for other predictions using FC layers
        x_flat = x.view(x.size(0), -1)  # Flatten
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        
        # Process each prediction using dedicated 3-layer FC networks per dimension
        
        # Process temporal_key_points (3 dimensions)
        temporal_key_points_list = []
        for i in range(3):
            x = F.relu(self.temporal_key_points_fc[i][0](shared_feat))
            x = F.relu(self.temporal_key_points_fc[i][1](x))
            x = self.temporal_key_points_fc[i][2](x)
            temporal_key_points_list.append(x)
        temporal_key_points = torch.cat(temporal_key_points_list, dim=1)
        
        # Process height_values (3 dimensions)
        height_values_list = []
        for i in range(3):
            x = F.relu(self.height_values_fc[i][0](shared_feat))
            x = F.relu(self.height_values_fc[i][1](x))
            x = self.height_values_fc[i][2](x)
            height_values_list.append(x)
        height_values = torch.cat(height_values_list, dim=1)
        
        # Process height_differences (2 dimensions)
        height_differences_list = []
        for i in range(2):
            x = F.relu(self.height_differences_fc[i][0](shared_feat))
            x = F.relu(self.height_differences_fc[i][1](x))
            x = self.height_differences_fc[i][2](x)
            height_differences_list.append(x)
        height_differences = torch.cat(height_differences_list, dim=1)
        
        # Process normed_height (3 dimensions)
        normed_height_list = []
        for i in range(3):
            x = F.relu(self.normed_height_fc[i][0](shared_feat))
            x = F.relu(self.normed_height_fc[i][1](x))
            x = self.normed_height_fc[i][2](x)
            normed_height_list.append(x)
        normed_height = torch.cat(normed_height_list, dim=1)
        
        # Process normed_height_differences (2 dimensions)
        normed_height_differences_list = []
        for i in range(2):
            x = F.relu(self.normed_height_differences_fc[i][0](shared_feat))
            x = F.relu(self.normed_height_differences_fc[i][1](x))
            x = self.normed_height_differences_fc[i][2](x)
            normed_height_differences_list.append(x)
        normed_height_differences = torch.cat(normed_height_differences_list, dim=1)
        
        # Process normed_sqrt_height_differences (2 dimensions)
        normed_sqrt_height_differences_list = []
        for i in range(2):
            x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
            x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
            x = self.normed_sqrt_height_differences_fc[i][2](x)
            normed_sqrt_height_differences_list.append(x)
        normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        
        # Process gt_retitusion (1 dimension)
        x = F.relu(self.gt_retitusion_fc[0](shared_feat))
        x = F.relu(self.gt_retitusion_fc[1](x))
        gt_retitusion = self.gt_retitusion_fc[2](x)
        
        return temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks




# multiply the mask on the feat
class PDecoderSupElasticityv1(nn.Module):
    def __init__(self):
        super(PDecoderSupElasticityv1, self).__init__()

        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)

        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # 3 separate groups of conv layers to predict 3 keytime_masks
        self.mask_convs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.Conv2d(16, 1, kernel_size=3, padding=1)
            ]) for _ in range(3)
        ])
        
        # Shared feature extractor for all other predictions
        self.shared_fc1 = nn.Linear(72 * 128 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        
        # Specific FC layers for each prediction dimension
        # temporal_key_points (3 dimensions with 3 FC layers each)
        self.temporal_key_points_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(3)
        ])
        
        # height_values (3 dimensions with 3 FC layers each)
        self.height_values_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(3)
        ])
        
        # height_differences (2 dimensions with 3 FC layers each)
        self.height_differences_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(2)
        ])
        
        # normed_height (3 dimensions with 3 FC layers each)
        self.normed_height_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(3)
        ])
        
        # normed_height_differences (2 dimensions with 3 FC layers each)
        self.normed_height_differences_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(2)
        ])
        
        # normed_sqrt_height_differences (2 dimensions with 3 FC layers each)
        self.normed_sqrt_height_differences_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ]) for _ in range(2)
        ])
        
        # gt_retitusion (1 dimension with 3 FC layers)
        self.gt_retitusion_fc = nn.ModuleList([
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        ])

    def forward(self, feat_dict, mask, full_resolution=0):
        # average pooling for temporal dimension
        # if full_resolution == 0:
        for key_i in feat_dict.keys():
            # ipdb.set_trace()
            mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
            feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
            feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        # else:
        #     assert 0 == 1

        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        
        # Process each mask separately using its own conv layers
        key_masks_list = []
        for i in range(3):
            mask_feat = F.relu(self.mask_convs[i][0](x))
            mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
            # mask = torch.sigmoid(self.mask_convs[i][2](mask_feat))  # Single channel mask
            mask = self.mask_convs[i][2](mask_feat)  # Single channel mask
            key_masks_list.append(mask)
        key_masks = torch.cat(key_masks_list, dim=1)  # Concatenate on channel dimension
        
        # Flatten and process for other predictions using FC layers
        x_flat = x.view(x.size(0), -1)  # Flatten
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        
        # Process each prediction using dedicated 3-layer FC networks per dimension
        
        # Process temporal_key_points (3 dimensions)
        temporal_key_points_list = []
        for i in range(3):
            x = F.relu(self.temporal_key_points_fc[i][0](shared_feat))
            x = F.relu(self.temporal_key_points_fc[i][1](x))
            x = self.temporal_key_points_fc[i][2](x)
            temporal_key_points_list.append(x)
        temporal_key_points = torch.cat(temporal_key_points_list, dim=1)
        
        # Process height_values (3 dimensions)
        height_values_list = []
        for i in range(3):
            x = F.relu(self.height_values_fc[i][0](shared_feat))
            x = F.relu(self.height_values_fc[i][1](x))
            x = self.height_values_fc[i][2](x)
            height_values_list.append(x)
        height_values = torch.cat(height_values_list, dim=1)
        
        # Process height_differences (2 dimensions)
        height_differences_list = []
        for i in range(2):
            x = F.relu(self.height_differences_fc[i][0](shared_feat))
            x = F.relu(self.height_differences_fc[i][1](x))
            x = self.height_differences_fc[i][2](x)
            height_differences_list.append(x)
        height_differences = torch.cat(height_differences_list, dim=1)
        
        # Process normed_height (3 dimensions)
        normed_height_list = []
        for i in range(3):
            x = F.relu(self.normed_height_fc[i][0](shared_feat))
            x = F.relu(self.normed_height_fc[i][1](x))
            x = self.normed_height_fc[i][2](x)
            normed_height_list.append(x)
        normed_height = torch.cat(normed_height_list, dim=1)
        
        # Process normed_height_differences (2 dimensions)
        normed_height_differences_list = []
        for i in range(2):
            x = F.relu(self.normed_height_differences_fc[i][0](shared_feat))
            x = F.relu(self.normed_height_differences_fc[i][1](x))
            x = self.normed_height_differences_fc[i][2](x)
            normed_height_differences_list.append(x)
        normed_height_differences = torch.cat(normed_height_differences_list, dim=1)
        
        # Process normed_sqrt_height_differences (2 dimensions)
        normed_sqrt_height_differences_list = []
        for i in range(2):
            x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
            x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
            x = self.normed_sqrt_height_differences_fc[i][2](x)
            normed_sqrt_height_differences_list.append(x)
        normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        
        # Process gt_retitusion (1 dimension)
        x = F.relu(self.gt_retitusion_fc[0](shared_feat))
        x = F.relu(self.gt_retitusion_fc[1](x))
        gt_retitusion = self.gt_retitusion_fc[2](x)
        
        return temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks




# multiply the mask on the feat, while offer options for each parameter
class PDecoderSupElasticityv2(nn.Module):
    def __init__(self,
                temporal_key_points=True,
                height_values=True,
                height_differences=True,
                normed_height=True,
                normed_height_differences=True,
                normed_sqrt_height_differences=True,
                keytime_masks=True,
                full_resolution=0,
                ):
        super(PDecoderSupElasticityv2, self).__init__()

        # ipdb.set_trace()

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

        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        
        if self.keytime_masks:
            # 3 separate groups of conv layers to predict 3 keytime_masks
            self.mask_convs = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.Conv2d(16, 1, kernel_size=3, padding=1)
                ]) for _ in range(3)
            ])
        
        # Shared feature extractor for all other predictions
        self.shared_fc1 = nn.Linear(72 * 128 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        
        # Specific FC layers for each prediction dimension
        if self.temporal_key_points:
            # temporal_key_points (3 dimensions with 3 FC layers each)
            self.temporal_key_points_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        
        if self.height_values:
            # height_values (3 dimensions with 3 FC layers each)
            self.height_values_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        if self.height_differences:
            # height_differences (2 dimensions with 3 FC layers each)
            self.height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        if self.normed_height:
            # normed_height (3 dimensions with 3 FC layers each)
            self.normed_height_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        if self.normed_height_differences:
            # normed_height_differences (2 dimensions with 3 FC layers each)
            self.normed_height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        if self.normed_sqrt_height_differences:
            # normed_sqrt_height_differences (2 dimensions with 3 FC layers each)
            self.normed_sqrt_height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        # gt_retitusion (1 dimension with 3 FC layers)
        self.gt_retitusion_fc = nn.ModuleList([
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        ])

    def forward(self, feat_dict, mask):
        # average pooling for temporal dimension
        if self.full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif self.full_resolution == 1:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].var(dim=1)
        elif self.full_resolution == 2:
            for key_i in feat_dict.keys():
                # if torch.any(torch.isinf(feat_dict[key_i])):
                #     print(key_i)
                # ipdb.set_trace()
                # mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                # feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].var(dim=1)
                # torch.any(torch.isinf(feat_dict[key_i]))
                # if torch.any(torch.isnan(feat_dict[key_i])):
                #     ipdb.set_trace()
                inf_mask = torch.isinf(feat_dict[key_i])
    
                # # Count and log the number of infinite values if they exist
                # inf_count = torch.sum(inf_mask).item()
                # if inf_count > 0:
                #     print(f"Found {inf_count} infinite values in feat_dict[{key_i}], replacing them")
                    
                # Replace inf values with the largest finite value for float32
                # torch.finfo(feat_dict[key_i].dtype).max gives the largest representable value for the tensor's dtype
                feat_dict[key_i] = torch.where(
                    inf_mask, 
                    torch.tensor(torch.finfo(feat_dict[key_i].dtype).max, 
                                device=feat_dict[key_i].device, 
                                dtype=feat_dict[key_i].dtype), 
                    feat_dict[key_i]
                )


        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))

        # x11 = F.relu(self.final_conv1(x0))
        # x12 = F.relu(self.final_conv2(x11))
        
        if self.keytime_masks:
            # Process each mask separately using its own conv layers
            key_masks_list = []
            for i in range(3):
                mask_feat = F.relu(self.mask_convs[i][0](x))
                mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
                # mask = torch.sigmoid(self.mask_convs[i][2](mask_feat))  # Single channel mask
                mask = self.mask_convs[i][2](mask_feat)  # Single channel mask
                key_masks_list.append(mask)
            key_masks = torch.cat(key_masks_list, dim=1)  # Concatenate on channel dimension
        else:
            key_masks = None
        
        # Flatten and process for other predictions using FC layers
        # ipdb.set_trace()
        x_flat = x.reshape(x.size(0), -1)  # Flatten using reshape instead of view
        # ipdb.set_trace()
        shared_feat = F.relu(self.shared_fc1(x_flat))
        # ipdb.set_trace()
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        # ipdb.set_trace()
        
        # Process each prediction using dedicated 3-layer FC networks per dimension
        
        if self.temporal_key_points:
            # Process temporal_key_points (3 dimensions)
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
            # Process height_values (3 dimensions)
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
            # Process height_differences (2 dimensions)
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
            # Process normed_height (3 dimensions)
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
            # Process normed_height_differences (2 dimensions)
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
            # Process normed_sqrt_height_differences (2 dimensions)
            normed_sqrt_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
                x = self.normed_sqrt_height_differences_fc[i][2](x)
                normed_sqrt_height_differences_list.append(x)
            normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        else:
            normed_sqrt_height_differences = None

        # ipdb.set_trace()
        
        # Process gt_retitusion (1 dimension)
        x = F.relu(self.gt_retitusion_fc[0](shared_feat))
        # ipdb.set_trace()
        x = F.relu(self.gt_retitusion_fc[1](x))
        # ipdb.set_trace()
        gt_retitusion = self.gt_retitusion_fc[2](x)
        # ipdb.set_trace()

        # print(gt_retitusion)
        
        return temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks





# multiply the mask on the feat, while offer options for each parameter
class PDecoderSupElasticityv2512(nn.Module):
    def __init__(self,
                temporal_key_points=True,
                height_values=True,
                height_differences=True,
                normed_height=True,
                normed_height_differences=True,
                normed_sqrt_height_differences=True,
                keytime_masks=True,
                full_resolution=0,
                ):
        super(PDecoderSupElasticityv2512, self).__init__()

        # ipdb.set_trace()

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

        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        
        if self.keytime_masks:
            # 3 separate groups of conv layers to predict 3 keytime_masks
            self.mask_convs = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.Conv2d(16, 1, kernel_size=3, padding=1)
                ]) for _ in range(3)
            ])
        
        # Shared feature extractor for all other predictions
        self.shared_fc1 = nn.Linear(40 * 64 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        
        # Specific FC layers for each prediction dimension
        if self.temporal_key_points:
            # temporal_key_points (3 dimensions with 3 FC layers each)
            self.temporal_key_points_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        
        if self.height_values:
            # height_values (3 dimensions with 3 FC layers each)
            self.height_values_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        if self.height_differences:
            # height_differences (2 dimensions with 3 FC layers each)
            self.height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        if self.normed_height:
            # normed_height (3 dimensions with 3 FC layers each)
            self.normed_height_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        if self.normed_height_differences:
            # normed_height_differences (2 dimensions with 3 FC layers each)
            self.normed_height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        if self.normed_sqrt_height_differences:
            # normed_sqrt_height_differences (2 dimensions with 3 FC layers each)
            self.normed_sqrt_height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        # gt_retitusion (1 dimension with 3 FC layers)
        self.gt_retitusion_fc = nn.ModuleList([
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        ])

    def forward(self, feat_dict, mask):

        # for key_i in feat_dict.keys():
        #     print(feat_dict[key_i].shape)

        # ipdb.set_trace()
        # average pooling for temporal dimension
        if self.full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif self.full_resolution == 1:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].var(dim=1)
        elif self.full_resolution == 2:
            for key_i in feat_dict.keys():
                # if torch.any(torch.isinf(feat_dict[key_i])):
                #     print(key_i)
                # ipdb.set_trace()
                # mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                # feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].var(dim=1)
                # torch.any(torch.isinf(feat_dict[key_i]))
                # if torch.any(torch.isnan(feat_dict[key_i])):
                #     ipdb.set_trace()
                inf_mask = torch.isinf(feat_dict[key_i])
    
                # # Count and log the number of infinite values if they exist
                # inf_count = torch.sum(inf_mask).item()
                # if inf_count > 0:
                #     print(f"Found {inf_count} infinite values in feat_dict[{key_i}], replacing them")
                    
                # Replace inf values with the largest finite value for float32
                # torch.finfo(feat_dict[key_i].dtype).max gives the largest representable value for the tensor's dtype
                feat_dict[key_i] = torch.where(
                    inf_mask, 
                    torch.tensor(torch.finfo(feat_dict[key_i].dtype).max, 
                                device=feat_dict[key_i].device, 
                                dtype=feat_dict[key_i].dtype), 
                    feat_dict[key_i]
                )


        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(10, 16), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(20, 32), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(40, 64), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))

        # x11 = F.relu(self.final_conv1(x0))
        # x12 = F.relu(self.final_conv2(x11))
        
        if self.keytime_masks:
            # Process each mask separately using its own conv layers
            key_masks_list = []
            for i in range(3):
                mask_feat = F.relu(self.mask_convs[i][0](x))
                mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
                # mask = torch.sigmoid(self.mask_convs[i][2](mask_feat))  # Single channel mask
                mask = self.mask_convs[i][2](mask_feat)  # Single channel mask
                key_masks_list.append(mask)
            key_masks = torch.cat(key_masks_list, dim=1)  # Concatenate on channel dimension
        else:
            key_masks = None
        
        # Flatten and process for other predictions using FC layers
        # ipdb.set_trace()
        x_flat = x.reshape(x.size(0), -1)  # Flatten using reshape instead of view
        # ipdb.set_trace()
        shared_feat = F.relu(self.shared_fc1(x_flat))
        # ipdb.set_trace()
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        # ipdb.set_trace()
        
        # Process each prediction using dedicated 3-layer FC networks per dimension
        
        if self.temporal_key_points:
            # Process temporal_key_points (3 dimensions)
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
            # Process height_values (3 dimensions)
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
            # Process height_differences (2 dimensions)
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
            # Process normed_height (3 dimensions)
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
            # Process normed_height_differences (2 dimensions)
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
            # Process normed_sqrt_height_differences (2 dimensions)
            normed_sqrt_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
                x = self.normed_sqrt_height_differences_fc[i][2](x)
                normed_sqrt_height_differences_list.append(x)
            normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        else:
            normed_sqrt_height_differences = None

        # ipdb.set_trace()
        
        # Process gt_retitusion (1 dimension)
        x = F.relu(self.gt_retitusion_fc[0](shared_feat))
        # ipdb.set_trace()
        x = F.relu(self.gt_retitusion_fc[1](x))
        # ipdb.set_trace()
        gt_retitusion = self.gt_retitusion_fc[2](x)
        # ipdb.set_trace()

        # print(gt_retitusion)
        
        return temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks






# multiply the mask on the feat, while use a temporal query to pool temporal information instead of U-Net for spatial features
class PDecoderSupElasticityv3(nn.Module):
    def __init__(self,
                temporal_key_points=True,
                height_values=True,
                height_differences=True,
                normed_height=True,
                normed_height_differences=True,
                normed_sqrt_height_differences=True,
                keytime_masks=True,
                full_resolution=0,
                retitution=True,
                ):
        super(PDecoderSupElasticityv3, self).__init__()

        # ipdb.set_trace()

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

        # self.unet_block_weights = nn.Parameter(torch.ones((8)) / 8)
        # self.temporal_attn = nn.Parameter(torch.ones((64, 72, 128)))
        
        # Initialize temporal attention with Kaiming/He initialization
        # This provides better gradient flow and prevents vanishing/exploding gradients
        self.temporal_attn = nn.Parameter(
            torch.randn((64, 72, 128)) * 0.02  # Small standard deviation for more stable initial training
        )
        
        # Alternative initialization approach:
        # fan_in = 64 * 72 * 128
        # std = math.sqrt(2.0 / fan_in)
        # self.temporal_attn = nn.Parameter(torch.randn((64, 72, 128)) * std)

        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        
        if self.keytime_masks:
            # 3 separate groups of conv layers to predict 3 keytime_masks
            self.mask_convs = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.Conv2d(16, 1, kernel_size=3, padding=1)
                ]) for _ in range(3)
            ])
        
        # Shared feature extractor for all other predictions
        self.shared_fc1 = nn.Linear(72 * 128 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        
        # Specific FC layers for each prediction dimension
        if self.temporal_key_points:
            # temporal_key_points (3 dimensions with 3 FC layers each)
            self.temporal_key_points_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        
        if self.height_values:
            # height_values (3 dimensions with 3 FC layers each)
            self.height_values_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        if self.height_differences:
            # height_differences (2 dimensions with 3 FC layers each)
            self.height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        if self.normed_height:
            # normed_height (3 dimensions with 3 FC layers each)
            self.normed_height_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        if self.normed_height_differences:
            # normed_height_differences (2 dimensions with 3 FC layers each)
            self.normed_height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        if self.normed_sqrt_height_differences:
            # normed_sqrt_height_differences (2 dimensions with 3 FC layers each)
            self.normed_sqrt_height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
       
        if self.retitution:
            # gt_retitusion (1 dimension with 3 FC layers)
            self.gt_retitusion_fc = nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ])

    def temporal_weighted_pooling(self, x):
        """
        Apply weighted temporal pooling using cosine similarity with learned temporal attention.
        Process each spatial position (H,W) separately using vectorized operations.
        
        Args:
            x: Input tensor of shape [B, T, C, H, W]
            
        Returns:
            Temporally pooled tensor of shape [B, C, H, W]
        """
        B, T, C, H, W = x.shape
        
        # Reshape input to [B, T, C, H*W]
        x_flat = x.reshape(B, T, C, H*W)
        
        # Permute to [B, T, H*W, C] to process each spatial position in parallel
        x_perm = x_flat.permute(0, 1, 3, 2)
        
        # Reshape attention pattern to [C, H*W]
        attn_flat = self.temporal_attn.reshape(C, H*W)
        # Transpose to [H*W, C] to align with spatial dimension
        attn_perm = attn_flat.t()
        
        # For each spatial position, normalize feature vectors across channels
        # x_perm is [B, T, H*W, C], normalize across dim=-1 (channel dim)
        x_norm = F.normalize(x_perm, p=2, dim=-1)
        
        # Normalize attention vectors, shape: [H*W, C]
        attn_norm = F.normalize(attn_perm, p=2, dim=-1)
        
        # Compute cosine similarity for each spatial position: [B, T, H*W]
        # We need to carefully align dimensions for batch matrix multiplication
        # Reshaping attention to [1, 1, H*W, C] for broadcasting
        attn_norm_expanded = attn_norm.unsqueeze(0).unsqueeze(0)
        
        # Compute cosine similarity using batched matrix multiplication
        # [B, T, H*W, C] × [1, 1, H*W, C] -> [B, T, H*W]
        similarity = torch.sum(x_norm * attn_norm_expanded, dim=-1)
        
        # Apply softmax along temporal dimension to get weights: [B, T, H*W]
        weights = F.softmax(similarity, dim=1)
        
        # Reshape weights to [B, T, H*W, 1] for multiplication
        weights_expanded = weights.unsqueeze(-1)
        
        # Apply weights to original features
        # [B, T, H*W, 1] * [B, T, H*W, C] -> [B, H*W, C]
        weighted_sum = torch.sum(weights_expanded * x_perm, dim=1)
        
        # Reshape back to [B, C, H, W]
        output = weighted_sum.permute(0, 2, 1).reshape(B, C, H, W)
        
        return output
    
    def forward(self, feat_dict, mask):

        B = feat_dict[0].shape[0]
        T = feat_dict[0].shape[1]
        # average pooling for temporal dimension
        if self.full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                # ipdb.set_trace()
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])
                # feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert self.full_resolution == 1
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])

       # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1) 

        # x1_up = F.interpolate(feat_dict[1], size=(72, 128), mode='bilinear', align_corners=False)
        # x2_up = F.interpolate(feat_dict[2], size=(72, 128), mode='bilinear', align_corners=False)
        # x3_up = F.interpolate(feat_dict[3], size=(72, 128), mode='bilinear', align_corners=False)

        # x5_up = F.interpolate(feat_dict[5], size=(72, 128), mode='bilinear', align_corners=False)
        # x6_up = F.interpolate(feat_dict[6], size=(72, 128), mode='bilinear', align_corners=False)
        # x7_up = F.interpolate(feat_dict[7], size=(72, 128), mode='bilinear', align_corners=False)

        # # x0 = torch.cat((feat_dict[0], x1_up, x2_up, x3_up, feat_dict[4], x5_up, x6_up, x7_up), dim=1)
        # x0 = self.unet_block_weights[0] * feat_dict[0] + self.unet_block_weights[1] * x1_up + self.unet_block_weights[2] * x2_up + self.unet_block_weights[3] * x3_up + self.unet_block_weights[4] * feat_dict[4] + self.unet_block_weights[5] * x5_up + self.unet_block_weights[6] * x6_up + self.unet_block_weights[7] * x7_up

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x)) # BT x 64 x 72 x 128
        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3]) # B x T x 64 x 72 x 128
        # for the T dimension, do a weighted average to reduce to 1
        x = self.temporal_weighted_pooling(x) # B x 64 x 72 x 128

        
        if self.keytime_masks:
            # Process each mask separately using its own conv layers
            key_masks_list = []
            for i in range(3):
                mask_feat = F.relu(self.mask_convs[i][0](x))
                mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
                # mask = torch.sigmoid(self.mask_convs[i][2](mask_feat))  # Single channel mask
                mask = self.mask_convs[i][2](mask_feat)  # Single channel mask
                key_masks_list.append(mask)
            key_masks = torch.cat(key_masks_list, dim=1)  # Concatenate on channel dimension
        else:
            key_masks = None
        
        # Flatten and process for other predictions using FC layers
        x_flat = x.reshape(x.size(0), -1)  # Flatten using reshape instead of view
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        
        # Process each prediction using dedicated 3-layer FC networks per dimension
        
        if self.temporal_key_points:
            # Process temporal_key_points (3 dimensions)
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
            # Process height_values (3 dimensions)
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
            # Process height_differences (2 dimensions)
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
            # Process normed_height (3 dimensions)
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
            # Process normed_height_differences (2 dimensions)
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
            # Process normed_sqrt_height_differences (2 dimensions)
            normed_sqrt_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
                x = self.normed_sqrt_height_differences_fc[i][2](x)
                normed_sqrt_height_differences_list.append(x)
            normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        else:
            normed_sqrt_height_differences = None

        # ipdb.set_trace()
        
        if self.retitution:
            # Process gt_retitusion (1 dimension)
            x = F.relu(self.gt_retitusion_fc[0](shared_feat))
            x = F.relu(self.gt_retitusion_fc[1](x))
            gt_retitusion = self.gt_retitusion_fc[2](x)
        else:
            gt_retitusion = None
        
        return temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks




# multiply the mask on the feat, while use a temporal query to pool temporal information instead of U-Net for spatial features
class PDecoderSupElasticityv3512(nn.Module):
    def __init__(self,
                temporal_key_points=True,
                height_values=True,
                height_differences=True,
                normed_height=True,
                normed_height_differences=True,
                normed_sqrt_height_differences=True,
                keytime_masks=True,
                full_resolution=0,
                ):
        super(PDecoderSupElasticityv3512, self).__init__()

        # ipdb.set_trace()

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

        # self.unet_block_weights = nn.Parameter(torch.ones((8)) / 8)
        # self.temporal_attn = nn.Parameter(torch.ones((64, 72, 128)))
        
        # Initialize temporal attention with Kaiming/He initialization
        # This provides better gradient flow and prevents vanishing/exploding gradients
        self.temporal_attn = nn.Parameter(
            torch.randn((64, 40, 64)) * 0.02  # Small standard deviation for more stable initial training
        )
        
        # Alternative initialization approach:
        # fan_in = 64 * 72 * 128
        # std = math.sqrt(2.0 / fan_in)
        # self.temporal_attn = nn.Parameter(torch.randn((64, 72, 128)) * std)

        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        
        if self.keytime_masks:
            # 3 separate groups of conv layers to predict 3 keytime_masks
            self.mask_convs = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.Conv2d(16, 1, kernel_size=3, padding=1)
                ]) for _ in range(3)
            ])
        
        # Shared feature extractor for all other predictions
        self.shared_fc1 = nn.Linear(40 * 64 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        
        # Specific FC layers for each prediction dimension
        if self.temporal_key_points:
            # temporal_key_points (3 dimensions with 3 FC layers each)
            self.temporal_key_points_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        
        if self.height_values:
            # height_values (3 dimensions with 3 FC layers each)
            self.height_values_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        if self.height_differences:
            # height_differences (2 dimensions with 3 FC layers each)
            self.height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        if self.normed_height:
            # normed_height (3 dimensions with 3 FC layers each)
            self.normed_height_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        if self.normed_height_differences:
            # normed_height_differences (2 dimensions with 3 FC layers each)
            self.normed_height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        if self.normed_sqrt_height_differences:
            # normed_sqrt_height_differences (2 dimensions with 3 FC layers each)
            self.normed_sqrt_height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        # gt_retitusion (1 dimension with 3 FC layers)
        self.gt_retitusion_fc = nn.ModuleList([
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        ])

    def temporal_weighted_pooling(self, x):
        """
        Apply weighted temporal pooling using cosine similarity with learned temporal attention.
        Process each spatial position (H,W) separately using vectorized operations.
        
        Args:
            x: Input tensor of shape [B, T, C, H, W]
            
        Returns:
            Temporally pooled tensor of shape [B, C, H, W]
        """
        B, T, C, H, W = x.shape
        
        # Reshape input to [B, T, C, H*W]
        x_flat = x.reshape(B, T, C, H*W)
        
        # Permute to [B, T, H*W, C] to process each spatial position in parallel
        x_perm = x_flat.permute(0, 1, 3, 2)
        
        # Reshape attention pattern to [C, H*W]
        attn_flat = self.temporal_attn.reshape(C, H*W)
        # Transpose to [H*W, C] to align with spatial dimension
        attn_perm = attn_flat.t()
        
        # For each spatial position, normalize feature vectors across channels
        # x_perm is [B, T, H*W, C], normalize across dim=-1 (channel dim)
        x_norm = F.normalize(x_perm, p=2, dim=-1)
        
        # Normalize attention vectors, shape: [H*W, C]
        attn_norm = F.normalize(attn_perm, p=2, dim=-1)
        
        # Compute cosine similarity for each spatial position: [B, T, H*W]
        # We need to carefully align dimensions for batch matrix multiplication
        # Reshaping attention to [1, 1, H*W, C] for broadcasting
        attn_norm_expanded = attn_norm.unsqueeze(0).unsqueeze(0)
        
        # Compute cosine similarity using batched matrix multiplication
        # [B, T, H*W, C] × [1, 1, H*W, C] -> [B, T, H*W]
        similarity = torch.sum(x_norm * attn_norm_expanded, dim=-1)
        
        # Apply softmax along temporal dimension to get weights: [B, T, H*W]
        weights = F.softmax(similarity, dim=1)
        
        # Reshape weights to [B, T, H*W, 1] for multiplication
        weights_expanded = weights.unsqueeze(-1)
        
        # Apply weights to original features
        # [B, T, H*W, 1] * [B, T, H*W, C] -> [B, H*W, C]
        weighted_sum = torch.sum(weights_expanded * x_perm, dim=1)
        
        # Reshape back to [B, C, H, W]
        output = weighted_sum.permute(0, 2, 1).reshape(B, C, H, W)
        
        return output
    
    def forward(self, feat_dict, mask):

        B = feat_dict[0].shape[0]
        T = feat_dict[0].shape[1]
        # average pooling for temporal dimension
        if self.full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                # ipdb.set_trace()
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])
                # feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert self.full_resolution == 1
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])

       # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(10, 16), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(20, 32), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(40, 64), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1) 

        # x1_up = F.interpolate(feat_dict[1], size=(72, 128), mode='bilinear', align_corners=False)
        # x2_up = F.interpolate(feat_dict[2], size=(72, 128), mode='bilinear', align_corners=False)
        # x3_up = F.interpolate(feat_dict[3], size=(72, 128), mode='bilinear', align_corners=False)

        # x5_up = F.interpolate(feat_dict[5], size=(72, 128), mode='bilinear', align_corners=False)
        # x6_up = F.interpolate(feat_dict[6], size=(72, 128), mode='bilinear', align_corners=False)
        # x7_up = F.interpolate(feat_dict[7], size=(72, 128), mode='bilinear', align_corners=False)

        # # x0 = torch.cat((feat_dict[0], x1_up, x2_up, x3_up, feat_dict[4], x5_up, x6_up, x7_up), dim=1)
        # x0 = self.unet_block_weights[0] * feat_dict[0] + self.unet_block_weights[1] * x1_up + self.unet_block_weights[2] * x2_up + self.unet_block_weights[3] * x3_up + self.unet_block_weights[4] * feat_dict[4] + self.unet_block_weights[5] * x5_up + self.unet_block_weights[6] * x6_up + self.unet_block_weights[7] * x7_up

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x)) # BT x 64 x 72 x 128
        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3]) # B x T x 64 x 72 x 128
        # for the T dimension, do a weighted average to reduce to 1
        x = self.temporal_weighted_pooling(x) # B x 64 x 72 x 128

        
        if self.keytime_masks:
            # Process each mask separately using its own conv layers
            key_masks_list = []
            for i in range(3):
                mask_feat = F.relu(self.mask_convs[i][0](x))
                mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
                # mask = torch.sigmoid(self.mask_convs[i][2](mask_feat))  # Single channel mask
                mask = self.mask_convs[i][2](mask_feat)  # Single channel mask
                key_masks_list.append(mask)
            key_masks = torch.cat(key_masks_list, dim=1)  # Concatenate on channel dimension
        else:
            key_masks = None
        
        # Flatten and process for other predictions using FC layers
        x_flat = x.reshape(x.size(0), -1)  # Flatten using reshape instead of view
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        
        # Process each prediction using dedicated 3-layer FC networks per dimension
        
        if self.temporal_key_points:
            # Process temporal_key_points (3 dimensions)
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
            # Process height_values (3 dimensions)
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
            # Process height_differences (2 dimensions)
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
            # Process normed_height (3 dimensions)
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
            # Process normed_height_differences (2 dimensions)
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
            # Process normed_sqrt_height_differences (2 dimensions)
            normed_sqrt_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
                x = self.normed_sqrt_height_differences_fc[i][2](x)
                normed_sqrt_height_differences_list.append(x)
            normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        else:
            normed_sqrt_height_differences = None

        # ipdb.set_trace()
        
        # Process gt_retitusion (1 dimension)
        x = F.relu(self.gt_retitusion_fc[0](shared_feat))
        x = F.relu(self.gt_retitusion_fc[1](x))
        gt_retitusion = self.gt_retitusion_fc[2](x)
        
        return temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks







class PDecoderSupElasticityv4(nn.Module):
    def __init__(self,
                temporal_key_points=True,
                height_values=True,
                height_differences=True,
                normed_height=True,
                normed_height_differences=True,
                normed_sqrt_height_differences=True,
                keytime_masks=True,
                full_resolution=0,
                retitution=True,
                ):
        super(PDecoderSupElasticityv4, self).__init__()

        # ipdb.set_trace()

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

        # self.unet_block_weights = nn.Parameter(torch.ones((8)) / 8)
        # self.temporal_attn = nn.Parameter(torch.ones((64, 72, 128)))
        
        # Initialize temporal attention with Kaiming/He initialization
        # This provides better gradient flow and prevents vanishing/exploding gradients
        self.temporal_attn = nn.Parameter(
            torch.randn((64, 1, 1)) * 0.02  # Small standard deviation for more stable initial training
        )

        # self.temporal_attn = self.temporal_attn_init.expand(64, 72, 128)
        
        # Alternative initialization approach:
        # fan_in = 64 * 72 * 128
        # std = math.sqrt(2.0 / fan_in)
        # self.temporal_attn = nn.Parameter(torch.randn((64, 72, 128)) * std)

        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        
        if self.keytime_masks:
            # 3 separate groups of conv layers to predict 3 keytime_masks
            self.mask_convs = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.Conv2d(16, 1, kernel_size=3, padding=1)
                ]) for _ in range(3)
            ])
        
        # Shared feature extractor for all other predictions
        self.shared_fc1 = nn.Linear(72 * 128 * 64, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        
        # Specific FC layers for each prediction dimension
        if self.temporal_key_points:
            # temporal_key_points (3 dimensions with 3 FC layers each)
            self.temporal_key_points_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        
        if self.height_values:
            # height_values (3 dimensions with 3 FC layers each)
            self.height_values_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        if self.height_differences:
            # height_differences (2 dimensions with 3 FC layers each)
            self.height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        if self.normed_height:
            # normed_height (3 dimensions with 3 FC layers each)
            self.normed_height_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(3)
            ])
        
        if self.normed_height_differences:
            # normed_height_differences (2 dimensions with 3 FC layers each)
            self.normed_height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
        if self.normed_sqrt_height_differences:
            # normed_sqrt_height_differences (2 dimensions with 3 FC layers each)
            self.normed_sqrt_height_differences_fc = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1)
                ]) for _ in range(2)
            ])
        
       
        if self.retitution:
            # gt_retitusion (1 dimension with 3 FC layers)
            self.gt_retitusion_fc = nn.ModuleList([
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1)
            ])

    def temporal_weighted_pooling(self, x):
        """
        Apply weighted temporal pooling using cosine similarity with learned temporal attention.
        Process each spatial position (H,W) separately using vectorized operations.
        
        Args:
            x: Input tensor of shape [B, T, C, H, W]
            
        Returns:
            Temporally pooled tensor of shape [B, C, H, W]
        """
        B, T, C, H, W = x.shape
        
        # Reshape input to [B, T, C, H*W]
        x_flat = x.reshape(B, T, C, H*W)
        
        # Permute to [B, T, H*W, C] to process each spatial position in parallel
        x_perm = x_flat.permute(0, 1, 3, 2)
        
        # Reshape attention pattern to [C, H*W]
        attn_flat = self.temporal_attn.expand(C, H, W).reshape(C, H*W)
        # Transpose to [H*W, C] to align with spatial dimension
        attn_perm = attn_flat.t()
        
        # For each spatial position, normalize feature vectors across channels
        # x_perm is [B, T, H*W, C], normalize across dim=-1 (channel dim)
        x_norm = F.normalize(x_perm, p=2, dim=-1)
        
        # Normalize attention vectors, shape: [H*W, C]
        attn_norm = F.normalize(attn_perm, p=2, dim=-1)
        
        # Compute cosine similarity for each spatial position: [B, T, H*W]
        # We need to carefully align dimensions for batch matrix multiplication
        # Reshaping attention to [1, 1, H*W, C] for broadcasting
        attn_norm_expanded = attn_norm.unsqueeze(0).unsqueeze(0)
        
        # Compute cosine similarity using batched matrix multiplication
        # [B, T, H*W, C] × [1, 1, H*W, C] -> [B, T, H*W]
        similarity = torch.sum(x_norm * attn_norm_expanded, dim=-1)
        
        # Apply softmax along temporal dimension to get weights: [B, T, H*W]
        weights = F.softmax(similarity, dim=1)
        
        # Reshape weights to [B, T, H*W, 1] for multiplication
        weights_expanded = weights.unsqueeze(-1)
        
        # Apply weights to original features
        # [B, T, H*W, 1] * [B, T, H*W, C] -> [B, H*W, C]
        weighted_sum = torch.sum(weights_expanded * x_perm, dim=1)
        
        # Reshape back to [B, C, H, W]
        output = weighted_sum.permute(0, 2, 1).reshape(B, C, H, W)
        
        return output
    
    def forward(self, feat_dict, mask):

        B = feat_dict[0].shape[0]
        T = feat_dict[0].shape[1]
        # average pooling for temporal dimension
        if self.full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                # ipdb.set_trace()
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])
                # feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert self.full_resolution == 1
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].reshape(B * T, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4])

       # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1) 

        # x1_up = F.interpolate(feat_dict[1], size=(72, 128), mode='bilinear', align_corners=False)
        # x2_up = F.interpolate(feat_dict[2], size=(72, 128), mode='bilinear', align_corners=False)
        # x3_up = F.interpolate(feat_dict[3], size=(72, 128), mode='bilinear', align_corners=False)

        # x5_up = F.interpolate(feat_dict[5], size=(72, 128), mode='bilinear', align_corners=False)
        # x6_up = F.interpolate(feat_dict[6], size=(72, 128), mode='bilinear', align_corners=False)
        # x7_up = F.interpolate(feat_dict[7], size=(72, 128), mode='bilinear', align_corners=False)

        # # x0 = torch.cat((feat_dict[0], x1_up, x2_up, x3_up, feat_dict[4], x5_up, x6_up, x7_up), dim=1)
        # x0 = self.unet_block_weights[0] * feat_dict[0] + self.unet_block_weights[1] * x1_up + self.unet_block_weights[2] * x2_up + self.unet_block_weights[3] * x3_up + self.unet_block_weights[4] * feat_dict[4] + self.unet_block_weights[5] * x5_up + self.unet_block_weights[6] * x6_up + self.unet_block_weights[7] * x7_up

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x)) # BT x 64 x 72 x 128
        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3]) # B x T x 64 x 72 x 128
        # for the T dimension, do a weighted average to reduce to 1
        x = self.temporal_weighted_pooling(x) # B x 64 x 72 x 128

        
        if self.keytime_masks:
            # Process each mask separately using its own conv layers
            key_masks_list = []
            for i in range(3):
                mask_feat = F.relu(self.mask_convs[i][0](x))
                mask_feat = F.relu(self.mask_convs[i][1](mask_feat))
                # mask = torch.sigmoid(self.mask_convs[i][2](mask_feat))  # Single channel mask
                mask = self.mask_convs[i][2](mask_feat)  # Single channel mask
                key_masks_list.append(mask)
            key_masks = torch.cat(key_masks_list, dim=1)  # Concatenate on channel dimension
        else:
            key_masks = None
        
        # Flatten and process for other predictions using FC layers
        x_flat = x.reshape(x.size(0), -1)  # Flatten using reshape instead of view
        shared_feat = F.relu(self.shared_fc1(x_flat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))
        
        # Process each prediction using dedicated 3-layer FC networks per dimension
        
        if self.temporal_key_points:
            # Process temporal_key_points (3 dimensions)
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
            # Process height_values (3 dimensions)
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
            # Process height_differences (2 dimensions)
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
            # Process normed_height (3 dimensions)
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
            # Process normed_height_differences (2 dimensions)
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
            # Process normed_sqrt_height_differences (2 dimensions)
            normed_sqrt_height_differences_list = []
            for i in range(2):
                x = F.relu(self.normed_sqrt_height_differences_fc[i][0](shared_feat))
                x = F.relu(self.normed_sqrt_height_differences_fc[i][1](x))
                x = self.normed_sqrt_height_differences_fc[i][2](x)
                normed_sqrt_height_differences_list.append(x)
            normed_sqrt_height_differences = torch.cat(normed_sqrt_height_differences_list, dim=1)
        else:
            normed_sqrt_height_differences = None

        # ipdb.set_trace()
        
        if self.retitution:
            # Process gt_retitusion (1 dimension)
            x = F.relu(self.gt_retitusion_fc[0](shared_feat))
            x = F.relu(self.gt_retitusion_fc[1](x))
            gt_retitusion = self.gt_retitusion_fc[2](x)
        else:
            gt_retitusion = None
        
        return temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks




class PDecoderProbev1(nn.Module):
    def __init__(self):
        super(PDecoderProbev1, self).__init__()

        # self.fc1 = nn.Linear(512 + 640 + 320 + 1280, 256)
        # self.fc2 = nn.Linear(2560 + 640, 512)

        # Final convolutional layers
        self.final_fc1 = nn.Linear(320 + 320, 128)
        self.final_fc2 = nn.Linear(128, 64)
        self.final_fc3 = nn.Linear(64, 1)

        # Fully connected layer
        self.fc = nn.Linear(1, 1)

    def forward(self, x0, mask=None, full_resolution=0):

        # Further processing
        x = F.relu(self.final_fc1(x0))
        x = F.relu(self.final_fc2(x))
        x = F.relu(self.final_fc3(x))

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x


# based on v1, further cope with features from different diffusion layers
class PDecoderProbev2(nn.Module):
    def __init__(self, layer_dims=[320, 320, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320], hidden_dim=128):
        super(PDecoderProbev2, self).__init__()
        
        # Project each layer to a common dimension (for both tensors in each layer)
        self.layer_projections = nn.ModuleList([
            nn.Linear(dim * 2, hidden_dim) for dim in layer_dims  # *2 because we concatenate both tensors
        ])
        
        # Final decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, probe_out_list):
        layer_features = []
        
        for i, (proj, probe_out) in enumerate(zip(self.layer_projections, probe_out_list)):
            # Use both tensors from each layer
            features_0 = probe_out[0]  # Shape: [B, seq_len_0, feature_dim]
            features_1 = probe_out[1]  # Shape: [B, seq_len_1, feature_dim]
            
            # Global average pooling for both tensors
            pooled_0 = torch.mean(features_0, dim=1)  # Shape: [B, feature_dim]
            pooled_1 = torch.mean(features_1, dim=1)  # Shape: [B, feature_dim]
            
            # Concatenate the two pooled features
            concatenated = torch.cat([pooled_0, pooled_1], dim=1)  # Shape: [B, feature_dim * 2]
            
            # Project to common dimension
            projected = proj(concatenated)  # Shape: [B, hidden_dim]
            layer_features.append(projected)
        
        # Aggregate features across layers (average)
        aggregated = torch.stack(layer_features, dim=1).mean(dim=1)  # Shape: [B, hidden_dim]
        
        # Pass through decoder
        output = self.decoder(aggregated)  # Shape: [B, 1]
        
        return output


# based on v2, further cope with T and HW dimensions with an MLP rather than avg pool
class PDecoderProbev3(nn.Module):
    def __init__(self, layer_dims=[320, 320, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320], hidden_dim=128):
        super(PDecoderProbev3, self).__init__()
        
        # Create MLPs for each layer to reduce sequence length dimension
        # We'll need different MLPs for the two different sequence lengths in each layer
        self.layer_mlps_0 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len_0, seq_len_0 // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(seq_len_0 // 2, 1)
            ) for seq_len_0 in [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
        ])
        
        self.layer_mlps_1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len_1, seq_len_1 // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(seq_len_1 // 2, 1)
            ) for seq_len_1 in [9216, 9216, 2304, 2304, 576, 576, 144, 576, 576, 576, 2304, 2304, 2304, 9216, 9216, 9216]
        ])
        
        # Project each layer to a common dimension (for both tensors in each layer)
        self.layer_projections = nn.ModuleList([
            nn.Linear(dim * 2, hidden_dim) for dim in layer_dims  # *2 because we concatenate both tensors
        ])
        
        # Final decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, probe_out_list):
        layer_features = []
        
        for i, (mlp_0, mlp_1, proj, probe_out) in enumerate(zip(self.layer_mlps_0, self.layer_mlps_1, self.layer_projections, probe_out_list)):
            # Use both tensors from each layer
            features_0 = probe_out[0]  # Shape: [B, seq_len_0, feature_dim]
            features_1 = probe_out[1]  # Shape: [B, seq_len_1, feature_dim]
            
            # Transpose to apply MLP along sequence dimension
            features_0_t = features_0.transpose(1, 2)  # Shape: [B, feature_dim, seq_len_0]
            features_1_t = features_1.transpose(1, 2)  # Shape: [B, feature_dim, seq_len_1]
            
            # Apply MLP to reduce sequence length to 1
            reduced_0_t = mlp_0(features_0_t)  # Shape: [B, feature_dim, 1]
            reduced_1_t = mlp_1(features_1_t)  # Shape: [B, feature_dim, 1]
            
            # Transpose back and squeeze to get [B, feature_dim]
            reduced_0 = reduced_0_t.transpose(1, 2).squeeze(1)  # Shape: [B, feature_dim]
            reduced_1 = reduced_1_t.transpose(1, 2).squeeze(1)  # Shape: [B, feature_dim]
            
            # Concatenate the two reduced features
            concatenated = torch.cat([reduced_0, reduced_1], dim=1)  # Shape: [B, feature_dim * 2]
            
            # Project to common dimension
            projected = proj(concatenated)  # Shape: [B, hidden_dim]
            layer_features.append(projected)
        
        # Aggregate features across layers (average)
        aggregated = torch.stack(layer_features, dim=1).mean(dim=1)  # Shape: [B, hidden_dim]
        
        # Pass through decoder
        output = self.decoder(aggregated)  # Shape: [B, 1]
        
        return output



class PDecoderFC1(nn.Module):
    def __init__(self):
        super(PDecoderFC1, self).__init__()

        self.fc1 = nn.Linear(512 + 640 + 320 + 1280, 256)
        self.fc2 = nn.Linear(2560 + 640, 512)

        # Final convolutional layers
        self.final_fc1 = nn.Linear(256 + 320 + 640, 128)
        self.final_fc2 = nn.Linear(128, 64)
        self.final_fc3 = nn.Linear(64, 1)

        # Fully connected layer
        self.fc = nn.Linear(1, 1)

    def forward(self, feat_dict, mask, full_resolution=0):

        # ipdb.set_trace()
        # full_resolution = 1 means interpolate the feature to the original mask size
        # full_resolution = 2 means no mask feature pooling
        # full_resolution = 3 means ROI align to pool the feature

        
        # average pooling for temporal dimension
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 1:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                org_feat_h, org_feat_w = feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]
                feat_dict[key_i] = F.interpolate(feat_dict[key_i].reshape(-1, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4]), size=(int(mask.shape[-2]/2), int(mask.shape[-1]/2)), mode='bilinear', align_corners=False)
                feat_dict[key_i] = feat_dict[key_i].reshape(mask.shape[0], -1, feat_dict[key_i].shape[-3], feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                # feat_dict[key_i] = feat_dict[key_i] * mask.permute(0,2,1,3,4)
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
                feat_dict[key_i] = F.interpolate(feat_dict[key_i], size=(org_feat_h, org_feat_w), mode='bilinear', align_corners=False)
        elif full_resolution == 2:
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 3: # roi align
            for key_i in feat_dict.keys():
                roi_features = extract_roi_features(feat_dict, mask, key_i)
                feat_dict[key_i] = roi_features
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1



        # Upsample x3 and concatenate with x2
        # x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2].mean(-1).mean(-1), feat_dict[7].mean(-1).mean(-1), feat_dict[3].mean(-1).mean(-1)), dim=1)
        x2 = F.relu(self.fc2(x2))

        # Upsample x2 and concatenate with x1
        # x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1].mean(-1).mean(-1), feat_dict[5].mean(-1).mean(-1), feat_dict[6].mean(-1).mean(-1), x2), dim=1)
        x1 = F.relu(self.fc1(x1))

        # Upsample x1 and concatenate with x0
        # x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0].mean(-1).mean(-1), feat_dict[4].mean(-1).mean(-1), x1), dim=1)

        # Further processing
        x = F.relu(self.final_fc1(x0))
        x = F.relu(self.final_fc2(x))
        x = F.relu(self.final_fc3(x))

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x
    



class PDecoderFC2(nn.Module):
    def __init__(self):
        super(PDecoderFC2, self).__init__()

        self.fc1 = nn.Linear(512 + (640 + 320 + 1280) * 36 * 64, 256)
        self.fc2 = nn.Linear((2560 + 640) * 18 * 32, 512)

        # Final convolutional layers
        self.final_fc1 = nn.Linear(256 + (320 + 640) * 72 * 128, 128)
        self.final_fc2 = nn.Linear(128, 64)
        self.final_fc3 = nn.Linear(64, 1)

        # Fully connected layer
        self.fc = nn.Linear(1, 1)

    def forward(self, feat_dict, mask, full_resolution=0):

        # ipdb.set_trace()
        # full_resolution = 1 means interpolate the feature to the original mask size
        # full_resolution = 2 means no mask feature pooling
        # full_resolution = 3 means ROI align to pool the feature

        
        # average pooling for temporal dimension
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 1:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                org_feat_h, org_feat_w = feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]
                feat_dict[key_i] = F.interpolate(feat_dict[key_i].reshape(-1, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4]), size=(int(mask.shape[-2]/2), int(mask.shape[-1]/2)), mode='bilinear', align_corners=False)
                feat_dict[key_i] = feat_dict[key_i].reshape(mask.shape[0], -1, feat_dict[key_i].shape[-3], feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                # feat_dict[key_i] = feat_dict[key_i] * mask.permute(0,2,1,3,4)
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
                feat_dict[key_i] = F.interpolate(feat_dict[key_i], size=(org_feat_h, org_feat_w), mode='bilinear', align_corners=False)
        elif full_resolution == 2:
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 3: # roi align
            for key_i in feat_dict.keys():
                roi_features = extract_roi_features(feat_dict, mask, key_i)
                feat_dict[key_i] = roi_features
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1


        B = feat_dict[2].shape[0]
        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2].reshape(B, -1), feat_dict[7].reshape(B, -1), x3_up.reshape(B, -1)), dim=1)
        x2 = F.relu(self.fc2(x2))

        # Upsample x2 and concatenate with x1
        # x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1].reshape(B, -1), feat_dict[5].reshape(B, -1), feat_dict[6].reshape(B, -1), x2), dim=1)
        x1 = F.relu(self.fc1(x1))

        # Upsample x1 and concatenate with x0
        # x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0].reshape(B, -1), feat_dict[4].reshape(B, -1), x1), dim=1)

        # Further processing
        x = F.relu(self.final_fc1(x0))
        x = F.relu(self.final_fc2(x))
        x = F.relu(self.final_fc3(x))

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x
    



class PDecoderSigmoid(nn.Module):
    def __init__(self):
        super(PDecoderSigmoid, self).__init__()

        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)


        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(72 * 128, 1)

    def forward(self, feat_dict, mask, full_resolution=0):

        # ipdb.set_trace()
        # full_resolution = 1 means interpolate the feature to the original mask size
        # full_resolution = 2 means no mask feature pooling
        # full_resolution = 3 means ROI align to pool the feature

        
        # average pooling for temporal dimension
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 1:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                org_feat_h, org_feat_w = feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]
                feat_dict[key_i] = F.interpolate(feat_dict[key_i].reshape(-1, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4]), size=(int(mask.shape[-2]/2), int(mask.shape[-1]/2)), mode='bilinear', align_corners=False)
                feat_dict[key_i] = feat_dict[key_i].reshape(mask.shape[0], -1, feat_dict[key_i].shape[-3], feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                # feat_dict[key_i] = feat_dict[key_i] * mask.permute(0,2,1,3,4)
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
                feat_dict[key_i] = F.interpolate(feat_dict[key_i], size=(org_feat_h, org_feat_w), mode='bilinear', align_corners=False)
        elif full_resolution == 2:
            for key_i in feat_dict.keys():
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        elif full_resolution == 3: # roi align
            for key_i in feat_dict.keys():
                roi_features = extract_roi_features(feat_dict, mask, key_i)
                feat_dict[key_i] = roi_features
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1



        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        x = torch.sigmoid(x)

        return x
    



class PDecoderTemporalGRU(nn.Module):
    def __init__(self):
        super(PDecoderTemporalGRU, self).__init__()

        # GRU layers for each feature level - initialize them properly
        # Assuming we know the channel dimensions for each feature level
        self.gru_0 = nn.GRU(input_size=320, hidden_size=320, batch_first=True)
        self.gru_1 = nn.GRU(input_size=640, hidden_size=640, batch_first=True)
        self.gru_2 = nn.GRU(input_size=1280, hidden_size=1280, batch_first=True)
        self.gru_3 = nn.GRU(input_size=1280, hidden_size=1280, batch_first=True)
        self.gru_4 = nn.GRU(input_size=640, hidden_size=640, batch_first=True)
        self.gru_5 = nn.GRU(input_size=1280, hidden_size=1280, batch_first=True)
        self.gru_6 = nn.GRU(input_size=320, hidden_size=320, batch_first=True)
        self.gru_7 = nn.GRU(input_size=640, hidden_size=640, batch_first=True)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)

        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(72 * 128, 1)

    def _apply_gru_to_feature(self, feature, key, mask_):
        """Apply GRU to feature tensor after reshaping to handle spatial dimensions."""
        B, T, C, H, W = feature.shape
        
        # Apply mask
        masked_feature = feature * mask_.unsqueeze(2)
        
        # Reshape for GRU: [B*H*W, T, C]
        reshaped_feature = masked_feature.permute(0, 3, 4, 1, 2).reshape(B*H*W, T, C)

        # # Pack padded sequences for more efficient computation
        # packed_input = nn.utils.rnn.pack_padded_sequence(
        #     reshaped_feature, lengths, batch_first=True, enforce_sorted=False
        # )
        
        # # Process through GRU
        # packed_output, hidden = self.gru(packed_input)
        
        # # We take the final hidden state
        # # hidden shape: (num_layers, batch_size, hidden_size)
        # # We want the last layer's hidden state for each batch item
        # last_hidden = hidden[-1]  # Shape: (batch_size, hidden_size)
        
        # Apply the appropriate GRU based on the feature key
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

        
        
        # Take the last output of the sequence
        # last_output = gru_output[:, -1, :]  # [B*H*W, C]
        last_output = gru_output[-1]  # [B*H*W, C]
        
        # Reshape back to [B, C, H, W]
        output_feature = last_output.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return output_feature

    def forward(self, feat_dict, mask, full_resolution=0):
        # Process features with GRU instead of average pooling
        if full_resolution == 0:
            processed_feat_dict = {}
            for key_i in feat_dict.keys():
                # Interpolate mask to match feature size
                mask_ = F.interpolate(mask[:,0].float(), 
                                     size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), 
                                     mode='nearest')
                
                # Process with GRU
                processed_feat_dict[key_i] = self._apply_gru_to_feature(feat_dict[key_i], key_i, mask_)
            
            # Update feat_dict with processed features
            feat_dict = processed_feat_dict
        else:
            assert 0 == 1

        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        # x = torch.sigmoid(x)

        return x


class PDecoderTemporalConcat(nn.Module):
    def __init__(self):
        super(PDecoderTemporalConcat, self).__init__()

        self.conv1 = nn.Conv2d(512 + (640 + 320 + 1280) * 16, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d((2560 + 640)*16, 512, kernel_size=3, padding=1)


        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + (320 + 640) * 16, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(72 * 128, 1)

    def forward(self, feat_dict, mask, full_resolution=0):

        # ipdb.set_trace()
        # full_resolution = 1 means interpolate the feature to the original mask size
        # full_resolution = 2 means no mask feature pooling

        
        # average pooling for temporal dimension
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].reshape(feat_dict[key_i].shape[0], -1, feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
        # elif full_resolution == 1:
        #     for key_i in feat_dict.keys():
        #         # ipdb.set_trace()
        #         org_feat_h, org_feat_w = feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]
        #         feat_dict[key_i] = F.interpolate(feat_dict[key_i].reshape(-1, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4]), size=(int(mask.shape[-2]/2), int(mask.shape[-1]/2)), mode='bilinear', align_corners=False)
        #         feat_dict[key_i] = feat_dict[key_i].reshape(mask.shape[0], -1, feat_dict[key_i].shape[-3], feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
        #         mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
        #         # feat_dict[key_i] = feat_dict[key_i] * mask.permute(0,2,1,3,4)
        #         feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
        #         feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        #         feat_dict[key_i] = F.interpolate(feat_dict[key_i], size=(org_feat_h, org_feat_w), mode='bilinear', align_corners=False)
        # elif full_resolution == 2:
        #     for key_i in feat_dict.keys():
        #         feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        # elif full_resolution == 3: # roi align
        #     assert 1 == 1
        else:
            assert 0 == 1



        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x



class PDecoderCondMask(nn.Module):
    def __init__(self):
        super(PDecoderCondMask, self).__init__()

        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280 + 3, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640 + 3, 512, kernel_size=3, padding=1)


        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640 + 2, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(72 * 128, 1)

    def forward(self, feat_dict, mask, full_resolution=0):

        # ipdb.set_trace()
        # full_resolution = 1 means interpolate the feature to the original mask size
        # full_resolution = 2 means no mask feature pooling

        
        # average pooling for temporal dimension
        
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                # ipdb.set_trace()
                # feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = torch.cat([feat_dict[key_i], mask_.unsqueeze(2)], dim=2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        # elif full_resolution == 1:
        #     for key_i in feat_dict.keys():
        #         # ipdb.set_trace()
        #         org_feat_h, org_feat_w = feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]
        #         feat_dict[key_i] = F.interpolate(feat_dict[key_i].reshape(-1, feat_dict[key_i].shape[2], feat_dict[key_i].shape[3], feat_dict[key_i].shape[4]), size=(int(mask.shape[-2]/2), int(mask.shape[-1]/2)), mode='bilinear', align_corners=False)
        #         feat_dict[key_i] = feat_dict[key_i].reshape(mask.shape[0], -1, feat_dict[key_i].shape[-3], feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1])
        #         mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
        #         # feat_dict[key_i] = feat_dict[key_i] * mask.permute(0,2,1,3,4)
        #         feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
        #         feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        #         feat_dict[key_i] = F.interpolate(feat_dict[key_i], size=(org_feat_h, org_feat_w), mode='bilinear', align_corners=False)
        # elif full_resolution == 2:
        #     for key_i in feat_dict.keys():
        #         feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1



        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x



class PDecoderAdv(nn.Module):
    def __init__(self):
        super(PDecoderAdv, self).__init__()

        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)


        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(72 * 128, 12)

    def forward(self, feat_dict, mask, full_resolution=0):

        # ipdb.set_trace()
        # full_resolution = 1 means interpolate the feature to the original mask size
        # full_resolution = 2 means no mask feature pooling

        
        # average pooling for temporal dimension
        if full_resolution == 0:
            for key_i in feat_dict.keys():
                # ipdb.set_trace()
                mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
                feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
                feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        # elif full_resolution == 1:
        #     for key_i in feat_dict.keys():
        #         ipdb.set_trace()
        #         mask_ = F.interpolate(mask[:,0].float(), size=(feat_dict[key_i].shape[-2], feat_dict[key_i].shape[-1]), mode='nearest')
        #         feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
        #         feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        # elif full_resolution == 2:
        #     for key_i in feat_dict.keys():
        #         feat_dict[key_i] = feat_dict[key_i].mean(dim=1)
        else:
            assert 0 == 1



        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x
    


class PDecoderTemporalConv(nn.Module):
    def __init__(self):
        super(PDecoderTemporalConv, self).__init__()

        # Temporal convolution layer (1D conv applied on the temporal axis)
        self.temporal_conv = nn.Conv1d(in_channels=2949120, out_channels=2949120, kernel_size=3, padding=1, stride=1)

        self.conv1 = nn.Conv2d(512 + 640 + 320 + 1280, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2560 + 640, 512, kernel_size=3, padding=1)


        # Final convolutional layers
        self.final_conv1 = nn.Conv2d(256 + 320 + 640, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(72 * 128, 1)

    def forward(self, feat_dict, mask, full_resolution=0):

        # ipdb.set_trace()
        # full_resolution = 1 means interpolate the feature to the original mask size
        # full_resolution = 2 means no mask feature pooling

        
        for key_i in feat_dict.keys():
            batch, time, channels, height, width = feat_dict[key_i].shape

            # Apply temporal convolution: reshape (B, T, C, H, W) -> (B, C, T, H, W) -> (B, C, T)
            feat_temporal = feat_dict[key_i].permute(0, 2, 3, 4, 1)  # (B, C, H, W, T)
            feat_temporal = feat_temporal.reshape(batch, -1, time)  # Flatten spatial dims

            # Apply 1D convolution along the temporal axis
            feat_temporal = self.temporal_conv(feat_temporal)  # (B, C, T, Spatial)

            # Reshape back to (B, T, C, H, W)
            feat_temporal = feat_temporal.view(batch, channels, height, width, time)
            feat_dict[key_i] = feat_temporal.permute(0, 4, 1, 2, 3)  # Back to (B, T, C, H, W)

            # Apply spatial mask pooling
            mask_ = F.interpolate(mask[:, 0].float(), size=(height, width), mode='nearest')
            feat_dict[key_i] = feat_dict[key_i] * mask_.unsqueeze(2)
            feat_dict[key_i] = feat_dict[key_i].mean(dim=1)  # Temporal pooling


        # Upsample x3 and concatenate with x2
        x3_up = F.interpolate(feat_dict[3], size=(18, 32), mode='bilinear', align_corners=False)
        x2 = torch.cat((feat_dict[2], feat_dict[7], x3_up), dim=1)
        x2 = F.relu(self.conv2(x2))

        # Upsample x2 and concatenate with x1
        x2_up = F.interpolate(x2, size=(36, 64), mode='bilinear', align_corners=False)
        x1 = torch.cat((feat_dict[1], feat_dict[5], feat_dict[6], x2_up), dim=1)
        x1 = F.relu(self.conv1(x1))

        # Upsample x1 and concatenate with x0
        x1_up = F.interpolate(x1, size=(72, 128), mode='bilinear', align_corners=False)
        x0 = torch.cat((feat_dict[0], feat_dict[4], x1_up), dim=1)

        # Further processing
        x = F.relu(self.final_conv1(x0))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x