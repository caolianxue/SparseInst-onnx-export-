# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import build_backbone
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .encoder import build_sparse_inst_encoder
from .decoder import build_sparse_inst_decoder
from .loss import build_sparse_inst_criterion
from .utils import nested_tensor_from_tensor_list

__all__ = ["SparseInst"]


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))


@META_ARCH_REGISTRY.register()
class SparseInst(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # move to target device
        self.device = torch.device(cfg.MODEL.DEVICE)

        # backbone
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        output_shape = self.backbone.output_shape()

        # encoder & decoder
        self.encoder = build_sparse_inst_encoder(cfg, output_shape)
        self.decoder = build_sparse_inst_decoder(cfg)

        # matcher & loss (matcher is built in loss)
        self.criterion = build_sparse_inst_criterion(cfg)

        # data and preprocessing
        self.mask_format = cfg.INPUT.MASK_FORMAT

        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        # self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # inference
        self.cls_threshold = cfg.MODEL.SPARSE_INST.CLS_THRESHOLD
        self.mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
        self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS
    
    def normalizer(self, image):
        image = (image - self.pixel_mean) / self.pixel_std
        return image

    def preprocess_inputs(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, 32)
        return images

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            gt_classes = targets_per_image.gt_classes
            target["labels"] = gt_classes.to(self.device)
            h, w = targets_per_image.image_size
            if not targets_per_image.has('gt_masks'):
                gt_masks = BitMasks(torch.empty(0, h, w))
            else:
                gt_masks = targets_per_image.gt_masks
                if self.mask_format == "polygon":
                    if len(gt_masks.polygons) == 0:
                        gt_masks = BitMasks(torch.empty(0, h, w))
                    else:
                        gt_masks = BitMasks.from_polygon_masks(gt_masks.polygons, h, w)

            target["masks"] = gt_masks.to(self.device)
            new_targets.append(target)

        return new_targets

    def forward(self, batched_inputs):
        images = self.preprocess_inputs(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        max_shape = images.tensor.shape[2:]
        # forward
        features = self.backbone(images.tensor)
        features = self.encoder(features)
        output = self.decoder(features)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            losses = self.criterion(output, targets, max_shape)
            return losses
        else:
            results = self.inference(output, batched_inputs, max_shape, images.image_sizes)
            processed_results = [{"instances": r} for r in results]
            return processed_results

    def forward_test(self, images):
        pass


    def inference(self, output, batched_inputs, max_shape, image_sizes):
        # max_detections = self.max_detections
        results = []
        pred_scores = output["pred_logits"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()
        pred_objectness = output["pred_scores"].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_objectness)

        for _, (scores_per_image, mask_pred_per_image, batched_input, img_shape) in enumerate(zip(
                pred_scores, pred_masks, batched_inputs, image_sizes)):

            ori_shape = (batched_input["height"], batched_input["width"])
            result = Instances(ori_shape)
            # max/argmax
            scores, labels = scores_per_image.max(dim=-1)
            # cls threshold
            keep = scores > self.cls_threshold
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = mask_pred_per_image[keep]

            if scores.size(0) == 0:
                result.scores = scores
                result.pred_classes = labels
                results.append(result)
                continue

            h, w = img_shape
            # rescoring mask using maskness
            scores = rescoring_mask(scores, mask_pred_per_image > self.mask_threshold, mask_pred_per_image)

            # upsample the masks to the original resolution:
            # (1) upsampling the masks to the padded inputs, remove the padding area
            # (2) upsampling/downsampling the masks to the original sizes
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image.unsqueeze(1), size=max_shape, mode="bilinear", align_corners=False)[:, :, :h, :w]
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image, size=ori_shape, mode='bilinear', align_corners=False).squeeze(1)

            mask_pred = mask_pred_per_image > self.mask_threshold
            mask_pred = BitMasks(mask_pred)

            # using Detectron2 Instances to store the final results
            result.pred_masks = mask_pred
            result.scores = scores
            result.pred_classes = labels
            results.append(result)

        return results

    

