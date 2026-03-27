import torch
import torch.nn as nn
import torch.nn.functional as F

import lmmd
from cnn_1d import cnn_features


class DSAN(nn.Module):
    def __init__(self, num_classes=12, bottle_neck=False, bottleneck_dim=128):
        super().__init__()
        self.feature_layers = cnn_features()
        self.lmmd_loss = lmmd.LMMD_loss(class_num=num_classes)
        self.bottle_neck = bottle_neck

        feat_dim = self.feature_layers.output_num() if hasattr(self.feature_layers, 'output_num') else 256
        self.feature_dim = feat_dim

        if bottle_neck:
            self.bottle = nn.Sequential(
                nn.Linear(feat_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
            )
            cls_in = bottleneck_dim
        else:
            self.bottle = None
            cls_in = feat_dim

        self.cls_fc = nn.Linear(cls_in, num_classes)

    def _forward_features(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck and self.bottle is not None:
            x = self.bottle(x)
        return x

    def forward(self, source, target, s_label, adaptation_weight=1.0):
        source_feat = self._forward_features(source)
        source_logits = self.cls_fc(source_feat)

        if adaptation_weight <= 0:
            loss_lmmd = source_logits.new_tensor(0.0)
            return source_logits, loss_lmmd

        target_feat = self._forward_features(target)
        target_logits = self.cls_fc(target_feat)
        target_prob = F.softmax(target_logits, dim=1)
        loss_lmmd = self.lmmd_loss.get_loss(source_feat, target_feat, s_label, target_prob)
        return source_logits, loss_lmmd

    def predict(self, x):
        x = self._forward_features(x)
        return self.cls_fc(x)
