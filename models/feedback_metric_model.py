import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel
from torch.nn.parameter import Parameter
import math
from typing import Any, Dict
import numpy as np


class NLPclsTokenPooling(nn.Module):
    """Max Pooling"""

    def __init__(self, dim):
        super(NLPclsTokenPooling, self).__init__()
        self.feat_mult = 1

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, input_ids: torch.Tensor, cfg:Any) -> torch.Tensor:
        cls_token = x[:, 0]  # Take embeddings of first token per sample
        return cls_token


class GeMText(nn.Module):
    """GeM Pooling for NLP"""

    def __init__(self, dim, p=3, eps=1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.feat_mult = 1

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, input_ids: torch.Tensor, cfg:Any) -> torch.Tensor:
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
        x = (x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class NLPPoolings:

    _poolings = {
        "GeM": GeMText,
        "[CLS] token": NLPclsTokenPooling,
    }

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._poolings.get(name)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, reduction="mean"):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        if reduction == "mean":
            return loss.mean()
        else:
            return loss


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.cfg = cfg

        self.s = self.cfg.training.arcface_scale
        self.margins = cfg.dataset._margins
        print(self.margins)
        self.out_dim = self.cfg.dataset._num_labels

        self.head = ArcMarginProduct(
            self.cfg.architecture.embedding_size, self.cfg.dataset._num_labels
        )

    def forward(self, embeddings, labels, batch=None):

        logits = self.head(embeddings)
        logits = logits.float()

        ms = []
        ms = self.margins[labels.cpu().numpy()]

        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss


class ArcMarginProduct(nn.Module):
    """Arc margin product for head of ArcFace Loss"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcFaceLoss(nn.modules.Module):
    """Calculate ArcFace Loss"""

    def __init__(self, cfg: Any):
        super().__init__()

        self.cfg = cfg

        s = self.cfg.training.arcface_scale
        m = self.cfg.training.arcface_margin

        self.head = ArcMarginProduct(
            self.cfg.architecture.embedding_size, self.cfg.dataset._num_labels
        )
        self.crit = nn.CrossEntropyLoss(reduction="mean")

        self.init(s, m)

    def init(self, s, m):
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        logits = self.head(embeddings)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)

        output = labels2 * phi
        output = output + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        return loss


class Net(nn.Module):

    def __init__(self, cfg: Any):

        super(Net, self).__init__()

        self.cfg = cfg
        self.embedding_size = self.cfg.architecture.embedding_size

        self.backbone = self.create_nlp_backbone(self.cfg)

        self.backbone.pooler = None
        self.pooling = NLPPoolings.get(self.cfg.architecture.pool)
        self.pooling = self.pooling(1)
        self.neck = self._create_neck()

        if self.cfg.training.loss_function == "ArcFaceLossAdaptiveMargin":
            self.loss_fn = ArcFaceLossAdaptiveMargin(self.cfg)
        else:
            self.loss_fn = ArcFaceLoss(self.cfg)

    def create_nlp_backbone(self, cfg, model_class=AutoModel):
        config = AutoConfig.from_pretrained(cfg.architecture.backbone)
        config.hidden_dropout_prob = cfg.architecture.intermediate_dropout
        config.attention_probs_dropout_prob = cfg.architecture.intermediate_dropout

        backbone = model_class.from_pretrained(
            cfg.architecture.backbone,
            config=config,
        )

        return backbone

    def _num_outputs(self):
        return self.backbone.config.hidden_size * self.pooling.feat_mult

    def get_features(self, batch: Dict) -> torch.Tensor:
        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]

        idx = torch.quantile(torch.where(attention_mask==1)[1].float(), self.cfg.tokenizer.padding_quantile).long()
        attention_mask = attention_mask[:, :idx]
        input_ids = input_ids[:, :idx]

        x = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        x = self.pooling(x, attention_mask, input_ids, self.cfg)

        return x

    def _create_neck(self) -> torch.nn.Module:
        neck = nn.Sequential(
            nn.Linear(
                self._num_outputs(),
                self.embedding_size,
                bias=True,
            ),
            nn.BatchNorm1d(self.embedding_size),
        )
        return neck

    def forward(self, batch: Dict) -> Dict:
        x = self.get_features(batch)
        if self.cfg.architecture.dropout > 0.0:
            x = F.dropout(x, p=self.cfg.architecture.dropout, training=self.training)

        embeddings = self.neck(x)
        outputs = {}

        if not self.training:
            outputs["embeddings"] = embeddings

        targets = batch["target"].long()

        if self.training:
            loss = self.loss_fn(embeddings, targets)
        else:
            loss = torch.Tensor([0]).to(embeddings.device)
        outputs["loss"] = loss

        return outputs
