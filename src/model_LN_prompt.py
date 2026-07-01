import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

def freeze_text_encoder(m):
    for name, param in m.named_parameters():
        if not name.startswith('visual.'):
            param.requires_grad_(False)

class Model(pl.LightningModule):
    def __init__(self, class_names=None):
        super().__init__()

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip.apply(freeze_all_but_bn)
        freeze_text_encoder(self.clip)
        self.sk_clip = copy.deepcopy(self.clip)

        self.class_names = list(class_names) if class_names is not None else []
        self.category_to_idx = {category: idx for idx, category in enumerate(self.class_names)}
        text_prompts = ['a photo of a {}'.format(category) for category in self.class_names]
        if text_prompts:
            class_text_tokens = clip.tokenize(text_prompts)
        else:
            class_text_tokens = torch.empty(0, 77, dtype=torch.long)
        self.register_buffer('class_text_tokens', class_text_tokens, persistent=False)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=self.opts.margin)

        self.best_metric = -1e3

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': [p for p in self.clip.parameters() if p.requires_grad], 'lr': self.opts.clip_LN_lr},
            {'params': [p for p in self.sk_clip.parameters() if p.requires_grad], 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.sk_clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def classification_loss(self, sk_feat, img_feat, categories):
        if not self.category_to_idx:
            raise RuntimeError('class_names must be provided to train classification loss.')

        labels = torch.tensor(
            [self.category_to_idx[category] for category in categories],
            dtype=torch.long,
            device=self.device)

        with torch.no_grad():
            text_feat = self.clip.encode_text(self.class_text_tokens.to(self.device))
            text_feat = F.normalize(text_feat, dim=-1)

        logit_scale = self.clip.logit_scale.exp().detach()
        sk_logits = logit_scale * F.normalize(sk_feat, dim=-1) @ text_feat.t()
        img_logits = logit_scale * F.normalize(img_feat, dim=-1) @ text_feat.t()
        return F.cross_entropy(sk_logits, labels) + F.cross_entropy(img_logits, labels)

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        cls_loss = self.classification_loss(sk_feat, img_feat, category)
        loss = triplet_loss + self.opts.cls_loss_weight * cls_loss
        self.log('train_loss', loss)
        self.log('train_triplet_loss', triplet_loss)
        self.log('train_cls_loss', cls_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', loss)
        return sk_feat, category

    def _get_validation_dataset(self):
        val_dataloaders = self.trainer.val_dataloaders
        if isinstance(val_dataloaders, (list, tuple)):
            val_dataloader = val_dataloaders[0]
        else:
            val_dataloader = val_dataloaders
        return val_dataloader.dataset

    def _encode_validation_gallery(self, dataset):
        gallery_feats = []
        batch_size = self.opts.batch_size

        with torch.no_grad():
            for start in range(0, len(dataset.all_photo_paths), batch_size):
                paths = dataset.all_photo_paths[start:start + batch_size]
                batch = torch.stack([
                    dataset.transform(dataset.load_image(path))
                    for path in paths
                ]).to(self.device)
                gallery_feats.append(self.forward(batch, dtype='image').detach())

        return torch.cat(gallery_feats)

    def validation_epoch_end(self, val_step_outputs):
        Len = len(val_step_outputs)
        if Len == 0:
            return
        query_feat_all = torch.cat([val_step_outputs[i][0] for i in range(Len)])
        query_category_all = np.array(sum([list(val_step_outputs[i][1]) for i in range(Len)], []))
        val_dataset = self._get_validation_dataset()
        gallery_feat_all = self._encode_validation_gallery(val_dataset)
        gallery_category_all = np.array(val_dataset.all_photo_categories)


        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        for idx, sk_feat in enumerate(query_feat_all):
            category = query_category_all[idx]
            distance = -1*self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(gallery_category_all == category)] = True
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
        
        mAP = torch.mean(ap)
        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        self.log('mAP', mAP, prog_bar=True)
        self.log('best_mAP', self.best_metric, prog_bar=True)
