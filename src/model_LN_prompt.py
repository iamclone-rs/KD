import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.val_step_outputs = []

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
        batch_size = sk_tensor.size(0)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_triplet_loss', triplet_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_cls_loss', cls_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=sk_tensor.size(0))
        self.val_step_outputs.append((sk_feat.detach(), list(category)))

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

        if len(gallery_feats) == 0:
            raise RuntimeError('Validation gallery is empty. Check data_dir, dataset split, and photo folders.')
        return torch.cat(gallery_feats)

    def on_validation_epoch_start(self):
        self.val_step_outputs = []

    def on_validation_epoch_end(self):
        Len = len(self.val_step_outputs)
        if Len == 0:
            return
        query_feat_all = torch.cat([self.val_step_outputs[i][0] for i in range(Len)])
        query_category_all = np.array(sum([self.val_step_outputs[i][1] for i in range(Len)], []))
        val_dataset = self._get_validation_dataset()
        gallery_feat_all = self._encode_validation_gallery(val_dataset)
        gallery_category_all = np.array(val_dataset.all_photo_categories)


        query_feat_all = F.normalize(query_feat_all.float(), dim=-1)
        gallery_feat_all = F.normalize(gallery_feat_all.float(), dim=-1)

        ap = []
        missing_positive = 0
        eval_batch_size = self.opts.batch_size
        for start in range(0, len(query_feat_all), eval_batch_size):
            end = start + eval_batch_size
            scores_batch = query_feat_all[start:end] @ gallery_feat_all.t()
            categories_batch = query_category_all[start:end]

            for local_idx, scores in enumerate(scores_batch):
                category = categories_batch[local_idx]
                target_np = gallery_category_all == category
                if not np.any(target_np):
                    missing_positive += 1
                    continue

                target = torch.from_numpy(target_np).to(scores.device)
                order = torch.argsort(scores, descending=True)
                target = target[order].float()
                precision = torch.cumsum(target, dim=0) / torch.arange(
                    1, target.numel() + 1, device=target.device, dtype=torch.float32)
                ap.append((precision * target).sum() / target.sum())

        if len(ap) == 0:
            mAP = torch.tensor(0.0, device=self.device)
        else:
            mAP = torch.stack(ap).mean()

        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        self.log('mAP', mAP)
        self.log('best_mAP', self.best_metric)
        val_loss = self.trainer.callback_metrics.get('val_loss')
        val_loss_text = 'n/a' if val_loss is None else '{:.4f}'.format(float(val_loss))
        print('val epoch {} | val_loss: {} | mAP: {:.4f} | best_mAP: {:.4f} | missing_positive_queries: {}'.format(
            self.current_epoch, val_loss_text, mAP.item(), self.best_metric, missing_positive))
        self.val_step_outputs = []
