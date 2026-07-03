import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm.auto import tqdm

from src.clip import clip
from experiments.options import opts

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DFN5B_HF_ID = 'hf-hub:apple/DFN5B-CLIP-ViT-H-14-378'

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
        if not self.opts.train_text_encoder:
            freeze_text_encoder(self.clip)
        self.sk_clip = copy.deepcopy(self.clip)
        freeze_text_encoder(self.sk_clip)

        self.class_names = list(class_names) if class_names is not None else []
        self.category_to_idx = {category: idx for idx, category in enumerate(self.class_names)}
        text_prompts = ['a photo of a {}'.format(category) for category in self.class_names]
        if text_prompts:
            class_text_tokens = clip.tokenize(text_prompts)
        else:
            class_text_tokens = torch.empty(0, 77, dtype=torch.long)
        self.register_buffer('class_text_tokens', class_text_tokens, persistent=False)

        # Prompt Engineering
        self.prompt_init = self.opts.prompt_init
        self.prompt_dropout = nn.Dropout(p=self.opts.prompt_dropout)
        if self.prompt_init == 'clip_text':
            prompt = clip.tokenize('a photo/sketch of')
            with torch.no_grad():
                embedding = self.clip.token_embedding(prompt).type(self.clip.dtype)
            text_ctx = embedding[0, 1:1 + self.opts.n_prompts, :].float()
            self.sk_prompt_ctx = nn.Parameter(text_ctx.clone())
            self.img_prompt_ctx = nn.Parameter(text_ctx.clone())
            self.sk_prompt_proj = nn.Linear(text_ctx.shape[-1], self.opts.prompt_dim)
            self.img_prompt_proj = nn.Linear(text_ctx.shape[-1], self.opts.prompt_dim)
            if self.clip.dtype == torch.float16:
                self.sk_prompt_proj.half()
                self.img_prompt_proj.half()
        else:
            self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
            self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=self.opts.margin)

        self.best_metric = -1e3
        self.val_step_outputs = []
        self.teacher_model = [None]
        self.teacher_device = None
        self.teacher_input_size = self.opts.teacher_input_size
        self.teacher_mean = CLIP_MEAN
        self.teacher_std = CLIP_STD
        self.teacher_dtype = torch.float32
        self.teacher_feature_cache = {}
        self._teacher_precomputed = False
        self._init_teacher()

    def _init_teacher(self):
        if self.opts.distill_teacher == 'none' or self.opts.distill_weight <= 0:
            return
        if self.opts.distill_teacher != 'dfn5b':
            raise ValueError('Unsupported distillation teacher: {}'.format(self.opts.distill_teacher))
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                'DFN5B distillation requires open_clip_torch. Install it with: '
                'pip install open_clip_torch'
            ) from exc

        teacher, _, preprocess_val = open_clip.create_model_and_transforms(DFN5B_HF_ID)
        teacher.eval()
        teacher.requires_grad_(False)
        if self.opts.precision == '16-mixed':
            teacher = teacher.half()
            self.teacher_dtype = torch.float16
        elif self.opts.precision == 'bf16-mixed':
            teacher = teacher.bfloat16()
            self.teacher_dtype = torch.bfloat16
        self.teacher_model[0] = teacher
        self._set_teacher_preprocess(preprocess_val)

    def _set_teacher_preprocess(self, preprocess):
        for transform in getattr(preprocess, 'transforms', []):
            if transform.__class__.__name__ == 'Normalize':
                self.teacher_mean = list(transform.mean)
                self.teacher_std = list(transform.std)
            elif transform.__class__.__name__ in ('Resize', 'CenterCrop'):
                size = transform.size
                if isinstance(size, (tuple, list)):
                    self.teacher_input_size = int(size[0])
                else:
                    self.teacher_input_size = int(size)

    def _get_teacher(self):
        teacher = self.teacher_model[0]
        if teacher is None:
            return None
        if self.teacher_device != self.device:
            teacher.to(self.device)
            teacher.eval()
            self.teacher_device = self.device
        return teacher

    def _prepare_teacher_input(self, images):
        dtype = images.dtype
        clip_mean = torch.tensor(CLIP_MEAN, device=images.device, dtype=dtype).view(1, 3, 1, 1)
        clip_std = torch.tensor(CLIP_STD, device=images.device, dtype=dtype).view(1, 3, 1, 1)
        images = (images * clip_std + clip_mean).clamp(0, 1)
        images = F.interpolate(
            images,
            size=(self.teacher_input_size, self.teacher_input_size),
            mode='bicubic',
            align_corners=False)
        teacher_mean = torch.tensor(self.teacher_mean, device=images.device, dtype=dtype).view(1, 3, 1, 1)
        teacher_std = torch.tensor(self.teacher_std, device=images.device, dtype=dtype).view(1, 3, 1, 1)
        return ((images - teacher_mean) / teacher_std).to(dtype=self.teacher_dtype)

    def configure_optimizers(self):
        prompt_params = self.prompt_parameters()
        param_groups = [
            {'params': [p for p in self.clip.parameters() if p.requires_grad], 'lr': self.opts.clip_LN_lr},
            {'params': [p for p in self.sk_clip.parameters() if p.requires_grad], 'lr': self.opts.clip_LN_lr},
            {'params': prompt_params, 'lr': self.opts.prompt_lr},
        ]
        if self.opts.optimizer == 'sgd':
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-3)
        else:
            optimizer = torch.optim.Adam(param_groups)
        return optimizer

    def prompt_parameters(self):
        if self.prompt_init == 'clip_text':
            return (
                [self.sk_prompt_ctx, self.img_prompt_ctx]
                + list(self.sk_prompt_proj.parameters())
                + list(self.img_prompt_proj.parameters())
            )
        return [self.sk_prompt, self.img_prompt]

    def visual_prompt(self, dtype, batch_size, branch):
        if self.prompt_init == 'clip_text':
            if branch == 'image':
                ctx = self.img_prompt_ctx
                if self.training:
                    ctx = self.prompt_dropout(ctx)
                prompt = self.img_prompt_proj(ctx.type(dtype))
            else:
                ctx = self.sk_prompt_ctx
                if self.training:
                    ctx = self.prompt_dropout(ctx)
                prompt = self.sk_prompt_proj(ctx.type(dtype))
        else:
            prompt = self.img_prompt if branch == 'image' else self.sk_prompt
            if self.training:
                prompt = self.prompt_dropout(prompt)
        return prompt.type(dtype).expand(batch_size, -1, -1)

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.visual_prompt(data.dtype, data.shape[0], 'image'))
        else:
            feat = self.sk_clip.encode_image(
                data, self.visual_prompt(data.dtype, data.shape[0], 'sketch'))
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

    def nt_xent_loss(self, sk_feat, img_feat, categories=None):
        sk_feat = F.normalize(sk_feat.float(), dim=-1)
        img_feat = F.normalize(img_feat.float(), dim=-1)

        batch_size = sk_feat.shape[0]
        z = torch.cat([img_feat, sk_feat], dim=0)
        logits = z @ z.t()
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(mask, float('-inf'))
        logits = logits / self.opts.nt_xent_temperature

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=logits.device),
            torch.arange(0, batch_size, device=logits.device),
        ], dim=0).long()

        return F.cross_entropy(logits, labels)

    def teacher_encode_image(self, images):
        teacher = self._get_teacher()
        if teacher is None:
            return None
        teacher_feats = []
        with torch.inference_mode():
            for start in range(0, images.size(0), self.opts.teacher_batch_size):
                batch = images[start:start + self.opts.teacher_batch_size]
                teacher_images = self._prepare_teacher_input(batch)
                teacher_feat = teacher.encode_image(teacher_images)
                teacher_feats.append(F.normalize(teacher_feat.float(), dim=-1))
        return torch.cat(teacher_feats, dim=0)

    def teacher_encode_path_batch(self, image_tensor, image_paths):
        if not self.opts.cache_teacher_features or any(path is None for path in image_paths):
            return self.teacher_encode_image(image_tensor)

        cached_feats = []
        missing_indices = []
        missing_images = []

        for idx, path in enumerate(image_paths):
            if path in self.teacher_feature_cache:
                cached_feats.append(self.teacher_feature_cache[path].to(self.device).float())
            else:
                cached_feats.append(None)
                missing_indices.append(idx)
                missing_images.append(image_tensor[idx])

        if missing_images:
            missing_batch = torch.stack(missing_images, dim=0)
            missing_feats = self.teacher_encode_image(missing_batch).detach().cpu().half()
            for path, feat in zip([image_paths[idx] for idx in missing_indices], missing_feats):
                self.teacher_feature_cache[path] = feat

        feats = [
            self.teacher_feature_cache[path].to(self.device).float()
            if feat is None else feat
            for path, feat in zip(image_paths, cached_feats)
        ]
        return torch.stack(feats, dim=0)

    def _precompute_teacher_cache(self, dataset, paths, desc):
        if not paths:
            return

        missing_paths = [path for path in dict.fromkeys(paths) if path not in self.teacher_feature_cache]
        if not missing_paths:
            return

        for start in tqdm(
            range(0, len(missing_paths), self.opts.teacher_batch_size),
            desc=desc,
            dynamic_ncols=True,
            leave=True):
            batch_paths = missing_paths[start:start + self.opts.teacher_batch_size]
            batch = torch.stack([
                dataset.transform(dataset.load_image(path))
                for path in batch_paths
            ]).to(self.device)
            feats = self.teacher_encode_image(batch).detach().cpu().half()
            for path, feat in zip(batch_paths, feats):
                self.teacher_feature_cache[path] = feat

    def on_train_start(self):
        if (
            self.opts.distill_teacher == 'none'
            or self.opts.distill_weight <= 0
            or not self.opts.cache_teacher_features
            or not self.opts.precompute_teacher_features
            or self._teacher_precomputed
        ):
            return

        train_dataloader = getattr(self.trainer, 'train_dataloader', None)
        if train_dataloader is None:
            train_dataloader = getattr(self.trainer, 'train_dataloaders', None)
        if isinstance(train_dataloader, (list, tuple)):
            train_dataloader = train_dataloader[0]
        if train_dataloader is None or not hasattr(train_dataloader, 'dataset'):
            return
        train_dataset = train_dataloader.dataset
        self._precompute_teacher_cache(
            train_dataset,
            train_dataset.all_photo_paths,
            'Precompute DFN photo')
        self._precompute_teacher_cache(
            train_dataset,
            train_dataset.all_sketches_path,
            'Precompute DFN sketch')
        teacher = self.teacher_model[0]
        if teacher is not None:
            teacher.to('cpu')
            self.teacher_device = torch.device('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._teacher_precomputed = True

    def distillation_loss(self, sk_tensor, img_tensor, sk_feat, img_feat, sk_paths, img_paths):
        if self.opts.distill_teacher == 'none' or self.opts.distill_weight <= 0:
            return img_feat.new_tensor(0.0)

        teacher_sk_feat = self.teacher_encode_path_batch(sk_tensor, sk_paths)
        teacher_img_feat = self.teacher_encode_path_batch(img_tensor, img_paths)
        temperature = self.opts.distill_temperature

        student_sk_feat = F.normalize(sk_feat.float(), dim=-1)
        student_img_feat = F.normalize(img_feat.float(), dim=-1)
        student_logits = student_sk_feat @ student_img_feat.t() / temperature
        teacher_logits = teacher_sk_feat @ teacher_img_feat.t() / temperature

        loss_sk_to_img = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean')
        loss_img_to_sk = F.kl_div(
            F.log_softmax(student_logits.t(), dim=1),
            F.softmax(teacher_logits.t(), dim=1),
            reduction='batchmean')
        return 0.5 * (loss_sk_to_img + loss_img_to_sk) * (temperature ** 2)

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        if len(batch) > 6:
            sk_paths = batch[5]
            img_paths = batch[6]
        else:
            sk_paths = [None] * sk_tensor.size(0)
            img_paths = [None] * img_tensor.size(0)
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        cls_loss = self.classification_loss(sk_feat, img_feat, category)
        if self.opts.nt_xent_weight > 0:
            nt_xent_loss = self.nt_xent_loss(sk_feat, img_feat, category)
        else:
            nt_xent_loss = img_feat.new_tensor(0.0)
        distill_loss = self.distillation_loss(sk_tensor, img_tensor, sk_feat, img_feat, sk_paths, img_paths)
        loss = (
            self.opts.triplet_weight * triplet_loss
            + self.opts.cls_loss_weight * cls_loss
            + self.opts.nt_xent_weight * nt_xent_loss
            + self.opts.distill_weight * distill_loss)
        batch_size = sk_tensor.size(0)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_triplet_loss', triplet_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_cls_loss', cls_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_nt_xent_loss', nt_xent_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_distill_loss', distill_loss, on_step=False, on_epoch=True, batch_size=batch_size)
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

    def _metric_config(self):
        if self.opts.dataset == 'sketchy_1':
            return {
                'map_k': None,
                'precision_k': 100,
                'map_name': 'mAP@all',
                'precision_name': 'P@100',
            }
        if self.opts.dataset in ('sketchy', 'sketchy_2'):
            return {
                'map_k': 200,
                'precision_k': 200,
                'map_name': 'mAP@200',
                'precision_name': 'P@200',
            }
        if self.opts.dataset == 'tuberlin':
            return {
                'map_k': None,
                'precision_k': 100,
                'map_name': 'mAP@all',
                'precision_name': 'P@100',
            }
        if self.opts.dataset == 'quickdraw':
            return {
                'map_k': None,
                'precision_k': 200,
                'map_name': 'mAP@all',
                'precision_name': 'P@200',
            }
        raise ValueError('Unsupported dataset for retrieval metrics: {}'.format(self.opts.dataset))

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

        metric_cfg = self._metric_config()
        ap = []
        precision_at_k = []
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

                if metric_cfg['map_k'] is None:
                    ap_limit = target.numel()
                    ap_denominator = target.sum()
                else:
                    ap_limit = min(metric_cfg['map_k'], target.numel())
                    ap_denominator = torch.clamp(target.sum(), max=float(ap_limit))
                ap.append((precision[:ap_limit] * target[:ap_limit]).sum() / ap_denominator)

                p_limit = min(metric_cfg['precision_k'], target.numel())
                precision_at_k.append(target[:p_limit].sum() / float(p_limit))

        if len(ap) == 0:
            mAP = torch.tensor(0.0, device=self.device)
            precision_metric = torch.tensor(0.0, device=self.device)
        else:
            mAP = torch.stack(ap).mean()
            precision_metric = torch.stack(precision_at_k).mean()

        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        self.log(metric_cfg['map_name'], mAP)
        self.log(metric_cfg['precision_name'], precision_metric)
        self.log('main_metric', mAP)
        self.log('best_mAP', self.best_metric)
        val_loss = self.trainer.callback_metrics.get('val_loss')
        val_loss_text = 'n/a' if val_loss is None else '{:.4f}'.format(float(val_loss))
        print('val epoch {} | val_loss: {} | {}: {:.4f} | {}: {:.4f} | best_{}: {:.4f} | missing_positive_queries: {}'.format(
            self.current_epoch,
            val_loss_text,
            metric_cfg['map_name'],
            mAP.item(),
            metric_cfg['precision_name'],
            precision_metric.item(),
            metric_cfg['map_name'],
            self.best_metric,
            missing_positive))
        self.val_step_outputs = []
