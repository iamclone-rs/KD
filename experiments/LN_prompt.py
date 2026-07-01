import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy
from experiments.options import opts

class TrainProgressBar(Callback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def on_train_epoch_start(self, trainer, pl_module):
        total = trainer.num_training_batches
        self.progress_bar = tqdm(
            total=total,
            desc='Epoch {}/{}'.format(trainer.current_epoch + 1, trainer.max_epochs),
            dynamic_ncols=True,
            leave=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False)

    loader_kwargs = {
        'batch_size': opts.batch_size,
        'num_workers': opts.workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': opts.workers > 0,
    }
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(dataset=val_dataset, **loader_kwargs)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='main_metric',
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}-{main_metric:.4f}",
        mode='max',
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if torch.cuda.is_available() else 'auto'
    precision = opts.precision if torch.cuda.is_available() else '32-true'

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        min_epochs=1, max_epochs=2000,
        benchmark=True,
        logger=logger,
        # val_check_interval=10, 
        # accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, TrainProgressBar()]
    )

    if ckpt_path is None:
        model = Model(class_names=train_dataset.all_categories)
    else:
        print ('resuming training from %s'%ckpt_path)
        model = Model.load_from_checkpoint(ckpt_path, class_names=train_dataset.all_categories, strict=False)

    print ('beginning training...good luck...')
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
