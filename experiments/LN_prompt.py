import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy
from experiments.options import opts

if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}-{val_loss:.4f}",
        mode='min',
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if torch.cuda.is_available() else 'auto'

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        min_epochs=1, max_epochs=2000,
        benchmark=True,
        logger=logger,
        # val_check_interval=10, 
        # accumulate_grad_batches=1,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback]
    )

    if ckpt_path is None:
        model = Model(class_names=train_dataset.all_categories)
    else:
        print ('resuming training from %s'%ckpt_path)
        model = Model.load_from_checkpoint(ckpt_path, class_names=train_dataset.all_categories, strict=False)

    print ('beginning training...good luck...')
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
