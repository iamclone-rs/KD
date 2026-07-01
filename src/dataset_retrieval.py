import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

from src.data_config import UNSEEN_CLASSES

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig
        self.mode = mode

        self._init_dataset(mode, used_cat)

    def _init_dataset(self, mode, used_cat=None):
        self.all_categories = sorted(os.listdir(os.path.join(self.opts.data_dir, 'sketch')))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')

        unseen_classes = UNSEEN_CLASSES[self.opts.dataset]
        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                used_cat = set(used_cat)
                self.all_categories = [cat for cat in self.all_categories if cat not in used_cat]
        else:
            if mode == 'train':
                unseen_cat = set(unseen_classes)
                self.all_categories = [cat for cat in self.all_categories if cat not in unseen_cat]
            else:
                available_cat = set(self.all_categories)
                self.all_categories = [cat for cat in unseen_classes if cat in available_cat]

        self.all_sketches_path = []
        self.all_sketch_categories = []
        self.all_photos_path = {}
        self.all_photo_paths = []
        self.all_photo_categories = []

        for category in self.all_categories:
            sketch_paths = self._list_images(os.path.join(self.opts.data_dir, 'sketch', category))
            photo_paths = self._list_images(os.path.join(self.opts.data_dir, 'photo', category))

            self.all_sketches_path.extend(sketch_paths)
            self.all_sketch_categories.extend([category] * len(sketch_paths))
            self.all_photos_path[category] = photo_paths
            self.all_photo_paths.extend(photo_paths)
            self.all_photo_categories.extend([category] * len(photo_paths))

    def _list_images(self, folder):
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.JPEG', '*.bmp')
        paths = []
        for extension in extensions:
            paths.extend(glob.glob(os.path.join(folder, extension)))
        return sorted(paths)

    def __len__(self):
        return len(self.all_sketches_path)

    def load_image(self, filepath):
        with Image.open(filepath) as image:
            image = ImageOps.pad(
                image.convert('RGB'),
                size=(self.opts.max_size, self.opts.max_size))
        return image
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]
        category = self.all_sketch_categories[index]
        filename = os.path.basename(filepath)
        
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path  = filepath
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data = self.load_image(sk_path)
        img_data = self.load_image(img_path)
        neg_data = self.load_image(neg_path)

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=True)
    dataset_val = Sketchy(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, filename,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
