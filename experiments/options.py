import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')

parser.add_argument('--exp_name', type=str, default='LN_prompt')

# --------------------
# DataLoader Options
# --------------------

# Path to dataset root. It should have sketch/photo folders with class subfolders.
parser.add_argument('--data_dir', type=str, default='/isize2/sain/data/Sketchy/')
parser.add_argument(
    '--dataset',
    type=str,
    default='sketchy_2',
    choices=['sketchy', 'sketchy_1', 'sketchy_2', 'tuberlin', 'quickdraw'])
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)

# ----------------------
# Training Params
# ----------------------

parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--cls_loss_weight', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=192)
parser.add_argument('--workers', type=int, default=128)

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

opts = parser.parse_args()
