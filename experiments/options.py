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
parser.add_argument('--prompt_lr', type=float, default=1e-5)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--cls_loss_weight', type=float, default=0.5)
parser.add_argument('--precision', type=str, default='16-mixed',
    choices=['16-mixed', 'bf16-mixed', '32-true'])
parser.add_argument('--distill_teacher', type=str, default='none', choices=['none', 'dfn5b'])
parser.add_argument('--distill_weight', type=float, default=0.0)
parser.add_argument('--distill_temperature', type=float, default=0.07)
parser.add_argument('--teacher_input_size', type=int, default=378)
parser.add_argument('--teacher_batch_size', type=int, default=8)
parser.add_argument('--cache_teacher_features', action='store_true')
parser.add_argument('--precompute_teacher_features', action='store_true')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--workers', type=int, default=8)

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

opts = parser.parse_args()
