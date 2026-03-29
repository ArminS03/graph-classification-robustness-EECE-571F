PROTEINS: 

  python robustness_eval.py \
  --dataset PROTEINS \
  --checkpoint_dir checkpoints/proteins \
  --experiment baseline

  python robustness_eval.py \
  --dataset PROTEINS \
  --checkpoint_dir checkpoints/proteins \
  --experiment edge_drop

  python robustness_eval.py \
  --dataset PROTEINS \
  --checkpoint_dir checkpoints/proteins \
  --experiment all_augmentations

obgb-molhiv:

  python robustness_eval.py \
  --dataset ogbg-molhiv \
  --checkpoint_dir results/2026-03-27_ogbg-molhiv/checkpoints \
  --experiment baseline

  python robustness_eval.py \
  --dataset ogbg-molhiv \
  --checkpoint_dir results/2026-03-27_ogbg-molhiv/checkpoints \
  --experiment edge_drop

  python robustness_eval.py \
  --dataset ogbg-molhiv \
  --checkpoint_dir results/2026-03-27_ogbg-molhiv/checkpoints \
  --experiment all_augmentations