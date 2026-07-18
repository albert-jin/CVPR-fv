"""Command-line entry point for CVPR-FV.

Examples
--------

    # Few-shot on FEVER with T0-3B, standard CVP-guided aggregation:
    python train.py --dataset fever --shot_num 4 --seed 0 \
                    --backbone t0-3b --exp_name fever_K4_seed0

    # Ablation: no CVP (falls back to hard-mapping decomposition):
    python train.py --dataset scifact --shot_num 16 --use_cvp false \
                    --exp_name scifact_K16_noCVP

    # Zero-shot on SciFACT with Llama-3.1-8B:
    python train.py --dataset scifact --zero_shot true \
                    --backbone llama-3.1-8b --exp_name scifact_zs_llama
"""

import argparse
import os
import time

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import configs
from data_reader import CVPRDataModule
from model import CVPRFV
from utils import set_seeds


def _bool(s):
    return str(s).lower() in ('1', 'true', 'yes', 'y', 't')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='fever', choices=configs.dataset_names)
    p.add_argument('--shot_num', type=int, default=configs.SHOT_NUM)
    p.add_argument('--few_shot', type=_bool, default=True)
    p.add_argument('--zero_shot', type=_bool, default=False)
    p.add_argument('--use_cvp', type=_bool, default=True)
    p.add_argument('--cvp_total_per_dataset', type=int, default=configs.cvp_total_per_dataset)

    p.add_argument('--backbone', type=str, default=configs.backbone,
                   choices=list(configs.BACKBONES))
    p.add_argument('--lam_prior', type=float, default=configs.lam_prior,
                   help='λ — verifiability prior strength.')
    p.add_argument('--nei_floor_gamma', type=float, default=configs.nei_floor_gamma,
                   help='γ — small NEI floor keeping non-zero mass under verifiable claims.')
    p.add_argument('--lam_cvp', type=float, default=configs.lam_cvp,
                   help='λ_cvp — weight of CVP loss in the joint objective.')

    p.add_argument('--lr', type=float, default=configs.lr)
    p.add_argument('--num_steps', type=int, default=configs.num_steps)
    p.add_argument('--num_epochs', type=int, default=configs.num_epochs)
    p.add_argument('--warmup_ratio', type=float, default=configs.warmup_ratio)
    p.add_argument('--eval_step_interval', type=int, default=configs.eval_step_interval)
    p.add_argument('--patience', type=int, default=configs.patience)
    p.add_argument('--train_batch_size', type=int, default=configs.train_batch_size)
    p.add_argument('--eval_batch_size', type=int, default=configs.eval_batch_size)
    p.add_argument('--grad_accum_factor', type=int, default=configs.grad_accum_factor)
    p.add_argument('--grad_clip_norm', type=float, default=configs.grad_clip_norm)
    p.add_argument('--max_seq_len', type=int, default=250)
    p.add_argument('--precision', type=str, default=configs.compute_precision,
                   choices=['16', '32', 'bf16'])

    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exp_root', type=str, default=configs.exp_root)
    p.add_argument('--exp_name', type=str, default='cvpr_fv_run')
    p.add_argument('--save_model', type=_bool, default=True)
    p.add_argument('--load_weight', type=str, default='')
    p.add_argument('--eval_only', type=_bool, default=False)
    p.add_argument('--accelerator', type=str, default='auto')
    p.add_argument('--devices', type=str, default='auto')
    return p.parse_args()


def apply_args(args):
    configs.SHOT = args.few_shot
    configs.SHOT_NUM = args.shot_num
    configs.zero_shot = args.zero_shot
    configs.use_cvp = args.use_cvp
    configs.cvp_total_per_dataset = args.cvp_total_per_dataset

    configs.backbone = args.backbone
    configs.pretrained_model_path = configs.resolve_backbone_path(args.backbone)
    configs.lam_prior = args.lam_prior
    configs.nei_floor_gamma = args.nei_floor_gamma
    configs.lam_cvp = args.lam_cvp

    configs.lr = args.lr
    configs.num_steps = args.num_steps
    configs.num_epochs = args.num_epochs
    configs.warmup_ratio = args.warmup_ratio
    configs.eval_step_interval = args.eval_step_interval
    configs.patience = args.patience
    configs.train_batch_size = args.train_batch_size
    configs.eval_batch_size = args.eval_batch_size
    configs.grad_accum_factor = args.grad_accum_factor
    configs.grad_clip_norm = args.grad_clip_norm
    configs.compute_precision = args.precision
    configs.seed = args.seed
    configs.exp_root = args.exp_root
    configs.exp_name = args.exp_name
    configs.save_model = args.save_model
    configs.load_weight = args.load_weight

    configs.get_tokenizer().model_max_length = args.max_seq_len


def main():
    args = parse_args()
    apply_args(args)
    set_seeds(args.seed)

    print('=' * 72)
    for k, v in vars(args).items():
        print(f'{k:30s}: {v}')
    print('=' * 72)

    datamodule = CVPRDataModule(
        dataset_name=args.dataset,
        few_shot=args.few_shot,
        shot_num=args.shot_num,
        seed=args.seed,
        zero_shot=args.zero_shot,
        use_cvp=args.use_cvp,
        cvp_total_per_dataset=args.cvp_total_per_dataset,
    )

    model = CVPRFV(backbone=args.backbone)
    if args.load_weight:
        model.load_adapter_weights(args.load_weight)

    exp_dir = os.path.join(args.exp_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=exp_dir, name='log')

    val_check_interval = args.eval_step_interval * args.grad_accum_factor
    max_steps = args.num_steps if not args.eval_only else 0

    trainer = Trainer(
        max_steps=max_steps,
        min_steps=max_steps if max_steps > 0 else None,
        max_epochs=args.num_epochs,
        logger=logger,
        log_every_n_steps=max(1, args.eval_step_interval // 5),
        accumulate_grad_batches=args.grad_accum_factor,
        gradient_clip_val=args.grad_clip_norm,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        val_check_interval=val_check_interval,
        precision=int(args.precision) if args.precision in ('16', '32') else args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
    )

    if args.eval_only:
        trainer.validate(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule)
        best_ckpt = os.path.join(exp_dir, 'best.pt')
        if os.path.exists(best_ckpt):
            model.load_adapter_weights(best_ckpt)
        trainer.validate(model, datamodule=datamodule)


if __name__ == '__main__':
    t0 = time.time()
    main()
    dt = time.time() - t0
    print(f'Done. Elapsed: {dt / 60:.2f} min ({dt:.1f} s)')
    print(time.strftime('***** %Y-%m-%d %H:%M:%S *****', time.localtime()))
