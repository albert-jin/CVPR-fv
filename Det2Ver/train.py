"""Command-line entry point for Det2Ver.

Example
-------

Few-shot (Det2Ver(20) on FEVER, K=4):

    python train.py \\
        --dataset fever --shot_num 4 --seed 0 \\
        --use_rumor_detection --rd_total_per_dataset 20 \\
        --exp_name fever_K4_rd20_seed0 --num_steps 1500

Zero-shot (Det2Ver(50) on SciFACT):

    python train.py \\
        --dataset scifact --zero_shot --rd_total_per_dataset 50 \\
        --exp_name scifact_zs_rd50_seed0 --num_steps 1500

Warm-starting from a T-Few / ProToCo (IA)^3 checkpoint::

    python train.py --dataset fever --shot_num 4 \\
        --load_weight pretrained_checkpoints/t03b_ia3_finish.pt
"""

import argparse
import os
import time

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import configs
from data_reader import V2DDataModule
from model import Det2Ver
from utils import set_seeds


def _bool(s: str) -> bool:
    return str(s).lower() in ('1', 'true', 'yes', 'y', 't')


def parse_args():
    p = argparse.ArgumentParser(description='Det2Ver: cross-task rumor detection to fact verification.')

    p.add_argument('--dataset', type=str, default='fever',
                   choices=configs.dataset_names,
                   help='Fact-verification target dataset.')
    p.add_argument('--shot_num', type=int, default=configs.SHOT_NUM,
                   help='K in K-shot (per class).')
    p.add_argument('--few_shot', type=_bool, default=True)
    p.add_argument('--zero_shot', type=_bool, default=False,
                   help='If True, no FV example is used (Table IV).')

    p.add_argument('--use_rumor_detection', type=_bool, default=True,
                   help='If False, disables cross-task RD supervision (Det2Ver(0)).')
    p.add_argument('--rd_total_per_dataset', type=int, default=20,
                   help='Number of RD instances per RD dataset (paper: 20, 50, 100).')

    p.add_argument('--lr', type=float, default=configs.lr)
    p.add_argument('--num_steps', type=int, default=configs.num_steps)
    p.add_argument('--warmup_ratio', type=float, default=configs.warmup_ratio)
    p.add_argument('--eval_step_interval', type=int, default=configs.eval_step_interval)
    p.add_argument('--patience', type=int, default=configs.patience)
    p.add_argument('--train_batch_size', type=int, default=configs.train_batch_size)
    p.add_argument('--eval_batch_size', type=int, default=configs.eval_batch_size)
    p.add_argument('--grad_accum_factor', type=int, default=configs.grad_accum_factor)
    p.add_argument('--grad_clip_norm', type=float, default=configs.grad_clip_norm)
    p.add_argument('--max_seq_len', type=int, default=250)

    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exp_root', type=str, default=configs.exp_root)
    p.add_argument('--exp_name', type=str, default='det2ver_run')
    p.add_argument('--save_model', type=_bool, default=True)
    p.add_argument('--load_weight', type=str, default='',
                   help='Optional path to a warm-start (IA)^3 / LoRA checkpoint.')

    p.add_argument('--eval_only', type=_bool, default=False,
                   help='Skip training and just run evaluation on the current adapter weights.')
    p.add_argument('--precision', type=str, default='32', choices=['16', '32', 'bf16'])
    p.add_argument('--accelerator', type=str, default='auto')
    p.add_argument('--devices', type=str, default='auto')
    return p.parse_args()


def apply_args_to_configs(args):
    """Push CLI overrides back into the ``configs`` module so that
    downstream modules that import from it see the updated values."""
    configs.SHOT_NUM = args.shot_num
    configs.SHOT = args.few_shot
    configs.zero_shot = args.zero_shot
    configs.use_rumor_detection = args.use_rumor_detection
    configs.rd_total_per_dataset = args.rd_total_per_dataset
    configs.lr = args.lr
    configs.num_steps = args.num_steps
    configs.warmup_ratio = args.warmup_ratio
    configs.eval_step_interval = args.eval_step_interval
    configs.patience = args.patience
    configs.train_batch_size = args.train_batch_size
    configs.eval_batch_size = args.eval_batch_size
    configs.grad_accum_factor = args.grad_accum_factor
    configs.grad_clip_norm = args.grad_clip_norm
    configs.seed = args.seed
    configs.exp_root = args.exp_root
    configs.exp_name = args.exp_name
    configs.save_model = args.save_model
    configs.load_weight = args.load_weight
    # Ensure the tokenizer has the requested max length.
    configs.get_tokenizer().model_max_length = args.max_seq_len


def main():
    args = parse_args()
    apply_args_to_configs(args)
    set_seeds(args.seed)

    print('=' * 72)
    for k, v in vars(args).items():
        print(f'{k:30s}: {v}')
    print('=' * 72)

    datamodule = V2DDataModule(
        dataset_name=args.dataset,
        few_shot=args.few_shot,
        shot_num=args.shot_num,
        seed=args.seed,
        zero_shot=args.zero_shot,
        use_rumor_detection=args.use_rumor_detection,
        rd_total_per_dataset=args.rd_total_per_dataset,
    )

    model = Det2Ver()
    if args.load_weight:
        model.load_adapter_weights(args.load_weight)

    exp_dir = os.path.join(args.exp_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=exp_dir, name='log')

    # ``check_val_every_n_epoch`` in PL 1.x pairs with the tiny few-shot
    # datasets: one epoch = one pass over K-shot. We rely on
    # ``val_check_interval`` to trigger evaluation every N optimizer steps.
    val_check_interval = args.eval_step_interval * args.grad_accum_factor

    trainer_kwargs = dict(
        max_steps=args.num_steps if not args.eval_only else 0,
        min_steps=args.num_steps if not args.eval_only else None,
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

    trainer = Trainer(**trainer_kwargs)

    if args.eval_only:
        trainer.validate(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule)
        # Final evaluation with the *best* adapter weights if we saved one.
        best_ckpt = os.path.join(exp_dir, 'best.pt')
        if os.path.exists(best_ckpt):
            model.load_adapter_weights(best_ckpt)
        trainer.validate(model, datamodule=datamodule)


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f'Done. Elapsed: {elapsed / 60.0:.2f} min ({elapsed:.1f} s)')
    print(time.strftime('***** %Y-%m-%d %H:%M:%S *****', time.localtime()))
