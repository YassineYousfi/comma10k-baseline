"""
Runs a model on a single node across multiple gpus.
"""
import wandb
from argparse import ArgumentParser
from LitModel import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

seed_everything(1994)


def main(args):
    """Main training routine specific for this project."""

    if args.seed_from_checkpoint:
        print("model seeded")
        model = LitModel.load_from_checkpoint(args.seed_from_checkpoint, **vars(args))
    else:
        model = LitModel(**vars(args))

    wandb_logger = WandbLogger(project="comma10k-baseline", name=args.version)
    wandb_logger.log_hyperparams(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath="/home/gregor/logs",  # TODO change before merge
        filename="{epoch:02d}_{val_loss:.4f}",
        save_top_k=10,
        monitor="val_loss",
        mode="min",
    )
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=args.gpus,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        precision=16,
        strategy="ddp_find_unused_parameters_true",
        benchmark=True,
        sync_batchnorm=True,
    )

    trainer.logger.log_hyperparams(model.hparams)

    trainer.fit(model, ckpt_path=args.resume_from_checkpoint)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)

    parser = LitModel.add_model_specific_args(parent_parser)

    parser.add_argument(
        "--version",
        default=None,
        type=str,
        metavar="V",
        help="version or id of the net",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        type=str,
        metavar="RFC",
        help="path to checkpoint",
    )
    parser.add_argument(
        "--seed-from-checkpoint",
        default=None,
        type=str,
        metavar="SFC",
        help="path to checkpoint seed",
    )

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    wandb.login()
    torch.set_float32_matmul_precision("high")
    run_cli()
