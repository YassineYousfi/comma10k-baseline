import numpy as np
import argparse
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
import random
from retriever import *
import segmentation_models_pytorch as smp


class LitModel(pl.LightningModule):
    """Transfer Learning"""

    def __init__(
        self,
        data_path: Union[str, Path],
        backbone: str = "efficientnet-b0",
        augmentation_level: str = "light",
        batch_size: int = 32,
        lr: float = 1e-4,
        eps: float = 1e-7,
        height: int = 14 * 32,
        width: int = 18 * 32,
        num_workers: int = 6,
        epochs: int = 50,
        gpus: int = 1,
        weight_decay: float = 1e-3,
        class_values: List[int] = [41, 76, 90, 124, 161, 0],  # 0 added for padding
        **kwargs
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.epochs = epochs
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.height = height
        self.width = width
        self.num_workers = num_workers
        self.gpus = gpus
        self.weight_decay = weight_decay
        self.eps = eps
        self.class_values = class_values
        self.augmentation_level = augmentation_level

        self.save_hyperparameters()

        self.preprocess_fn = smp.encoders.get_preprocessing_fn(
            self.backbone, pretrained="imagenet"
        )

        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. net:

        self.net = smp.Unet(
            self.backbone,
            classes=len(self.class_values),
            activation=None,
            encoder_weights="imagenet",
        )

        # 2. Loss:
        self.loss_func = lambda x, y: torch.nn.CrossEntropyLoss()(
            x, torch.argmax(y, axis=1)
        )

    def forward(self, x):
        """Forward pass. Returns logits."""

        x = self.net(x)

        return x

    def loss(self, logits, labels):
        """Use the loss_func"""
        return self.loss_func(logits, labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)

        # 2. Compute loss & accuracy:
        train_loss = self.loss(y_logits, y)
        self.log("train_loss", train_loss)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)

        # 2. Compute loss & accuracy:
        val_loss = self.loss(y_logits, y)
        self.log("val_loss", val_loss)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam
        optimizer_kwargs = {"eps": self.eps}

        optimizer = optimizer(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **optimizer_kwargs
        )

        scheduler_kwargs = {
            "T_max": self.epochs
            * len(self.train_dataset)
            // self.gpus
            // self.batch_size,
            "eta_min": self.lr / 50,
        }

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        interval = "step"
        scheduler = scheduler(optimizer, **scheduler_kwargs)

        return [optimizer], [
            {"scheduler": scheduler, "interval": interval, "name": "lr"}
        ]

    def prepare_data(self):
        """Data download is not part of this script
        Get the data from https://github.com/commaai/comma10k
        """
        assert (self.data_path / "imgs").is_dir(), "Images not found"
        assert (self.data_path / "masks").is_dir(), "Masks not found"
        assert (
            self.data_path / "files_trainable"
        ).exists(), "Files trainable file not found"

        print("data ready")

    def setup(self, stage: str):
        image_names = np.loadtxt(
            self.data_path / "files_trainable", dtype="str"
        ).tolist()

        random.shuffle(image_names)

        self.train_dataset = TrainRetriever(
            data_path=self.data_path,
            image_names=[
                x.split("masks/")[-1] for x in image_names if not x.endswith("9.png")
            ],
            preprocess_fn=self.preprocess_fn,
            transforms=get_train_transforms(
                self.height, self.width, self.augmentation_level
            ),
            class_values=self.class_values,
        )

        self.valid_dataset = TrainRetriever(
            data_path=self.data_path,
            image_names=[
                x.split("masks/")[-1] for x in image_names if x.endswith("9.png")
            ],
            preprocess_fn=self.preprocess_fn,
            transforms=get_valid_transforms(self.height, self.width),
            class_values=self.class_values,
        )

    def __dataloader(self, train):
        """Train/validation loaders."""

        _dataset = self.train_dataset if train else self.valid_dataset
        loader = DataLoader(
            dataset=_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if train else False,
        )

        return loader

    def train_dataloader(self):
        log.info("Training data loaded.")
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info("Validation data loaded.")
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            "--backbone",
            default="efficientnet-b0",
            type=str,
            metavar="BK",
            help="Name as in segmentation_models_pytorch",
        )
        parser.add_argument(
            "--augmentation-level",
            default="light",
            type=str,
            help="Training augmentation level c.f. retiriever",
        )
        parser.add_argument(
            "--data-path",
            default="/home/gregor/Desktop/segnet/comma10k",  # TODO: change to your path
            type=str,
            metavar="dp",
            help="data_path",
        )
        parser.add_argument(
            "--epochs", default=30, type=int, metavar="N", help="total number of epochs"
        )
        parser.add_argument(
            "--batch-size",
            default=32,
            type=int,
            metavar="B",
            help="batch size",
            dest="batch_size",
        )
        parser.add_argument("--gpus", type=int, default=1, help="number of gpus to use")
        parser.add_argument(
            "--lr",
            "--learning-rate",
            default=1e-4,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--eps",
            default=1e-7,
            type=float,
            help="eps for adaptive optimizers",
            dest="eps",
        )
        parser.add_argument("--height", default=14 * 32, type=int, help="image height")
        parser.add_argument("--width", default=18 * 32, type=int, help="image width")
        parser.add_argument(
            "--num-workers",
            default=6,
            type=int,
            metavar="W",
            help="number of CPU workers",
            dest="num_workers",
        )
        parser.add_argument(
            "--weight-decay",
            default=1e-3,
            type=float,
            metavar="wd",
            help="Optimizer weight decay",
        )

        return parser
