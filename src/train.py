import argparse

import lightning as L
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split

from preprocessing import DitchDataset
from model import LightningDitchNet

from utils.config import TrainConfig, ModelConfig
from utils.cli_args.training_args import add_training_args
from utils.cli_args.model_args import add_model_args, add_scheduler_args

L.seed_everything(14, workers=True)


class Train:
    def __init__(self, config: TrainConfig):
        self.config = config

        # Initialize the segmentation model
        self.model = LightningDitchNet(config.model_config)

        # Split dataset into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = self._construct_train_val_sets(config.feature_dir,
                                                                                            config.label_dir)

        # Define augmentation and preprocessing pipelines
        self.train_transform, self.val_transform = self._construct_transforms()

        # Create PyTorch DataLoaders for training and validation
        self.train_dataloader, self.validation_dataloader = self._construct_dataloaders(config.batch_size,
                                                                                        config.num_workers)

        # Initialize logger and callbacks for model tracking and checkpointing
        self.logger = CSVLogger(save_dir=Path.cwd() / "lightning_logs", name="train_logs")
        self.callbacks = self._set_callbacks()

    def _construct_train_val_sets(self, feature_dir, label_dir):
        # Resolve and sort all feature and label paths
        X = sorted(Path(feature_dir).resolve().iterdir())
        y = sorted(Path(label_dir).resolve().iterdir())

        if len(X) != len(y):
            raise ValueError("Feature and label directories must contain the same number of files.")

        # Split into training and validation sets
        return train_test_split(X, y, test_size=self.config.val_size, random_state=14)

    @staticmethod
    def _construct_transforms():
        train_transform = A.Compose([A.HorizontalFlip(p=0.5),
                                     A.VerticalFlip(p=0.5),
                                     A.RandomRotate90(p=0.5),
                                     A.Transpose(p=0.2),
                                     ToTensorV2()],
                                    additional_targets={"label": "mask"})

        val_transform = A.Compose([ToTensorV2()],
                                  additional_targets={"label": "mask"})

        return train_transform, val_transform

    def _construct_dataloaders(self, batch_size, num_workers):
        # Dataset and DataLoader construction
        training_dataset = DitchDataset(X=self.X_train, y=self.y_train, transform=self.train_transform)
        validation_dataset = DitchDataset(X=self.X_val, y=self.y_val, transform=self.val_transform)

        training_dataloader = DataLoader(training_dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         persistent_workers=(num_workers > 0),
                                         shuffle=True,
                                         pin_memory=True)

        validation_dataloader = DataLoader(validation_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           persistent_workers=(num_workers > 0),
                                           pin_memory=True)

        return training_dataloader, validation_dataloader

    def _set_callbacks(self):
        # Save top-performing checkpoints and enable early stopping
        checkpoint = ModelCheckpoint(save_weights_only=self.config.save_weights_only,
                                     save_top_k=self.config.save_top_k,
                                     monitor=self.config.checkpoint_monitor,
                                     mode=self.config.checkpoint_mode)

        if not self.config.use_early_stop:
            return [checkpoint]

        early_stop = EarlyStopping(patience=self.config.early_stop_patience,
                                   monitor=self.config.early_stop_monitor,
                                   mode=self.config.early_stop_mode)

        return [checkpoint, early_stop]

    @staticmethod
    def _is_weights_only_checkpoint(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "optimizer_states" not in ckpt:
            print(f'[WARNING] Checkpoint "{ckpt_path}" is weights-only '
                  f'(no optimizer or scheduler state). Fine-tuning mode enabled.')

            return True

        return False

    def run(self):
        # Configure the Lightning trainer and launch training
        trainer = L.Trainer(max_epochs=self.config.max_epochs,
                            accelerator="auto",
                            devices="auto",
                            strategy="auto",
                            callbacks=self.callbacks,
                            logger=self.logger,
                            precision=self.config.compute_precision)

        if self.config.ckpt_path:
            if self._is_weights_only_checkpoint(self.config.ckpt_path):
                model = LightningDitchNet(self.config.model_config)
                state = torch.load(self.config.ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state["state_dict"])

                trainer.fit(model,
                            train_dataloaders=self.train_dataloader,
                            val_dataloaders=self.validation_dataloader)

                return

        trainer.fit(self.model,
                    ckpt_path=self.config.ckpt_path,
                    train_dataloaders=self.train_dataloader,
                    val_dataloaders=self.validation_dataloader)


class Main:
    def __init__(self):
        args = self._parse_arguments()
        model_config = ModelConfig(encoder_name=args.encoder_name,
                                   pos_weight=args.pos_weight,
                                   lr=args.learning_rate,
                                   in_channels=args.in_channels,
                                   weight_decay=args.weight_decay,

                                   use_scheduler=args.use_scheduler,
                                   scheduler_monitor=args.scheduler_monitor,
                                   scheduler_mode=args.scheduler_mode,
                                   scheduler_factor=args.scheduler_factor,
                                   scheduler_patience=args.scheduler_patience,
                                   scheduler_cooldown=args.scheduler_cooldown,
                                   scheduler_min_lr=args.scheduler_min_lr,
                                   scheduler_threshold=args.scheduler_threshold,
                                   scheduler_threshold_mode=args.scheduler_threshold_mode)

        train_config = TrainConfig(args.feature_dir,
                                   args.label_dir,
                                   args.max_epochs,
                                   ckpt_path=args.ckpt_path,
                                   val_size=args.val_size,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   compute_precision=args.compute_precision,

                                   save_weights_only=args.save_weights_only,
                                   save_top_k=args.save_top_k,
                                   checkpoint_monitor=args.checkpoint_monitor,
                                   checkpoint_mode=args.checkpoint_mode,

                                   use_early_stop=args.use_early_stop,
                                   early_stop_patience=args.early_stop_patience,
                                   early_stop_monitor=args.early_stop_monitor,
                                   early_stop_mode=args.early_stop_mode,

                                   model_config=model_config)

        self.trainer = Train(train_config)
        self.run()

    @staticmethod
    def _parse_arguments():
        parser = argparse.ArgumentParser(description="Train the LightningDitchNet segmentation model.")

        add_training_args(parser)
        add_model_args(parser)
        add_scheduler_args(parser)

        return parser.parse_args()

    def run(self):
        self.trainer.run()


if __name__ == "__main__":
    Main()
