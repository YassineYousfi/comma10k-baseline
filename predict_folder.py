"""
Runs a model on a single node across multiple gpus.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from pathlib import Path
from argparse import ArgumentParser
from LitModel import *
from retriever import *
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer, seed_everything

seed_everything(1994)

def main(args):
    """ Main inference routine specific for this project. """
    
    model = LitModel.load_from_checkpoint(args.resume_from_checkpoint, data_path=args.data_path)
    
    trainer = Trainer(logger=None,
                     gpus=args.gpus,
                     precision=16,
                     amp_backend='native',
                     row_log_interval=100,
                     log_save_interval=100,
                     distributed_backend='ddp',
                     benchmark=True,
                     resume_from_checkpoint=args.resume_from_checkpoint)

    folder_to_predict = Path(args.folder_to_predict)
    assert (folder_to_predict/'imgs').is_dir(), 'Images not found'
    
    images_list = list(folder_to_predict.glob('*/*.png'))
    images_list = [os.path.basename(x) for x in images_list]
    (folder_to_predict/'predicted_masks').mkdir(exist_ok=True)

    test_dataset = InferenceRetriever(
            data_path=folder_to_predict,
            image_names=images_list,
            preprocess_fn=model.preprocess_fn,
            transforms=get_valid_transforms(model.height, model.width))

    test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

    model.folder_to_predict = folder_to_predict # Passing this little attribute
    
    trainer.test(model, test_dataloader)


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    
    parent_parser = ArgumentParser(add_help=False)

    parser = LitModel.add_model_specific_args(parent_parser)
    
    parser.add_argument('--resume-from-checkpoint',
                         default=None,
                         type=str,
                         metavar='RFC',
                         help='path to checkpoint')
    parser.add_argument('--folder-to-predict',
                         default=None,
                         type=str,
                         help='path to folder to predict must contain a imgs/ subfolder')
    
    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    run_cli()