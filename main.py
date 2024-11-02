import argparse

import torch
from torch.utils.data import random_split

from src.dataset import ScanImageDataset, ScanImageTestDataset
from src.model import *
from src.trainer import Trainer
from src.transform import get_test_transform, get_transform
from src.utils import set_seed, get_parameter_nums


def main(args):
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    transform = get_transform()
    test_transform = get_test_transform()

    model = Restormer()

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        print(f"=> from resuming checkpoint '{args.resume}' ")

    dataset = ScanImageDataset(
        noisy_image_dir_path=args.noisy_image_dir, clean_image_dir_path=args.clean_image_dir, transform=transform
    )

    dataset_size = len(dataset)
    train_size = int(dataset_size * args.split_ratio)
    validation_size = dataset_size - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    test_dataset = ScanImageTestDataset(noisy_image_paths=args.test_image_dir, transform=test_transform)

    trainer = Trainer(args, model, train_dataset, validation_dataset, test_dataset)
    print(f"Num parameters: {get_parameter_nums(model)}")

    if args.do_train:
        trainer.train()
    if args.do_eval:
        trainer.evaluate()
    if args.do_inference:
        trainer.inference(args.output_path, args.output_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=200, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--split_ratio", default=0.95, type=float)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=1, type=int)
    parser.add_argument("--noisy_image_dir", default="C:\\Users\\rlaal\\OneDrive\\Desktop\\Training\\noisy", type=str)
    parser.add_argument("--clean_image_dir", default="C:\\Users\\rlaal\\OneDrive\\Desktop\\Training\\clean", type=str)
    parser.add_argument("--test_image_dir", default="C:\\Users\\rlaal\\OneDrive\\Desktop\\Validation\\noisy", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--do_wandb", default=True, type=bool)
    parser.add_argument("--do_eval", default=True, type=bool)
    parser.add_argument("--do_inference", default=True, type=bool)
    parser.add_argument("--dataset_path", default="dataset", type=str)
    parser.add_argument("--save_logs", default=True, type=bool)
    parser.add_argument("--save_frequency", default=5, type=int)
    parser.add_argument("--checkpoint_path", default="model_output/", type=str)
    parser.add_argument("--submission_path", default="submission/", type=str)
    parser.add_argument("--output_path", default="output", type=str)
    parser.add_argument("--output_file_name", default="result.csv", type=str)
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "--val_frequency", default=1, type=int, help="How often to run evaluation with validation data."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )

    args = parser.parse_args()

    main(args)
