import os
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', help="The path to save CAD templates")
parser.add_argument('--target', help="The pipe target name")
parser.add_argument('--epochs', type=int, help="The number of epochs")
parser.add_argument('--batch_size', type=int, help="The batch size")
parser.add_argument('--save_period', type=int, help="The save period")
args = parser.parse_args()

os.makedirs(f"{args.output_dir}/segmentation/train", exist_ok=True)

model = YOLO(model="/home/th/ws/research/PipeIsoGen/SAM-6D/Instance_Segmentation_Model/FastSAM-s.pt")

model.train(
    data=f"/home/th/ws/research/PipeIsoGen/data/train/{args.target}/data.yaml",
    task='segment',
    epochs=args.epochs,
    augment=False,
    batch=args.batch_size,
    overlap_mask=False,
    save=True,
    save_period=args.save_period,
    project=f"{args.output_dir}/segmentation/train",
    name=args.target,
    val=False,
)
