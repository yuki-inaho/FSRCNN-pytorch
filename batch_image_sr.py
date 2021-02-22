
import cv2
import numpy as np
import click
from pathlib import Path
from tqdm import tqdm
from script.superresolution import SuperResolutionFSRCNN
from script.utils import show_image, write_image

import pdb


SCRIPT_DIR = str(Path(__file__).parent)


@click.command()
@click.option("--input-image-dir", "-i", default=f"{SCRIPT_DIR}/images_raw")
@click.option("--weights-file", "-w", "weights_file_path_str", type=str, default="./weights/fsrcnn_x2.pth")
@click.option("--scale", "-s", type=int, default=2)
@click.option("--output-image-dir", "-o", default=f"{SCRIPT_DIR}/images")
def main(input_image_dir, weights_file_path_str, scale, output_image_dir):
    output_image_dir_path = Path(output_image_dir)
    if not output_image_dir_path.exists():
        output_image_dir_path.mkdir()

    fsrcnn = SuperResolutionFSRCNN(weights_file_path_str=weights_file_path_str, scale=scale)

    extf = [".jpg", ".png"]
    image_pathes = [path for path in Path(input_image_dir).glob("*") if path.suffix in extf]
    image_path_list = [str(image_path) for image_path in image_pathes]
    for image_path in tqdm(image_path_list):
        image = cv2.imread(image_path)
        image_hr = fsrcnn(image)
        base_name = Path(image_path).name
        cv2.imwrite(str(Path(output_image_dir, base_name)), image_hr)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()