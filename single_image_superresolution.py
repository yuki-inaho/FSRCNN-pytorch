# TODO: refactor
import cv2
import numpy as np
import click
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from models import FSRCNN
import PIL.Image as pil_image
from script.utils import show_image, write_image
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


def cv2pil(image: np.ndarray):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = pil_image.fromarray(new_image)
    return new_image


def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


@click.command()
@click.option("--input-image", "-i", "input_image_path_str", type=str, default="./image/001.jpg")
@click.option("--weights-file", "-w", "weights_file_path_str", type=str, default="./weights/fsrcnn_x2.pth")
@click.option("--scale", "-s", type=int, default=2)
@click.option("--output-dir", "-o", "output_image_dir", type=str, default="./output")
def main(input_image_path_str, weights_file_path_str, scale, output_image_dir):
    input_image_path = Path(input_image_path_str)
    output_image_dir_path = Path(output_image_dir)
    if not output_image_dir_path.exists():
        output_image_dir_path.mkdir()
    output_image_path_str = str(Path(output_image_dir, input_image_path.name.replace(".jpg", ".png")))

    image = cv2.imread(input_image_path_str)
    # show_image("raw", image)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FSRCNN(scale_factor=scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file_path_str, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image_height, image_width, _ = image.shape
    image_width_hr = image_width * scale
    image_height_hr = image_height * scale

    image_pil = cv2pil(image)
    image_bicubic_pil = image_pil.resize((image_width_hr, image_height_hr), resample=pil_image.BICUBIC)

    image_tensor, _ = preprocess(image_pil, device)
    _, ycbcr = preprocess(image_bicubic_pil, device)

    with torch.no_grad():
        preds = model(image_tensor).clamp(0.0, 1.0)

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    image_hr_pil = pil_image.fromarray(output)
    image_hr = pil2cv(image_hr_pil)
    #show_image("hr", image_hr)
    write_image(output_image_path_str, image_hr)


if __name__ == "__main__":
    main()