# TODO: refactor
import cv2
import numpy as np
import click
from pathlib import Path
import time

import torch
import torch.backends.cudnn as cudnn
import PIL.Image as pil_image
from script.models import FSRCNN
from script.utils import convert_ycbcr_to_rgb, preprocess, show_image, write_image, cv2pil, pil2cv


class SuperResolutionFSRCNN:
    def __init__(self, weights_file_path_str: str, scale: int):
        self._scale = scale
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = FSRCNN(scale_factor=scale).to(self._device)
        state_dict = self._model.state_dict()
        for n, p in torch.load(weights_file_path_str, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        cudnn.benchmark = True
        self._model.eval()

    def __call__(self, image_cv: np.ndarray):
        self.set_image(image_cv)
        return self.upsample()

    def set_image(self, image: np.ndarray):
        image_height, image_width, _ = image.shape
        image_width_hr = image_width * self._scale
        image_height_hr = image_height * self._scale

        image_pil = cv2pil(image)
        image_bicubic_pil = image_pil.resize((image_width_hr, image_height_hr), resample=pil_image.BICUBIC)

        self._image_tensor, _ = preprocess(image_pil, self._device)
        _, self._ycbcr = preprocess(image_bicubic_pil, self._device)

    def upsample(self):
        with torch.no_grad():
            preds = self._model(self._image_tensor).clamp(0.0, 1.0)
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        output = np.array([preds, self._ycbcr[..., 1], self._ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)

        image_hr_pil = pil_image.fromarray(output)
        image_hr = pil2cv(image_hr_pil)
        return image_hr


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

    fsrcnn = SuperResolutionFSRCNN(weights_file_path_str=weights_file_path_str, scale=scale)
    image_hr = fsrcnn(image)
    # show_image("raw", image)
    # show_image("hr", image_hr)
    write_image(output_image_path_str, image_hr)
    print("Done")


if __name__ == "__main__":
    main()