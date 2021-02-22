import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from script.models import FSRCNN
from script.utils import convert_ycbcr_to_rgb, preprocess, cv2pil, pil2cv


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
