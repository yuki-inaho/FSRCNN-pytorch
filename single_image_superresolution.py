# TODO: refactor
import cv2
import click
from pathlib import Path
from script.superresolution import SuperResolutionFSRCNN
from script.utils import show_image, write_image


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