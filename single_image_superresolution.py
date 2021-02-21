import cv2
import click
from pathlib import Path
from script.utils import show_image


@click.command()
@click.option("--input-image", "-i", "input_image_path_str", type=str, default="./image/001.jpg")
@click.option("--output-dir", "-o", "output_image_dir", type=str, default="./output")
def main(input_image_path_str, output_image_dir):
    output_image_dir_path = Path(output_image_dir)
    if not output_image_dir_path.exists():
        output_image_dir_path.mkdir()

    image = cv2.imread(input_image_path_str)
    show_image("raw", image)




if __name__ == "__main__":
    main()