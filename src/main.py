from __future__ import annotations

import argparse

from utils.io_handler import IOHandler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2D image to 3D object pipeline entrypoint")
    parser.add_argument("--input-dir", default="data/inputs", help="Input images directory")
    parser.add_argument("--output-dir", default="data/outputs", help="Output directory")
    parser.add_argument("--max-images", type=int, default=None, help="Optional maximum number of images to load")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load and validate input images without running reconstruction",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    io_handler = IOHandler(input_dir=args.input_dir, output_dir=args.output_dir)
    images, image_names = io_handler.load_images(max_images=args.max_images)
    io_handler.ensure_output_dir()

    print(f"Loaded {len(images)} images")
    print(f"Input names: {', '.join(image_names)}")

    if args.dry_run:
        print("Dry run completed. Images and folders are ready.")
        return

    print("Feature extraction -> Geometry reconstruction -> Surface generation")
    print("Pipeline skeleton is ready. Implement module engines to produce final 3D object.")


if __name__ == "__main__":
    main()