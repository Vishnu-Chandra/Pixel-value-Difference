import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_image, rgb_to_ycbcr, divide_into_blocks
from features import extract_features


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COVER_FOLDER = os.path.join(PROJECT_ROOT, "data", "cover")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "features.csv")

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)


def create_dataset():

    all_features = []

    image_files = sorted([
        file
        for file in os.listdir(COVER_FOLDER)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ])

    print("Images found:", len(image_files))
    print()

    if len(image_files) == 0:
        print("No images found.")
        return

    for image_name in image_files:

        image_path = os.path.join(
            COVER_FOLDER,
            image_name
        )

        print("Processing:", image_name)

        image = load_image(image_path)

        print("  Image shape:", image.shape)

        ycbcr = rgb_to_ycbcr(image)

        Y = ycbcr[:, :, 0]

        blocks = divide_into_blocks(
            Y,
            block_size=8
        )

        print("  Number of blocks:", len(blocks))

        for block_id, block in enumerate(blocks):

            features = extract_features(block)

            all_features.append([
                image_name,
                block_id,

                # Feature 1
                features[0],

                # Feature 2
                features[1],

                # Feature 3
                features[2],

                # Feature 4
                features[3],

                # Feature 5
                features[4],

                # Feature 6
                features[5],

                # Feature 7
                features[6]
            ])

        print(
            "  Features extracted:",
            len(blocks)
        )

        print()

    columns = [
        "image_name",
        "block_id",

        "variance",
        "entropy",
        "edge_density",
        "gradient_magnitude",

        "mean_pixel_difference",
        "max_pixel_difference",
        "std_pixel_difference"
    ]

    df = pd.DataFrame(
        all_features,
        columns=columns
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("----------------------------------------")
    print("Dataset creation completed!")
    print("----------------------------------------")

    print("Total images:", len(image_files))
    print("Total blocks:", len(df))
    print("Features per block: 7")
    print()
    print("Dataset shape:", df.shape)
    print()
    print("Columns:")
    print(list(df.columns))
    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    create_dataset()