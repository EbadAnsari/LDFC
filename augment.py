import argparse
import csv
import os

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from scipy.ndimage import rotate


def augment(sample, bboxes, do_flip=False, do_rotate=False):

    if do_rotate and bboxes.size > 0:

        angle1 = np.random.rand() * 180
        size = np.array(sample.shape[2:4]).astype('float')

        sample = rotate(sample, angle1, axes=(2, 3), reshape=False)

        for box in bboxes[0]:

            diameter1 = box[3] - box[2]

            center_y = size[1] - ((box[3] - box[2]) // 2 + box[2])
            center_x = (box[5] - box[4]) // 2 + box[4]

            rotate_x = (
                (center_x - size[0] // 2) * np.cos(angle1 / 180 * np.pi)
                - (center_y - size[0] // 2) * np.sin(angle1 / 180 * np.pi)
            ) + size[0] / 2

            rotate_y = size[1] - (
                (center_x - size[1] // 2) * np.sin(angle1 / 180 * np.pi)
                + (center_y - size[1] // 2) * np.cos(angle1 / 180 * np.pi)
                + size[1] / 2
            )

            xmin = rotate_x - diameter1 // 2
            xmax = rotate_x + diameter1 // 2
            ymin = rotate_y - diameter1 // 2
            ymax = rotate_y + diameter1 // 2

            zmin = box[0]
            zmax = box[1]

            box[0] = xmin
            box[1] = xmax
            box[2] = ymin
            box[3] = ymax
            box[4] = zmin
            box[5] = zmax

    return sample, bboxes


def main(args):

    data_path = args.data_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    annos_all = pd.read_csv(args.csv_dir)

    for filename in os.listdir(data_path):

        folder_path = os.path.join(data_path, filename)

        if not os.path.isdir(folder_path):
            continue

        lines = sorted(os.listdir(folder_path))

        slice_files = [os.path.join(folder_path, s) for s in lines]

        slices = [imageio.imread(s) for s in slice_files]

        imgs = np.array(slices)
        imgs = imgs[np.newaxis, ...]

        annos = annos_all[annos_all['index'] == int(filename)]

        temp_annos = []

        if len(annos) > 0:

            for index in range(len(annos)):
                anno = annos.iloc[index]

                temp_annos.append([
                    anno['z_min'],
                    anno['z_max'],
                    anno['y_min'],
                    anno['y_max'],
                    anno['x_min'],
                    anno['x_max']
                ])

        if len(temp_annos) == 0:
            labels = np.empty((1,0,6))
        else:
            labels = np.array([temp_annos])

        print(f"Start augmenting case {filename}")

        sample, bboxes = augment(
            sample=imgs,
            bboxes=labels,
            do_rotate=True
        )

        save_folder = os.path.join(output_path, filename)
        os.makedirs(save_folder, exist_ok=True)

        idx = 0

        for imgname in lines:

            imageio.imwrite(
                os.path.join(save_folder, imgname),
                sample[0, idx, :, :]
            )

            idx += 1

        if bboxes.size == 0:
            print("No bounding boxes for this case")
            continue

        center_list = []

        for i in range(bboxes.shape[1]):

            x_center = (bboxes[0, i, 0] + bboxes[0, i, 1]) // 2
            y_center = (bboxes[0, i, 2] + bboxes[0, i, 3]) // 2
            z_center = (bboxes[0, i, 4] + bboxes[0, i, 5]) // 2

            diameter = bboxes[0, i, 1] - bboxes[0, i, 0]

            center_list.append([
                x_center,
                y_center,
                z_center,
                diameter
            ])

        center_list = np.array(center_list)

        bboxes_new = bboxes[0]

        if center_list.size > 0:
            bboxes_new = np.concatenate((bboxes_new, center_list), axis=1)

        for row in bboxes_new:

            with open(args.csv_save, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

        print(f"Finished case {filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--csv-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--csv-save",
        type=str,
        required=True
    )

    opt = parser.parse_args()

    main(opt)

#     python augment.py ^
# --data-path "classification/data/8422229/BMP_3D/BMP_3D" ^
# --output-path "classification/data/augment_output" ^
# --csv-dir "classification/data/8422229/BMP_2D/BMP_2D/Annotations/all_anno_3D.csv" ^
# --csv-save "classification/data/augment_output/augment_rotate.csv"