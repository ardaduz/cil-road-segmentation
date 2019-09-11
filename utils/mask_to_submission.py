#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
import pandas as pd

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+(?=\.png$)", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


if __name__ == '__main__':
    submission_filename = 'graph_cut_baseline.csv'
    image_filenames = []

    filenames = os.listdir("../baseline-graph-cut/prediction-1-graph-cut")

    for i in range(len(filenames)):
        f = filenames[i]

        path = os.path.join("/home/ardaduz/ETH/CIL/project/cil-road-segmentation/baseline-graph-cut/prediction-1-graph-cut", f)
        filenames[i] = path

    masks_to_submission(submission_filename, filenames)

    print("Validating submission file:", submission_filename)
    df = pd.read_csv(submission_filename)

    print('Shape of csv:', df.shape)
    assert df.shape == (135736, 2), "Invalid number of rows or columns in submission file!"
    assert df['id'].unique().size == 135736, "Column 'id' should contain 135736 unique values!"

    meanPred = df['prediction'].mean()
    print("Mean prediction: {:.3f}".format(meanPred))
