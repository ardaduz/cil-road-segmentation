import os
import cv2
import numpy as np
from graph_cut_segmentation import GraphCutSegmentation


# %%
def load_images_to_array(directory, cv2_flag, threshold=None):
    filenames = sorted(os.listdir(directory))

    images = []
    for filename in filenames:
        image = cv2.imread(directory + filename, cv2_flag)

        if threshold is not None:
            _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        image = image.tolist()
        images.append(image)

    images_array = np.asarray(images)
    return images_array, filenames


# %%
def scaledown_rgb_images(images, new_size):
    n_image = np.size(images, 0)
    scaled_images = np.empty((n_image, new_size, new_size, 3))
    for i in range(0, n_image):
        image = images[i, :, :, :].astype(np.uint8)
        new_image = cv2.resize(image, dsize=(new_size, new_size), interpolation=cv2.INTER_LANCZOS4)
        scaled_images[i, :, :, :] = new_image.astype(np.int32)
    return scaled_images


# %%
def scaleup_bw_images(images, new_size):
    n_image = np.size(images, 0)
    scaled_images = np.empty((n_image, new_size, new_size))
    for i in range(0, n_image):
        image = (images[i, :, :] * 255).astype(np.uint8)
        new_image = cv2.resize(image, dsize=(new_size, new_size), interpolation=cv2.INTER_LANCZOS4)
        _, scaled_images[i, :, :] = cv2.threshold(new_image, 127, 255, cv2.THRESH_BINARY)

    return (scaled_images / 255.0)


# %%
print("Loading images...")
images, image_filenames = load_images_to_array("../competition-data/training/images/", cv2.IMREAD_COLOR, threshold=None)
print("Loading ground-truths...")
gts, gt_filenames = load_images_to_array("../competition-data/training/groundtruth/", cv2.IMREAD_GRAYSCALE, threshold=127)

histogram_resolution = 256

# parameters for prediction
unary_lambda_value = 1e-20
pairwise_sigma_value = 5
window_size = 16

X_train = images
y_train = gts
X_test, test_filenames = load_images_to_array("../competition-data/test/", cv2.IMREAD_COLOR, threshold=None)

print("Training...")
GCSeg = GraphCutSegmentation(histogram_resolution, X_train, y_train)
GCSeg.train()

print("Predicting...")
X_test = scaledown_rgb_images(X_test, 304)
predicted_labels = GCSeg.predict(X_test, unary_lambda_value, pairwise_sigma_value, window_size, show_prediction=True)
predicted_labels = scaleup_bw_images(predicted_labels, 608)

n_predictions = np.size(predicted_labels, 0)
for i in range(0, n_predictions):
    prediction = (predicted_labels[i, :, :] * 255).astype(np.uint8)
    filename = test_filenames[i]
    cv2.imwrite("results/" + filename, prediction)
# %%
