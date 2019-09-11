import numpy as np
from graph_cut import GraphCut
from scipy import ndimage
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import cv2

class InputPair:
    def __init__(self, image, gt):
        self.image = image
        self.gt = gt
        self.fg_locations = None
        self.bg_locations = None
        self.fg_pixels = None
        self.bg_pixels = None

    def __set_fg_bg_locations(self):
        is_fg = self.gt == 255
        is_bg = ~is_fg
        fg_i, fg_j = np.nonzero(is_fg)
        bg_i, bg_j = np.nonzero(is_bg)
        fg_locations = np.transpose(np.vstack((fg_i, fg_j)))
        bg_locations = np.transpose(np.vstack((bg_i, bg_j)))
        self.fg_locations = fg_locations
        self.bg_locations = bg_locations

    def __set_fg_bg_pixels(self):
        if self.fg_locations is None or self.bg_locations is None:
            self.__set_fg_bg_locations()
        fg_pixels = self.image[self.fg_locations[:, 0], self.fg_locations[:, 1], :]
        bg_pixels = self.image[self.bg_locations[:, 0], self.bg_locations[:, 1], :]
        self.fg_pixels = fg_pixels
        self.bg_pixels = bg_pixels

    def get_fg_bg_locations(self):
        if self.fg_locations is None or self.bg_locations is None:
            self.__set_fg_bg_locations()
        return self.fg_locations, self.bg_locations

    def get_fg_bg_pixels(self):
        if self.fg_pixels is None or self.bg_pixels is None:
            self.__set_fg_bg_pixels()
        return self.fg_pixels, self.bg_pixels


class GraphCutSegmentation:
    def __init__(self, hist_resolution, train_images, train_labels):
        self.hist_resolution = hist_resolution
        self.train_images = train_images
        self.train_labels = train_labels
        self.n_train = np.size(train_images, 0)
        self.fg_histogram = None
        self.bg_histogram = None
        self.window_size = None

    def __get_neighbours(self, i, j, image_rows, image_cols, window_size):
        dim = np.int((window_size - 1) / 2)

        neighbours = []
        for ii in range(i - dim, i + dim):
            for jj in range(j - dim, j + dim):
                if ii != i and jj != j:
                    neighbours.append([i, j])
        neighbours = np.asarray(neighbours)

        is_boundary_1 = 0 <= neighbours[:, 0]
        is_boundary_2 = image_rows > neighbours[:, 0]
        is_boundary_3 = 0 <= neighbours[:, 1]
        is_boundary_4 = image_cols > neighbours[:, 1]

        valid = np.logical_and(np.logical_and(is_boundary_1, is_boundary_2),
                               np.logical_and(is_boundary_3, is_boundary_4))

        return neighbours[valid, :]

    def __get_unaries(self, image, lambda_param):
        """
        :param image: color image as a numpy array
        :param lambda_param: lamdba as set by the user
        :param hist_fg: foreground color histogram
        :param hist_bg: background color histogram
        :param seed_fg: pixels marked as foreground by the user
        :param seed_bg: pixels marked as background by the user
        :return: unaries : Nx2 numpy array containing the unary cost for every pixels in I (N = number of pixels in I)
        """
        print("Computing unaries...")
        hist_fg = self.fg_histogram
        hist_bg = self.bg_histogram

        image_rows = np.size(image, 0)
        image_cols = np.size(image, 1)

        hist_step = 255.0 / self.hist_resolution
        unaries = np.empty((image_rows, image_cols, 2))
        for i in tqdm_notebook(range(0, image_rows)):
            for j in range(0, image_cols):
                neighbour_coordinates = self.__get_neighbours(i, j, image_rows, image_cols, self.window_size)
                all_coordinates = np.vstack((neighbour_coordinates, np.array([i, j])))
                pixel_values = image[all_coordinates[:, 0], all_coordinates[:, 1]]

                pixel_bins = np.floor(pixel_values / hist_step).astype(int)
                pixel_bins[pixel_bins == self.hist_resolution] = self.hist_resolution - 1

                # TODO: Try Gaussian type weighted averaging
                cost_fg = -np.mean(np.log(hist_fg[pixel_bins[:, 0], pixel_bins[:, 1], pixel_bins[:, 2]] + 1e-10))
                cost_bg = -np.mean(np.log(hist_bg[pixel_bins[:, 0], pixel_bins[:, 1], pixel_bins[:, 2]] + 1e-10))

                unaries[i, j, 1] = lambda_param * cost_bg
                unaries[i, j, 0] = lambda_param * cost_fg

        unariesN = np.reshape(unaries, (-1, 2))

        return unariesN

    def __get_pairwise(self, image, sigma):
        """
        Get pairwise terms for each pairs of pixels on image
        :param image: color image as a numpy array
        :param sigma: ad-hoc cost function parameter
        :return: pairwise : ivj (triplet or coo) formatted list of lists containing the pairwise costs for image
        """
        print("Computing pairwises...")
        image_rows = np.size(image, 0)
        image_cols = np.size(image, 1)

        pairwise = []
        for i in tqdm_notebook(range(0, image_rows)):
            for j in range(0, image_cols):
                current_coordinates = np.array([i, j])
                current_index = i * image_cols + j
                current_pixel = image[i, j].astype(float)
                neighbour_coordinates = self.__get_neighbours(i, j, image_rows, image_cols, self.window_size)
                neighbour_indices = neighbour_coordinates[:, 0] * image_cols + neighbour_coordinates[:, 1]
                neighbour_pixels = image[neighbour_coordinates[:, 0], neighbour_coordinates[:, 1]].astype(float)

                pixel_differences = np.subtract(neighbour_pixels, current_pixel)
                pixel_distances = np.linalg.norm(pixel_differences, axis=1)
                spatial_differences = current_coordinates - neighbour_coordinates
                spatial_differences = np.linalg.norm(spatial_differences, axis=1)

                neighbour_costs = np.divide(np.exp(-np.square(pixel_distances) / (2 * np.square(sigma))),
                                            spatial_differences)

                for k in range(0, np.size(neighbour_indices.ravel())):
                    neighbour_index = neighbour_indices[k]
                    cost = neighbour_costs[k]
                    pairwise.append([current_index, neighbour_index, 0, cost, 0, 0])

        pairwise = np.asarray(pairwise)
        return pairwise

    def train(self):
        """
        Compute a color histograms based on selected points from an image
        :param image: color image
        :param seed: Nx2 matrix containing the the position of pixels which will be used to compute the color histogram
        :param histRes: resolution of the histogram
        :return hist: color histogram
        """
        n_images = self.n_train

        dim = self.hist_resolution
        histogram_sum_fg = np.zeros((dim, dim, dim))
        histogram_sum_bg = np.zeros((dim, dim, dim))

        for i in range(0, n_images):
            image = self.train_images[i, :, :, :]
            gt = self.train_labels[i, :, :]

            pair = InputPair(image, gt)

            fg_pixels, bg_pixels = pair.get_fg_bg_pixels()

            histogram_fg, _ = np.histogramdd(fg_pixels, self.hist_resolution, range=[(0, 255), (0, 255), (0, 255)])
            histogram_bg, _ = np.histogramdd(bg_pixels, self.hist_resolution, range=[(0, 255), (0, 255), (0, 255)])

            histogram_sum_fg = histogram_sum_fg + histogram_fg
            histogram_sum_bg = histogram_sum_bg + histogram_bg

        # window_size = 2*int(truncate*sigma + 0.5) + 1, truncate = 4 by default
        smoothed_histogram_fg = ndimage.gaussian_filter(histogram_sum_fg, 1)
        normalized_smoothed_histogram_fg = smoothed_histogram_fg / np.sum(smoothed_histogram_fg.ravel())

        smoothed_histogram_bg = ndimage.gaussian_filter(histogram_sum_bg, 1)
        normalized_smoothed_histogram_bg = smoothed_histogram_bg / np.sum(smoothed_histogram_bg.ravel())

        self.fg_histogram = normalized_smoothed_histogram_fg.astype(np.float32)
        self.bg_histogram = normalized_smoothed_histogram_bg.astype(np.float32)

    def predict(self, test_images, unary_lambda_value, pairwise_sigma_value, window_size, show_prediction=False):
        self.window_size = window_size
        n_images = np.size(test_images, 0)
        predicted_labels = []
        for i in range(0, n_images):
            print(i + 1, "/", n_images)
            image = test_images[i, :, :, :]
            image_rows = np.size(image, 0)
            image_cols = np.size(image, 1)

            unaries = self.__get_unaries(image, unary_lambda_value)
            pairwises = self.__get_pairwise(image, sigma=pairwise_sigma_value)

            n_non_terminal_edges = np.size(pairwises, 0)
            n_nodes = image_rows * image_cols
            print("Finding minimum cut...")
            g = GraphCut(n_nodes, n_non_terminal_edges)
            g.set_unary(unaries)
            g.set_pairwise(pairwises)
            g.minimize()
            predicted_label = g.get_labeling()

            predicted_label = np.reshape(np.logical_not(predicted_label), (image_rows, image_cols)).astype(np.float32)
            if show_prediction:
                plt.figure()
                plt.imshow(predicted_label, cmap="gray")
                plt.show()

            predicted_label = predicted_label.tolist()
            predicted_labels.append(predicted_label)

        predicted_labels = np.asarray(predicted_labels)
        return predicted_labels
