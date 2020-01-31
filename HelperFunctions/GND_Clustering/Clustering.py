import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import math
from scipy.ndimage import affine_transform
from scipy.signal import argrelmin
import os
import concurrent.futures as cf
import time
import argparse


class GaussianNormalDistributionCluster:
    """
    GaussianNormalDistributionCluster provides methods for extracting the density distribution of an image, it's summed gaussian normal distributions and it's minimas for digit seperation.
    In order to render the plots, matplotlib.pyplot.show() must be called after the rendering methods are called.
    The load_image(path) method must be called before using any other method.
    """

    def __init__(self, num_components=3):
        """
        :param num_components: number of gaussian normal distributions
        """
        self.image = None
        self.components = num_components
        self.shape = (100, 100)

    @staticmethod
    def gaussian(x, mu, sig, weight):
        """
        Creates a gaussian normal distribution
        :param x: ndarray of points along the x-axis
        :param mu: standard deviation
        :param sig: covariance
        :param weight: the weight for the normal distribution
        :return: a ndarray containing the points for the normal distribution
        """
        return (np.exp(-np.power(x - mu, 2.) / (2 * sig)) / (math.sqrt(2 * math.pi) * math.sqrt(sig))) * weight

    def load_image(self, path):
        """
        Loads an image in grayscale using opencv
        :param path: path to the image
        :return: ndarray of pixel values, grayscale
        """
        self.image = cv2.imread(path, 0)
        if self.image is None:
            print("Unable to load image, check path")
            raise ValueError
        return self.image

    def get_x_density(self):
        """
        Creates a 1d array containing the location of pixel values on the x-axis above a threshold, load_image must be called first
        :return: list of pixel locations
        """
        if self.image is None:
            raise ValueError
        rows, cols = self.image.shape

        np.random.seed(0)
        affine = np.array([[1, 0, 0], [-0.2, 1, 0], [0, 0, 1]])
        img = affine_transform(self.image, affine, cval=255)
        img_flat = img.flatten()
        img_flat = [v / 255 for v in img_flat]

        x_density = []
        for i in range(0, len(img_flat)):
            if img_flat[i] < 0.1:
                x_density.append(np.array([i % cols]))

        return np.array(x_density)

    def get_minimas(self, summed_gaussian=None):
        """
        Returns local minimas of the gaussian function
        :param summed_gaussian: sum of gaussian normal distributions. If None, the method will retrieve a summed gaussian for the given number of components
        :return: local minimas. None if the image contains no valid pixels, see method get_x_density().
        """
        if summed_gaussian is None:
            summed_gaussian = self.get_summed_gaussian()
            if summed_gaussian is None:
                return None
        minims = argrelmin(summed_gaussian)
        return minims

    @staticmethod
    def render_hist(x_density, num_bins=100):
        """
        Render method for a histogram
        :param x_density: list of x-axis pixel locations
        :param num_bins: number of bins to separate the values in to
        :return:
        """
        plt.hist(x_density, histtype='bar', normed=True, bins=num_bins)

    @staticmethod
    def render_dist(gaussian):
        """
        Render the given gaussian distribution
        :param gaussian: list containing the gaussian distribution
        :return:
        """
        plt.plot(gaussian)

    def get_summed_gaussian(self, x_density=None, init_weight=1 / 3):
        """
        Creates and summarizes the gaussian normal distributions
        :param x_density: list of pixel locations on the x-axis
        :param init_weight: initial weight for the distributions
        :return: summed gaussian distribution. If None, no valid (normalized pixels < 0.1) pixels are in the image
        """

        if x_density is None:
            x_density = self.get_x_density()

        if len(x_density) == 0:
            return None

        weights = np.full(self.components, init_weight)
        gmm = GaussianMixture(n_components=self.components, weights_init=weights)
        gmm.fit(x_density)

        mu = gmm.means_.flatten()
        sig = gmm.covariances_.flatten()
        gausses = []
        for i in range(0, len(mu)):
            g = self.gaussian(np.arange(self.image.shape[1]), mu[i], sig[i], gmm.weights_[i])
            gausses.append(g)
        gausses = np.array(gausses)
        sum_g = gausses.sum(axis=0)

        return sum_g

    def resize_images(self, images):
        completed = []
        for image in images:
            x_off = y_off = 1
            img = cv2.resize(image, (self.shape[0] - y_off, self.shape[1] - x_off), interpolation=cv2.INTER_AREA)
            reshaped = np.full(self.shape, 255, dtype='uint8')
            p = np.array(img)
            reshaped[x_off:p.shape[0] + x_off, y_off:p.shape[1] + y_off] = p
            completed.append(reshaped)
        return completed

    def split_image(self, image, split_points):

        new1 = [row[:split_points[0]] for row in image]
        new2 = [row[split_points[0]:split_points[1]] for row in image]
        new3 = [row[split_points[1]:] for row in image]
        new1 = np.array(new1)
        new2 = np.array(new2)
        new3 = np.array(new3)
        return self.resize_images([new1, new2, new3])


def run_test(path):
    np.random.seed(0)
    gnc = GaussianNormalDistributionCluster()
    img = gnc.load_image(path)
    x_density = gnc.get_x_density()
    gnc.render_hist(x_density)
    sum_g = gnc.get_summed_gaussian(x_density)
    gnc.render_dist(sum_g)
    mins = gnc.get_minimas(sum_g)
    plt.show()
    cv2.line(img, (mins[0][0], img.shape[1]), (mins[0][0], 0), (0, 0, 0))
    cv2.line(img, (mins[0][1], img.shape[1]), (mins[0][1], 0), (0, 0, 0))
    cv2.imwrite("example.jpg", img)
    new_images = gnc.split_image(img, np.array([mins[0][0], mins[0][1]]))
    cv2.imshow("0", new_images[0])
    cv2.imshow("1", new_images[1])
    cv2.imshow("2", new_images[2])
    cv2.imshow("original", img)
    cv2.waitKey()


def execute(root, file, output):
    """
    Function to handle the launching of a parallel task
    :param root: root directory
    :param file: name of the file
    :return: list of images separated, name of the new folder, name of the new file
    """
    gnc = GaussianNormalDistributionCluster()
    path = os.path.join(root, file)
    image = gnc.load_image(path)
    try:
        mins = gnc.get_minimas()
        if mins is None:
            return None, None, None
    except ValueError:
        # Unsure of what exactly happens here, but the x_density vector is only a single dimension
        # which causes the GMM to fail. This can happen if there is only a single row containing pixels, or none
        # These images are however not relevant and can be skipped.

        print(ValueError)
        return None, None, None

    try:
        new_images = gnc.split_image(image, np.array([mins[0][0], mins[0][1]]))
        new_folder = os.path.join(output, file.split(".jpg")[0])
        return new_images, new_folder, file
    except IndexError as e:
        # Only one minima is found, this is the wrong result for the profession field. Should be two minimas
        # So these images are just skipped.
        print(e)
        return None, None, None


def handle_done(done):
    """
    Function to handle the output of a parallel task
    :param done: Handle to the concurrent.future task object
    :return:
    """
    new_images, new_folder, file = done.result()
    if new_images is None or new_folder is None or file is None:
        return
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    for i, im in enumerate(new_images):
        new_image_filename = os.path.join(str(new_folder), str(i) + "_" + file)
        # Lagt inn av meg for aa lagre invertet
        im = cv2.bitwise_not(im)
        cv2.imwrite(new_image_filename, im)


def run_parallel(path):
    np.random.seed(0)
    start_time = time.time()
    futures = []
    num = 0
    num_read = 0
    # The following is not big data safe, could run out of memory, best would be to create a stream, but python....
    image_strings = []
    # All usage of image strings might be volatile
    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        for root, subdirs, files in os.walk(path):
            for file in files:
                num_read += 1
                image_strings.append((root, file))

        for name in image_strings:
            futures.append(executor.submit(execute, name[0], name[1], args.output))

        for done in cf.as_completed(futures):
            handle_done(done)
            futures.remove(done)
            num += 1
            if num % 100 == 0:
                print("Number of images segmented is: {} out of a total of {}".format(num, num_read))
        print("--- " + str(time.time() - start_time) + " ---")


def handle_main(args):
    if args.test:
        run_test(args.path)
    else:
        run_parallel(args.path)


if __name__ == '__main__':
    arg = argparse.ArgumentParser("Extract individual digits from image")
    arg.add_argument("-t", "--test", action="store_true", default=False, help="Run the program in test_mode")
    arg.add_argument("-p", "--path", type=str,
                     help="path to root directory if not running test. If test, full path to image")
    arg.add_argument("-o", "--output", type=str, help="output path")
    args = arg.parse_args()
    handle_main(args)
