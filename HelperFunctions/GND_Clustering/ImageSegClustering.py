import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import math
from scipy.ndimage import affine_transform
from scipy.signal import argrelmin, argrelmax
import concurrent.futures as cf
import time
import argparse

import sys
sys.path.insert(0, '//homer.uit.no/bpe043/Desktop/Test_Projects/HelperFunctions')

from Database.dbHandler import DbHandler

from sheer_image import sheer_image

from color_convert import convert_img_gray
from color_convert import convert_img_bw


class GaussianNormalDistributionCluster:
    """
    GaussianNormalDistributionCluster provides methods for extracting the density distribution of an image,
    it's summed gaussian normal distributions and it's minimas for digit seperation.
    In order to render the plots, matplotlib.pyplot.show() must be called after the rendering methods are called.
    The load_image(path) method must be called before using any other method.
    """

    # num_components = How many digits there are
    def __init__(self, num_components):
        """
        :param num_components: number of gaussian normal distributions
        :param img: image to process
        """
        self.image = None
        self.components = num_components
        self.shape = (100, 100)
        self.gaussian_values = None
        self.single = False
        

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

    def load_image(self, img, height, width):
        """
        Loads an image in grayscale using opencv
        :param img: image in byte values
        :return: ndarray of pixel values, grayscale
        :type:ndarray
        """
        
        # Check if the image type is bytes (Normal use) or ... (Training set use)
        if type(img) == np.ndarray:
            self.image = img
            return self.image
        
        # Convert the bytes-data from the database into a numpy array
        np_img = np.frombuffer(img, dtype = np.uint8)
        
        # Decode the array back to an image
        image = cv2.imdecode(np_img, cv2.IMREAD_ANYCOLOR)
    
        self.image = image
        affine = np.array([[1, 0, 0], [-0.3, 1, 0], [0, 0, 1]])
        img = affine_transform(self.image, affine, cval=255)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        self.image = img
        if self.image is None:
            print("Image is None")
            raise ValueError("Unable to load image, check path")
            
        return self.image

    def get_x_density(self):
        """
        Creates a 1d array containing the location of pixel values on the x-axis above a threshold,
        load_image must be called first
        :return: list of pixel locations
        """
        if self.image is None:
            raise ValueError
        if len(self.image.shape) == 3:
            cols = self.image.shape[1]
        else:
            rows, cols = self.image.shape

        np.random.seed(0)
        img_flat = self.image.flatten()
        img_flat = [v / 255 for v in img_flat]
        img_flat = np.array(img_flat)
        x_density = []
        for i in range(0, len(img_flat)):
            if img_flat[i] < 0.2:
                x_density.append(np.array([i % cols]))

        return np.array(x_density)

    def get_minimas(self, summed_gaussian=None):
        """
        Returns local minimas of the gaussian function
        :param summed_gaussian: sum of gaussian normal distributions. If None, the method will retrieve a summed
        gaussian for the given number of components
        :return: local minimas. None if the image contains no valid pixels, see method get_x_density().
        """
        if summed_gaussian is None:
            summed_gaussian = self.get_summed_gaussian()
            if summed_gaussian is None:
                return None
        minims = argrelmin(summed_gaussian)
        return minims

    def get_maxims(self, summed_gaussian=None):
        """
        Finds the maximum points for the summed gaussian function. Can handle single gaussian functions as well.
        :param summed_gaussian: Function of which to find the local maximum
        :return: array of local maximum values
        """
        if summed_gaussian is None:
            summed_gaussian = self.get_summed_gaussian()
            if summed_gaussian is None:
                return None
        maxims = argrelmax(summed_gaussian)
        return maxims

    @staticmethod
    def render_hist(x_density, num_bins=28):
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

    
    def get_summed_gaussian(self, x_density=None):
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
    
        # 1/3 = 3 digits, 1/2 = 2 digits    
        init_weight = 1 / self.components

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
        self.gaussian_values = gausses
        sum_g = gausses.sum(axis=0)
        return sum_g

    def resize_images(self, images):
        completed = []
        for image in images:
            if image.shape[0] == 0:
                print("The image shape on the x axis is {}".format(image.shape[0]))
            if image.shape[1] == 0:
                print("The image shape on the y axis is {}".format(image.shape[1]))
            if image.shape[0] > self.shape[0]:
                # Resize the image if an axis is too large to fit in the new image
                if image.shape[1] > self.shape[1]:
                    # Both axis in the image is greater than the wanted shape, resize both axis
                    image = cv2.resize(image, self.shape, interpolation=cv2.INTER_CUBIC)
                else:
                    # Only the X axis is greater, resize only this
                    image = cv2.resize(image, (image.shape[1], self.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                if image.shape[1] > self.shape[1]:
                    # Only the Y axis is greater, resize only this
                    image = cv2.resize(image, (self.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

            reshaped = np.full(self.shape, 0, dtype='uint8')
            p = np.array(image)
            x_offset = int(abs(image.shape[0] - self.shape[0]) / 2)
            y_offset = int(abs(image.shape[1] - self.shape[1]) / 2)
            reshaped[x_offset:p.shape[0] + x_offset, y_offset:p.shape[1] + y_offset] = p
            completed.append(reshaped)

        return completed

    def split_image(self, image, split_points, mid_points):
        """
        Splits the image based on the location of the minimum points given by the summed gaussian function
        :param image: Input image in grayscale
        :param split_points: Local minimum points of the summed gaussian function
        :param mid_points: Maximum points of the summed gaussian function
        :return: an array of the split images
        """

        def test_for_value(col):
            for col_val in col:
                if col_val > 200:
                    # We found a value in this column, so go to next
                    return True
            return False

        if self.components == 3:
            new1 = np.array([row[:split_points[0]] for row in image])
            new2 = np.array([row[split_points[0]:split_points[1]] for row in image])
            new3 = np.array([row[split_points[1]:] for row in image])
            
            center1 = mid_points[0]
            center3 = mid_points[2] - split_points[1]
        
        else:
            new1 = np.array([row[:split_points[0]] for row in image])
            new3 = np.array([row[split_points[0]:] for row in image])
            
            center1 = mid_points[0]
            center3 = mid_points[1]

        
        """ The following code will be done for both 3-digit and 2-digit"""

        # Left (First) image
        try:
            new1 = self.reshape_left_image(new1, test_for_value, center1)
        except ValueError as e:
            try:
                intersections = self.find_intersections()
                new1 = np.array([row[:intersections[0]] for row in image])
                new1 = self.reshape_left_image(new1, test_for_value, mid_points[0])
            except Exception as e:
                print("Left image has wrong shape {}, exception: {}".format(new1.shape, e))
                return None
        
        # Right (Third) image
        try:
            new3 = self.reshape_right_image(new3, test_for_value, center3)
        except ValueError as e:
            try:
                intersections = self.find_intersections()
                new3 = np.array([row[intersections[1]:] for row in image])
                new3 = self.reshape_right_image(new3, test_for_value, mid_points[2] - intersections[1])
            except Exception as e:
                print("Right image has wrong shape {}, exception: {}".format(new3.shape, e))
                return None
        
        
        all_i = [new1, new3]
        
        """ The below code will only be done for 3-digit """
            
        if self.components == 3:
        
            # Middle (Second) image
            try:
                new2 = self.reshape_middle_image(new2)
            except ValueError as e:
                try:
                    intersections = self.find_intersections()
                    new2 = np.array([row[intersections[0]:intersections[1]] for row in image])
                    new2 = self.reshape_middle_image(new2)
                except Exception as e:
                    print("Middle image has wrong shape {}, exception: {}".format(new2.shape, e))
                    return None

            all_i.insert(1, new2)

        if self.single is True:
            return all_i
        
        all_images_resized = self.resize_images(all_i)
        
        return all_images_resized

    @staticmethod
    def reshape_right_image(new3, test_for_value, digit_center_point):
        # Right image
        # Calculate offset from the total image length
        from_mid = np.swapaxes(new3[:, digit_center_point:], 1, 0)
        
        for i in range(0, from_mid.shape[0] - 2, 2):
            # Iterate from the top of the new image
            # Check if the row contains values
            if not test_for_value(from_mid[i]):
                # Check the next row for values
                if not test_for_value(from_mid[i + 1]) and not test_for_value(from_mid[i + 2]):
                    # We found a row without values, and the next does not either
                    # Copy over the values based on the new first column containing values
                    new3 = new3[:, :i + digit_center_point]
                    break
        if new3.shape[0] == 0 or new3.shape[1] == 0:
            raise ValueError
        
        return new3

    @staticmethod
    def reshape_middle_image(new2):
        # left = self.reshape_left_image(new2, test_for_value, digit_center_point)
        # right = self.reshape_right_image(new2, test_for_value, digit_center_point)
        # if left.shape[0] < right.shape[0]:
        #     new2 = left
        # else:
        #     new2 = right
        if new2.shape[0] == 0 or new2.shape[1] == 0:
            raise ValueError

        return new2

    @staticmethod
    def reshape_left_image(new1, test_for_value, digit_center_point):
        # Left image
        # Extract array from mid point of the digit and switch to column major order
        from_mid = np.swapaxes(new1[:, digit_center_point:0:-1], 1, 0)


        for i in range(0, from_mid.shape[0] - 2, 2):
            # Iterate from the bottom of the new image
            # Check if the row contains values
            if not test_for_value(from_mid[i]):
                # Check the next row for values
                if not test_for_value(from_mid[i + 1]) and not test_for_value(from_mid[i + 2]):
                    # We found a row without values, and the next does not either
                    # Copy over the values based on the new first column containing values
                    new1 = new1[:, digit_center_point - i:]
                    break
        if new1.shape[0] == 0 or new1.shape[1] == 0:
            raise ValueError
        
        return new1

    def find_intersections(self):
        """
        Finds the intersection between the gaussian functions. These are loaded from the class and assumes that the
        gaussian functions have already been created. Fails with an exception by default if the functions are not
        created
        :return:
        """
        gaus_and_mid = []
        for val in self.gaussian_values:
            gaus_and_mid.append((self.get_maxims(val)[0][0], val))
        gaus_and_mid = sorted(gaus_and_mid, key=lambda q: q[0])
        intersections = []
        try:
            for i in range(0, len(gaus_and_mid) - 1):
                for k, val in enumerate(gaus_and_mid[i][1]):
                    if k == len(gaus_and_mid[i][1]) - 3:
                        break
                    a = val
                    b = gaus_and_mid[i + 1][1][k]
                    c = gaus_and_mid[i][1][k + 3]
                    d = gaus_and_mid[i + 1][1][k + 3]
                    if a > c:
                        tmp = c
                        c = a
                        a = tmp
                    if b > d:
                        tmp = d
                        d = b
                        b = tmp
                    if (a <= d and c >= b) and k > gaus_and_mid[i][0]:
                        intersections.append(k)
                        break
        except Exception as e:
            print(e)
        return intersections


def execute(name, img, height, width, nr_digits, gray_img = None):
    """
    Function to handle the launching of a parallel task
    :param name: Name of the image
    :param img: image
    :return: list of images separated, name of the file, error message if not completed
    """
    
    
    
    gnc = GaussianNormalDistributionCluster(nr_digits)
    try:
        image = gnc.load_image(img, height, width)
        x_density = gnc.get_x_density()
        sum_g = gnc.get_summed_gaussian(x_density)
        mins = gnc.get_minimas(sum_g)
        if mins is None:
            return None, name, "No minimums found"
        maxes = gnc.get_maxims(sum_g)
        if maxes is None:
            return None, name, "No maximums found"
    except ValueError as e:
        # Unsure of what exactly happens here, but the x_density vector is only a single dimension
        # which causes the GMM to fail. This can happen if there is only a single row containing pixels, or none
        # These images are however not relevant and can be skipped.

        print("{} Skipping image at path: {} due to lacking values in x_density".format(e, name))
        return None, name, " lacking values in x_density. Exception {}".format(e)
    except Exception as e:
        print(e)
        return None, name, str(e)

    try:
        
# =============================================================================
#         cv2.imshow('before', image)
#         cv2.waitKey(0)
# =============================================================================
        
        
        # If we are not working with a grayscale image, operate as normal
        if gray_img is None:
            image = cv2.bitwise_not(image)
        
        # If we are working with a grayscale image, the splitting points have been calculated using the black and white image
        # Now we pass the grayscale image to the function that splits it based on the previous calculations
        else:
            image = gnc.load_image(gray_img, height, width)
        
# =============================================================================
#         cv2.imshow('after', image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
# =============================================================================
        
        new_images = gnc.split_image(image, mins[0], maxes[0])
        if new_images is None:
            return None, name, "No images returned"
        return new_images, name, ""
    except IndexError as e:
        # Only one minima is found, this is the wrong result for the profession field. Should be two minimas
        # So these images are just skipped.
        print("{} Skipping image at path: {} due to single minima or maxima".format(e, name))
        return None, name, "single minima or maxima. Exception {}".format(e)
    except Exception as e:
        print(e)
        return None, name, str(e)


def handle_done(done, db):
    """
    Function to handle the output of a parallel task
    :param done: Handle to the result
    :type: Future
    :param db: database handler
    :type: DbHandler
    :return:
    """

    new_images, name, err = done.result()
    if new_images is None or err != "":
        try:
            db.store_dropped(name, err)
        except Exception as e:
            print(e)
    else:
        for i, im in enumerate(new_images):
            name = str(i) + "_" + name
            try:
                db.store_digit(name, im)
            except Exception as e:
                print(e)
                

def run_parallel(db_loc, nr_digits, gray_loc = None):
    """
    Launches the parallel executor and submits all the jobs. This function parses the entire folder structure and keeps
    it in memory
    :param db_loc: black and white image database location, full path
    :param gray_loc: grayscale image database location, full path
    :return:
    """
    np.random.seed(0)
    start_time = time.time()
    futures = []
    with cf.ProcessPoolExecutor(max_workers=6) as executor:
        with DbHandler(db_loc) as db:
            
            # read_and_submit is the function where we read in images from the database
            # As such, we need to pass both databases
            if gray_loc is not None:
                with DbHandler(gray_loc) as gray_db:
                    read_and_submit(db, executor, futures, nr_digits, gray_db)
            else:
                read_and_submit(db, executor, futures, nr_digits)
            
            print("--- " + str(time.time() - start_time) + " ---")


def process_futures(db, futures, num, num_read):
    for done in cf.as_completed(futures):
        num += 1
 
        if num % 100 == 0:
            print("Number of images segmented is: {}/{}".format(num, num_read))
            db.connection.commit()
        futures.remove(done)
        handle_done(done, db)
    return num


def read_and_submit(db, executor, futures, nr_digits, gray_db = None):
    num = 0
    skipped = 0
    gray_skipped = 0
    
    """
    # After this function, everything is about uploading split images to the database
    # As such, there is no longer need to keep track of two databases.
    # If we are working with a grayscale database, then that is the only one that should be uploaded to
    # Hence we set the grayscale database as our 'active_db'
    """
    
    # Variable for when we no longer need to consider two databases
    active_db = None
    
    if gray_db is not None:
        active_db = gray_db
        num_read = gray_db.count_rows_in_fields().fetchone()[0]
    else:
        active_db = db
        num_read = db.count_rows_in_fields().fetchone()[0]


    try:
        rows = db.select_all_images()
        while True:   
            db_img = rows.fetchone()
            gray_img = None
                  
            if db_img is None or num == num_read:
                print("Reached the end, number of skipped images: ", str(skipped))
                break
            
            if gray_db is not None:
                # Getting the same image but in grayscale. The black and white image will be used to compute changes that need to be done to the grayscale image
                gray_img = gray_db.select_image(db_img[0])
                
                # If the black and white image does not exist in the grayscale database, continue to the next image
                if gray_img is None:
                    gray_skipped += 1
                    print("Skipping image that does not exist in the grayscale database. Total: {}".format(gray_skipped))
                    continue
                else:
                    gray_img = gray_img[1]
                
            exists_digit = active_db.test_exists_digit(db_img[0])[0]
            exists_dropped = active_db.test_exists_dropped(db_img[0])[0]
            if exists_digit == 1 or exists_dropped == 1:
                skipped += 1
                continue

            
            if len(futures) > 1000:
                # Each time a limit is reached, process all the executed
                num = process_futures(active_db, futures, num + skipped, num_read)
                
            futures.append(executor.submit(execute, db_img[0], db_img[1], db_img[2], db_img[3], nr_digits, gray_img))
            

        # Do the final batch
        process_futures(active_db, futures, num, num_read)
    except TypeError as e:
        print(e)
    except Exception as e:
        print(e)
        
def split_and_convert(image):
    
    orig = image
    bw = convert_img_bw(image)
    
    new_dims = sheer_image(bw)
    bw = bw[:, new_dims[0]:new_dims[1]]
    orig = orig[:, new_dims[0]:new_dims[1]]
    
    new_bws = split_single(bw)
    
    # Check if the splitting gave an error. e.g not enough split points (minimums)
    if new_bws == 'error':
        return None
    
    # Using the splitting points from the B&W split images, we can split the original colour image as well
    new_originals = []
    new_originals.append(orig[:, :new_bws[0].shape[1]])
    new_originals.append(orig[:, new_bws[0].shape[1]:(new_bws[0].shape[1] + new_bws[1].shape[1])])
    new_originals.append(orig[:, new_bws[0].shape[1] + new_bws[1].shape[1]:])
    
    i = 0
    while i < len(new_bws):
        new_bws[i] = cv2.resize(new_bws[i], (100, 100), interpolation = cv2.INTER_AREA)
        new_originals[i] = cv2.resize(new_originals[i], (100, 100), interpolation = cv2.INTER_AREA)
        
        i += 1
    
    # Once we have a split original, we can convert those into greyscale
    new_greys = []
    for image in new_originals:
        grey = convert_img_gray(image)
        new_greys.append(grey[1])
    
    return new_originals, new_bws, new_greys


def handle_main():    
    arg = argparse.ArgumentParser("Extract individual digits from image")
    arg.add_argument("-t", "--test", action="store_true", default=False, help="Run the program in test_mode")
    arg.add_argument("--db", type=str, help="full path to database location",
                     default="/mnt/remote/Yrke/ft1950_ml.db")
    arg.add_argument("--gray", type=str, help="full path to grayscale database location",
                     default="")
    arg.add_argument('-nr', '--digits', type=int, help='the number of sub-images you want the image split into, should be equalt to number of digits in the image',
                     default=3)
    arg.add_argument("-tn","--test_name", type=str, help='Name of the test image', default=False)
    args = arg.parse_args()
    if args.test:
        run_test(args.db, args.test_name, args.digits)
    elif args.gray:
        run_parallel(args.db, args.digits, args.gray)
    else:
        run_parallel(args.db, args.digits)


def run_test(db_loc, image_name, nr_digits):
    """
    Test run against single images
    :param path: path to the image
    :return:
    """

    db = DbHandler(db_loc)
    db_image_entry = db.select_image(image_name)
    gnc = GaussianNormalDistributionCluster(nr_digits)
    img = gnc.load_image(db_image_entry[1], db_image_entry[2], db_image_entry[3])
    x_density = gnc.get_x_density()
    gnc.render_hist(x_density)
    sum_g = gnc.get_summed_gaussian(x_density)
    gnc.render_dist(sum_g)
    mins = gnc.get_minimas(sum_g)
    maxes = gnc.get_maxims(sum_g)
    plt.scatter(np.append(mins[0], maxes[0]), np.append(sum_g[mins[0]], sum_g[maxes[0]]), c='r', zorder=10)
    plt.show()
    new_images, _, _ = execute("", db_image_entry[1], db_image_entry[2], db_image_entry[3], nr_digits)

# =============================================================================
#     cv2.line(gnc.image, (mins[0][0], img.shape[1]), (mins[0][0], 0), 0)
#     cv2.line(gnc.image, (mins[0][1], img.shape[1]), (mins[0][1], 0), 0)
# =============================================================================
    
    
    cv2.imshow('First', new_images[0])
    cv2.imshow('Second', new_images[1])
    if nr_digits == 3:
        cv2.imshow("Third", new_images[2])
        
    cv2.imshow("image", gnc.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
""" Test to use the splitter functionality to make a new training set"""
def split_single(image):
    
    
    gnc = GaussianNormalDistributionCluster(3)
    gnc.single = True                                                          # Used to return splitting values instead of resized, split images
    gnc.load_image(image, image.shape[0], image.shape[1])
    x_density = gnc.get_x_density()
    #gnc.render_hist(x_density)
    sum_g = gnc.get_summed_gaussian(x_density)
    #gnc.render_dist(sum_g)
    mins = gnc.get_minimas(sum_g)
    maxes = gnc.get_maxims(sum_g)
    
    if mins is None:
        return None, "No minimums found"
    maxes = gnc.get_maxims(sum_g)
    if maxes is None:
        return None, "No maximums found"
    
    if len(mins[0]) < 2:
        return 'error'
    elif len(maxes[0]) < 3:
        return 'error'

    new_images = gnc.split_image(image, mins[0], maxes[0])
    if new_images is None:
        return None, "No images returned"
    return new_images


if __name__ == '__main__':
    __spec__ = None
    handle_main()
