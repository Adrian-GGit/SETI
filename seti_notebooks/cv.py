import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import cv2
from PIL import Image
from time import perf_counter
from collections import Counter

filter_x_test = np.array([
    [0, 0, 0], 
    [1, -2, 1], 
    [0, 0, 0]])
filter_y_test = np.array([
    [0, 1, 0], 
    [0, -2, 0], 
    [0, 1, 0]])
filter_diagonal_test = np.array([
    [1, 0, 1],
    [0, -4, 0],
    [1, 0, 1]])

filter_x_prewitt = np.array([
    [-1, 0, 1], 
    [-1, 0, 1], 
    [-1, 0, 1]])
filter_y_prewitt = np.array([
    [-1, -1, -1], 
    [0, 0, 0], 
    [1, 1, 1]])
filter_x_sobel = np.array([
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1]])
filter_y_sobel = np.array([
    [-1, -2, -1], 
    [0, 0, 0], 
    [1, 2, 1]])
filter_diagonal_x = np.array([
    [0, 1],
    [-1, 0]
])
filter_gauss = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
])
filter_mexican_hat = np.array([
    [0, 0, -1, 0, 0],
    [0, -1, -8, -1, 0],
    [-1, -8, 32, -8, -1],
    [0, -1, -8, -1, 0],
    [0, 0, -1, 0, 0],
])
class Cadence():
    def __init__(self, path):
        self.path = path # path of .npy file
        self.cadence = np.load(self.path)
        self.cadence_f32 = self.cadence.astype(np.float32)
        self.cadence_transformed = np.load(path)
        self.cadence_transformed_f32 = self.cadence_transformed.astype(np.float32)
        self.vmin = 0
        self.vmax = 255
        self.mapped_to_grayscale = False


    def get_cadence_f32(self):
        return self.cadence_f32

    def get_cadence_transformed_f32(self):
        return self.cadence_transformed_f32

    def print_cadence_f32(self):
        print(self.cadence_f32)

    def print_cadence_transformed_f32(self):
        print(self.cadence_transformed_f32)

    def plot_cadence(self, plot_transformed, plot_gray=False) -> None:
        if plot_transformed:
            cadence = self.cadence_transformed_f32
        else:
            cadence = self.cadence_f32
        cmap = "gray" if plot_gray else None
        fig = plt.figure(figsize=(20,15))
        gs = GridSpec(2, 3)
        num_imgs = cadence.shape[0]
        for location, on_target in zip(range(int(num_imgs / 2)), range(0, num_imgs, 2)):
            ax = fig.add_subplot(gs[0, location])
            if self.mapped_to_grayscale:
                ax.imshow(cadence[on_target], cmap=cmap, vmin=self.vmin, vmax=self.vmax)
            else:
                ax.imshow(cadence[on_target], cmap=cmap)
            ax.set_title("ON TARGET")
        for location, off_target in zip(range(int(num_imgs / 2)), range(1, num_imgs, 2)):
            ax = fig.add_subplot(gs[1, location])
            if self.mapped_to_grayscale:
                ax.imshow(cadence[off_target], cmap=cmap, vmin = self.vmin, vmax = self.vmax)
            else:
                ax.imshow(cadence[off_target], cmap=cmap)
            ax.set_title("OFF TARGET")
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

    def plot_cadence_for_comparing_differences(self, plot_gray=False) -> None:
        cmap = "gray" if plot_gray else None
        fig = plt.figure(figsize=(40,15))
        gs = GridSpec(2, 6) 
        num_imgs = self.cadence_f32.shape[0]
        for location, on_target in zip(range(int(num_imgs / 2)), range(0, num_imgs, 2)):
            ax = fig.add_subplot(gs[0, location * 2])
            ax.imshow(self.cadence_f32[on_target], cmap=cmap)
            ax.set_title("ON TARGET ORIGINAL")
        for location, on_target in zip(range(int(num_imgs / 2)), range(0, num_imgs, 2)):
            ax = fig.add_subplot(gs[0, location * 2 + 1])
            if self.mapped_to_grayscale:
                ax.imshow(self.cadence_transformed_f32[on_target], cmap=cmap, vmin=self.vmin, vmax=self.vmax)
            else:
                ax.imshow(self.cadence_transformed_f32[on_target], cmap=cmap)
            ax.set_title("ON TARGET TRANSFORMED")
        for location, off_target in zip(range(int(num_imgs / 2)), range(1, num_imgs, 2)):
            ax = fig.add_subplot(gs[1, location * 2])
            ax.imshow(self.cadence_f32[off_target], cmap=cmap)
            ax.set_title("OFF TARGET ORIGINAL")
        for location, off_target in zip(range(int(num_imgs / 2)), range(1, num_imgs, 2)):
            ax = fig.add_subplot(gs[1, location * 2 + 1])
            if self.mapped_to_grayscale:
                ax.imshow(self.cadence_transformed_f32[off_target], cmap=cmap, vmin = self.vmin, vmax = self.vmax)
            else:
                ax.imshow(self.cadence_transformed_f32[off_target], cmap=cmap)
            ax.set_title("OFF TARGET TRANSFORMED")
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

    def plot_histogram(self, plot_transformed) -> None:
        if plot_transformed:
            cadence = self.cadence_transformed_f32
        else:
            cadence = self.cadence_f32
        cadence_flatted = cadence.flatten()
        plt.hist(cadence_flatted)
        plt.show()

    def map_to_colorscale(self, max=None, min=None) -> None:
        if max is None:
            max = self.vmax
        if min is None:
            min = self.vmin
        color_scale_full_difference = abs(max - min)
        for image_num, image in enumerate(self.cadence_transformed_f32):
            image_min = np.amin(image)
            image_max = np.amax(image)
            image_full_difference = image_max - image_min
            for row_num, row in enumerate(image):
                for cell_num, cell in enumerate(row):
                    cell_difference = cell - image_min
                    cell_relative_difference = cell_difference / image_full_difference
                    cell_new_value = cell_relative_difference * color_scale_full_difference
                    self.cadence_transformed_f32[image_num][row_num][cell_num] = cell_new_value
        if min == 0 and max == 255:
            self.mapped_to_grayscale = True
        return self.cadence_transformed_f32

    def invert(self) -> None:
        for image_num, image in enumerate(self.cadence_transformed_f32):
            for row_num, row in enumerate(image):
                for cell_num, cell in enumerate(row):
                    inverted = self.vmax - cell
                    self.cadence_transformed_f32[image_num][row_num][cell_num] = inverted
        return self.cadence_transformed_f32

    def change_contrast(self, percentage=0) -> None:
        contrast = 1 + (percentage / 100)
        for image_num, image in enumerate(self.cadence_transformed_f32):
            for row_num, row in enumerate(image):
                for cell_num, cell in enumerate(row):
                    changed_contrast = cell * contrast
                    if changed_contrast > self.vmax:
                        changed_contrast = self.vmax
                    if changed_contrast < self.vmin:
                        changed_contrast = self.vmin
                    self.cadence_transformed_f32[image_num][row_num][cell_num] = changed_contrast
        return self.cadence_transformed_f32

    def shift(self, level) -> None:
        for image_num, image in enumerate(self.cadence_transformed_f32):
            for row_num, row in enumerate(image):
                for cell_num, cell in enumerate(row):
                    shifted = cell + level
                    if shifted > self.vmax:
                        shifted = self.vmax
                    if shifted < self.vmin:
                        shifted = self.vmin
                    self.cadence_transformed_f32[image_num][row_num][cell_num] = shifted
        return self.cadence_transformed_f32

    def gamma(self, level) -> None:
        for image_num, image in enumerate(self.cadence_transformed_f32):
            for row_num, row in enumerate(image):
                for cell_num, cell in enumerate(row):
                    used_gamma = (math.pow((cell / self.vmax), level)) * self.vmax
                    self.cadence_transformed_f32[image_num][row_num][cell_num] = used_gamma
        return self.cadence_transformed_f32

    def remove_from_level(self, level, remove_lower): # set everything under level (%) to vmin if remove_lower - if not remove_lower => set everything above level (%) to vmax
        for image_num, image in enumerate(self.cadence_transformed_f32):
            image_max = np.amax(image)
            for row_num, row in enumerate(image):
                for cell_num, cell in enumerate(row):
                    exact_value = image_max * (level / 100)
                    if remove_lower:
                        if cell <= exact_value:
                            new_value = self.vmin
                        else:
                            new_value = cell
                    else:
                        if cell >= exact_value:
                            new_value = self.vmax
                        else:
                            new_value = cell
                    self.cadence_transformed_f32[image_num][row_num][cell_num] = new_value
        return self.cadence_transformed_f32

    def add_0_border(self, to_add):
        # self.cadence_transformed_f32 = np.array([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
        new_cadence_transformed = np.zeros(shape=(self.cadence_transformed_f32.shape[0], self.cadence_transformed_f32.shape[1] + 2*to_add, self.cadence_transformed_f32.shape[2] + 2*to_add))
        for image in range(new_cadence_transformed.shape[0]):
            new_cadence_transformed[image] = np.pad(self.cadence_transformed_f32[image], pad_width=to_add, mode='constant', constant_values=0)
        return new_cadence_transformed

    def convolution_filter(self, kernel):
        kernel_size = len(kernel)
        to_add = math.ceil(kernel_size / 2) - 1
        new_cadence_transformed = self.add_0_border(to_add)
        images = new_cadence_transformed.shape[0]
        rows = new_cadence_transformed.shape[1]
        cols = new_cadence_transformed.shape[2]

        for image in range(images):
            for i in range(rows - 2*to_add):
                for j in range(cols - 2*to_add):
                    current = new_cadence_transformed[image][i:i+kernel_size, j:j+kernel_size] # get for example 3x3 matrix out of image
                    multiplication = sum(sum(current * kernel)) # weighted addition
                    self.cadence_transformed_f32[image][i, j] = multiplication # set center where filter on image is to result of weighted addition

    def clean_image_columnwise(self) -> np.ndarray: # https://www.kaggle.com/tatamikenn/image-cleaning-column-wise-normalization-knn
        for image in range(self.cadence_transformed_f32.shape[0]):
            neigh = NearestNeighbors(n_neighbors=2, radius=0.001, algorithm="brute", metric="minkowski", p=2)
            neigh.fit(self.cadence_transformed_f32[image].T)
            _, inds = neigh.kneighbors(self.cadence_transformed_f32[image].T)
            nearest_columns = inds[:, 1]
            # nearest_columns2 = inds[:, 2]
            self.cadence_transformed_f32[image] = abs(self.cadence_transformed_f32[image] - self.cadence_transformed_f32[image][:, nearest_columns])
            # self.cadence_transformed_f32[image] = abs(self.cadence_transformed_f32[image] - self.cadence_transformed_f32[image][:, nearest_columns2])

    def clean_image_rowwise(self) -> np.ndarray: # https://www.kaggle.com/tatamikenn/image-cleaning-column-wise-normalization-knn
        for image in range(self.cadence_transformed_f32.shape[0]):
            neigh = NearestNeighbors(n_neighbors=3, algorithm="brute", metric="minkowski", p=2)
            neigh.fit(self.cadence_transformed_f32[image])
            _, inds = neigh.kneighbors(self.cadence_transformed_f32[image])
            nearest_rows = inds[:, 1]
            nearest_rows2 = inds[:, 1]
            self.cadence_transformed_f32[image] = abs(self.cadence_transformed_f32[image] - self.cadence_transformed_f32[image][nearest_rows, :])
            self.cadence_transformed_f32[image] = abs(self.cadence_transformed_f32[image] - self.cadence_transformed_f32[image][nearest_rows2, :])

    def remove_identical_colors(self, tolerance, filter_size=1):
        num_images = self.cadence_transformed_f32.shape[0]
        num_rows = self.cadence_transformed_f32.shape[1]
        num_cols = self.cadence_transformed_f32.shape[2]
        # set identical pixels to None
        for row in range(num_rows):
            for col in range(num_cols):
                current_cells = []
                for image in range(num_images):
                    current_cells.append(self.cadence_transformed_f32[image][row][col])
                min_val = min(current_cells)
                max_val = max(current_cells)
                if max_val - min_val <= tolerance:
                    for image in range(num_images):
                        self.cadence_transformed_f32[image][row][col] = None
        self.set_nones_to_noise()

    def set_nones_to_noise(self, use_average=True):
        num_images = self.cadence_transformed_f32.shape[0]
        num_rows = self.cadence_transformed_f32.shape[1]
        num_cols = self.cadence_transformed_f32.shape[2]
        cadence_transformed_deep_copy = np.copy(self.cadence_transformed_f32)
        # set None pixels to average noise around pixel
        for image in range(num_images):
            for row in range(num_rows):
                for col in range(num_cols):
                    if np.isnan(self.cadence_transformed_f32[image][row][col]):
                        set_noise = False
                        noise_search_size = 1
                        safe_row_start = row
                        safe_col_start = col
                        safe_row_end = row
                        safe_col_end = col
                        while not set_noise:
                            row_start = row - noise_search_size if row - noise_search_size >= 0 else safe_row_start
                            col_start = col - noise_search_size if col - noise_search_size >= 0 else safe_col_start
                            row_end = row + noise_search_size if row + noise_search_size < num_rows else safe_row_end
                            col_end = col + noise_search_size if col + noise_search_size < num_cols else safe_col_end

                            safe_row_start = row_start
                            safe_col_start = col_start
                            safe_row_end = row_end
                            safe_col_end = col_end

                            current_noise_filter = self.cadence_transformed_f32[image][row_start:row_end, col_start:col_end]
                            num_nones = np.count_nonzero(np.isnan(current_noise_filter))
                            noise_search_filter_size = current_noise_filter.size
                            max_num_nones = math.ceil(noise_search_filter_size) / 2
                            if use_average:
                                noise = np.nanmean(current_noise_filter)
                            else:
                                noise = Counter(np.reshape(current_noise_filter, current_noise_filter.size)).most_common(1)[0][0]
                            if (num_nones < max_num_nones or noise_search_filter_size >= num_rows * num_cols) and not np.isnan(noise):
                                cadence_transformed_deep_copy[image][row][col] = noise # probably median is better
                                set_noise = True
                            else:
                                noise_search_size += 1
        self.cadence_transformed_f32 = cadence_transformed_deep_copy

    def safe_as_cv2_img(self, image_index):
        # test_cadence.map_to_colorscale()
        img = self.cadence_transformed_f32[image_index]
        cv2.imwrite('color_img.png', img)

    def cv2_canny(self, lower, upper):
        img = cv2.imread("color_img.jpg", cv2.IMREAD_GRAYSCALE)
        edges =cv2.Canny(img, lower, upper)
        plt.imshow(edges, cmap="gray")

    def cv2_filter(self, depth, kernel):
        img = cv2.imread("color_img.jpg", cv2.IMREAD_GRAYSCALE)
        gauss = cv2.filter2D(img, depth, kernel)
        plt.imshow(gauss, cmap="gray")

    def unify_similar_pixels(self, epsilon):
        # self.cadence_transformed_f32 = np.array([[
        #     [1.1, 2.1, 3.1, 4.1, 5.1], 
        #     [1.2, 2.2, 3.2, 4.2, 5.2], 
        #     [1.3, 2.3, 3.3, 4.3, 5.3], 
        #     [1.4, 2.4, 3.4, 4.4, 5.4], 
        #     [1.5, 2.5, 3.5, 4.5, 5.5]
        # ]])
        # self.cadence_transformed_f32 = np.array([[
        #     [1.1, 1.2, 1.3, 1.5, 1.9], 
        #     [1.2, 2.2, 3.2, 4.2, 5.2], 
        #     [1.3, 2.3, 3.3, 4.3, 5.3], 
        #     [1.4, 2.4, 3.4, 4.4, 5.4], 
        #     [1.5, 2.5, 3.5, 4.5, 5.5]
        # ]])
        num_images = self.cadence_transformed_f32.shape[0]
        num_rows = self.cadence_transformed_f32.shape[1]
        num_cols = self.cadence_transformed_f32.shape[2]
        epsilon = epsilon / 100 # => epsilon is given in percentage
        kernel_size = 2
        cadence_transformed_deep_copy = np.copy(self.cadence_transformed_f32)
        for image in np.arange(num_images):
            image_min = np.amin(self.cadence_transformed_f32[image])
            image_max = np.amax(self.cadence_transformed_f32[image])
            image_full_difference = image_max - image_min
            epsilon_cur_img = epsilon * image_full_difference
            for row in np.arange(num_rows - 1):
                for col in np.arange(num_cols - 1):
                    filter = self.cadence_transformed_f32[image][row:row+kernel_size, col:col+kernel_size]
                    current_pixel = self.cadence_transformed_f32[image][row][col]
                    for kernel_row in np.arange(0, kernel_size):
                        for kernel_col in np.arange(0, kernel_size):
                            possible_similar_pixel = filter[kernel_row][kernel_col]
                            if abs(possible_similar_pixel - current_pixel) <= epsilon_cur_img:
                                cadence_transformed_deep_copy[image][row + kernel_row][col + kernel_col] = cadence_transformed_deep_copy[image][row][col]
        self.cadence_transformed_f32 = cadence_transformed_deep_copy

    def map_to_proportional_colorscale(self, max=1000, min=0) -> None:
        colorscale_full_difference = abs(max - min)
        num_images = self.cadence_transformed_f32.shape[0]

        # occurrences_border = 1
        # for image in range(num_images):
        #     unique_values = np.unique(self.cadence_transformed_f32[image])
        #     for unique_value in unique_values:
        #         if np.count_nonzero(self.cadence_transformed_f32[image][self.cadence_transformed_f32[image] == unique_value]) <= occurrences_border:
        #             self.cadence_transformed_f32[image][self.cadence_transformed_f32[image] == unique_value] = None
        # # self.plot_cadence(True)
        # self.set_nones_to_noise(False)

        for image in range(num_images):
            unique, counts = np.unique(self.cadence_transformed_f32[image], return_counts=True)
            unique_values = dict(zip(unique, counts))
            # print("uniwuq: ", unique_values, "| keys: ", unique_values.keys())
            num_unique_values = len(unique_values.keys())
            distance_between = colorscale_full_difference / num_unique_values
            # factor = 10
            # index_array = np.arange(num_unique_values)
            # parts = 10
            # array_part = num_unique_values / parts
            # counter = 0
            # print("array-part: ", array_part)
            for index, unique_value in enumerate(unique_values.keys()):
                self.cadence_transformed_f32[image][self.cadence_transformed_f32[image] == unique_value] = distance_between * index

                # occurrence = unique_values[unique_value]
                # self.cadence_transformed_f32[image][self.cadence_transformed_f32[image] == unique_value] = (distance_between * (occurrence * factor)) * index

                # self.cadence_transformed_f32[image][self.cadence_transformed_f32[image] == unique_value] = distance_between * index_array[int((index % parts) * array_part + counter)]
                # print("| (index % parts): ", (index % parts), "| counter: ", counter,
                # "| int((index % parts) * array_part + counter): ", int((index % parts) * array_part + counter), "| len(index_array): ", len(index_array))
                # if index % parts == 0 and index != 0:
                #     counter += 1


    def map_similar_to_one(self, max=1024, min=0, similarity=0.05):
        colorscale_full_difference = abs(max - min)
        num_images = self.cadence_transformed_f32.shape[0]
        for image in range(num_images):
            unique_values = np.unique(self.cadence_transformed_f32[image])
            num_unique_values = len(unique_values)
            distance_between = colorscale_full_difference / num_unique_values
            is_similar = distance_between * similarity
            while len(unique_values) != 0:
                unique_value = unique_values[0]
                self.cadence_transformed_f32[image][np.logical_and(
                    self.cadence_transformed_f32[image] <= unique_value + is_similar,
                    self.cadence_transformed_f32[image] >= unique_value - is_similar
                )] = unique_value
                unique_values = np.delete(unique_values, np.where(np.logical_and(
                    unique_values <= unique_value + is_similar,
                    unique_values >= unique_value - is_similar
                )))

    def map_part_to_proportional(self, part):
        parts = 4
        num_images = self.cadence_transformed_f32.shape[0]
        for image in range(num_images):
            unique_values = np.unique(self.cadence_transformed_f32[image])
            unique_part_values = np.array_split(unique_values, parts)[part]
            if part != 0 and part != parts - 1:
                self.cadence_transformed_f32[image][np.logical_or(
                    self.cadence_transformed_f32[image] > unique_part_values[len(unique_part_values) - 1],
                    self.cadence_transformed_f32[image] < unique_part_values[0]
                )] = -1000
            elif part == 0:
                self.cadence_transformed_f32[image][
                    self.cadence_transformed_f32[image] > unique_part_values[len(unique_part_values) - 1],
                ] = -1000
            elif part == parts - 1:
                self.cadence_transformed_f32[image][
                    self.cadence_transformed_f32[image] < unique_part_values[0]
                ] = -1000
            self.map_to_proportional_colorscale()

    def stress_vertical_lines(self, filter_size, epsilon):
        to_add = math.ceil(filter_size / 2) - 1
        new_cadence_transformed = self.add_0_border(to_add)
        images = new_cadence_transformed.shape[0]
        rows = new_cadence_transformed.shape[1]
        cols = new_cadence_transformed.shape[2]

        for image in range(images):
            for i in range(rows - 2*to_add):
                for j in range(cols - 2*to_add):
                    current = new_cadence_transformed[image][i:i+filter_size, j:j+filter_size]
                    current_mid = int(filter_size / 2)
                    current_filter_values = []
                    for filter_row in range(filter_size):
                        current_filter_value = current[filter_row][current_mid]
                        current_filter_values.append(current_filter_value)
                    for filter_value in range(len(current_filter_values)):
                        if filter_value != len(current_filter_values) - 1:
                            if abs(current_filter_values[filter_value] - current_filter_values[filter_value + 1]) > epsilon:
                                vertical_value = -1000
                                break
                        else:
                            vertical_value = 1000
                    self.cadence_transformed_f32[image][i, j] = vertical_value

    def line_detection(self): # from stackoverflow
        # img = cv2.imread("test_hough_lines.png")
        img = cv2.imread("color_img.png")
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        # blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        low_threshold = 40
        high_threshold = 150
        # edges = cv2.Canny(gray, low_threshold, high_threshold)
        # cv2.imwrite("canny_edges.png", edges)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 20  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(gray, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
        
        lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        cv2.imwrite('lines_edges.png', lines_edges)

    def gradienten_richtung(self):
        img = cv2.imread("color_img.png")
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        cv2.imwrite("sobelx.png", sobelx)
        cv2.imwrite("sobely.png", sobely)

    def cv(self):
        # self.gradienten_richtung()
        # self.line_detection()
        self.convolution_filter(filter_x_sobel)
        self.map_similar_to_one() # reduces noise
        # self.unify_similar_pixels(0.1)
        self.map_to_proportional_colorscale() # reduces noise
        self.plot_cadence_for_comparing_differences()