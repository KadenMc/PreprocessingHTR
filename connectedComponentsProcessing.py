import numpy as np
import cv2
import math

# Draw a border given coordinates
def draw_border(img, border, col=255):
    l, r, t, b = border
    cv2.line(img, (l, t), (l, b), col, 1)
    cv2.line(img, (l, t), (r, t), col, 1)
    cv2.line(img, (l, b), (r, b), col, 1)
    cv2.line(img, (r, t), (r, b), col, 1)

# Displays connected components
def show_connected_components(img):
    ret, labels = cv2.connectedComponents(img, connectivity=8)
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Convert to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # Set bg label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img

# Class for cv2's connected components
class Components():
    def __init__(self, img):
        # Get connected compontents & related metrics
        self.img = img
        self.nb_components, self.output, self.stats, self.centroids = cv2.connectedComponentsWithStats(
            img, connectivity=8)
        self.nb_components -= 1

        #self.show_connected_components()

        self.left = self.stats[1:, 0]
        self.top = self.stats[1:, 1]
        self.width = self.stats[1:, 2]
        self.height = self.stats[1:, 3]
        self.area = self.stats[1:, 4]
        self.right = self.left + self.width
        self.bottom = self.top + self.height

        self.x = self.left + self.width//2
        self.y = self.top + self.height//2

        self.diagonal = math.sqrt(
            math.pow(self.output.shape[0], 2) + math.pow(self.output.shape[1], 2))

    def __len__(self):
        return len(self.x)

    def filter(self, min_area=500):
        # Find minimum area under which we can delete components
        # TODO: min_area must be made variable
        '''sorted_arr = np.sort(area)
        diff_arr = np.diff(sorted_arr)
        smooth_factor = int(len(diff_arr) * 0.05)
        smooth_diff_arr = smooth(diff_arr, smooth_factor)
        indices = np.argwhere(smooth_diff_arr <= 5)
        min_size = sorted_arr[indices[-1]][0]'''
        self.min_area = min_area
        self.filtered = np.zeros((self.img.shape))

        # Use bounding boxes area to get rid of noise
        self.bounding_area = self.width*self.height
        allowed_area = np.argwhere(self.bounding_area > self.min_area)[:, 0]

        # The average paper has up to ~35 lines of text.
        # This divides the page into 35 lines, which implies that a text contour should have
        # height no more than img.height/35. To be generous, allow lines 5 times bigger than this
        allowed_height = np.argwhere(self.height <= (self.filtered.shape[0]/35)*5)[:, 0]
        self.allowed = np.intersect1d(allowed_area, allowed_height)

        # Getting rid of the remnants of horizontal lines can be done via a height to width ratio
        # If width/height is VERY high, it must be a horizontal line (and the area can't be too large)
        allowed_horizontal_ratio = np.argwhere(np.logical_or(self.width/self.height < 5, self.bounding_area >= self.min_area*4))[:, 0]
        self.allowed = np.intersect1d(self.allowed, allowed_horizontal_ratio)

        self.left = self.left[self.allowed]
        self.right = self.right[self.allowed]
        self.top = self.top[self.allowed]
        self.bottom = self.bottom[self.allowed]
        self.area = self.area[self.allowed]
        self.x = self.x[self.allowed]
        self.y = self.y[self.allowed]
        self.width = self.width[self.allowed]
        self.height = self.height[self.allowed]
        self.bounding_area = self.bounding_area[self.allowed]

        # Draw components with acceptable area and height
        for i in range(len(self.allowed)):
            self.filtered[self.output == self.allowed[i] + 1] = 255

    def bounding_boxes(self):
        self.borders = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)

        for i in range(len(self.x)):
            draw_border(self.borders, (self.left[i], self.right[i], self.top[i], self.bottom[i]), col=255)

        self.bounding_rect = np.stack([self.x, self.y, self.width, self.height], axis=1)

# Creates connected components
def connected_components(dilated, save=None):
    components = Components(dilated)
    components.filter()
    components.bounding_boxes()
    display_img = show_connected_components(dilated)
    return components, display_img
