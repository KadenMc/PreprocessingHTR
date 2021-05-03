import numpy as np
import cv2
import math

def draw_border(img, border, col=255):
    """Draw a border on an image given the border coordinates."""
    l, r, t, b = border
    cv2.line(img, (l, t), (l, b), col, 2)
    cv2.line(img, (l, t), (r, t), col, 2)
    cv2.line(img, (l, b), (r, b), col, 2)
    cv2.line(img, (r, t), (r, b), col, 2)


def show_connected_components(img):
    """
    Displays connected components colorfully.
    Credit: https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
    
    Parameters:
        img (np.ndarray): The image for which to show the connected components
    """
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



def consecutive_groups(data, stepsize=1):
    """Finds groups of consequtive numbers."""
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def mean_top_k(areas, k=10):
    """Get mean of the top k elements."""
    top_k_vals = min([k, len(areas)])
    return (-np.sort(-areas))[:top_k_vals].mean()


class Components():
    """
    Class to organize connected components and related statistics.

    Methods
    -------
    filter1():
        Filters components based on height, horizontal ratio, and small area.

    filter_strays(y):
        Filters 'stray' components by according to y-value/closeness to other components.

    filter2():
        Filters components based on closeness to other components (stray) and smaller area.

    filter(config):
        Filters components.
    """

    def __init__(self, img):
        self.img = img
        self.nb_components, self.output, self.stats, self.centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        self.nb_components -= 1

        self.left = self.stats[1:, 0]
        self.top = self.stats[1:, 1]
        self.width = self.stats[1:, 2]
        self.height = self.stats[1:, 3]
        self.area = self.stats[1:, 4]
        self.bounding_area = self.width * self.height
        self.right = self.left + self.width
        self.bottom = self.top + self.height

        self.x = self.left + self.width//2
        self.y = self.top + self.height//2

        self.diagonal = math.sqrt(
            math.pow(self.output.shape[0], 2) + math.pow(self.output.shape[1], 2))

    def __len__(self):
        return len(self.x)

    def bounding_boxes(self):
        """Draws bounding box for every component."""
        borders = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        for i in range(len(self.x)):
            draw_border(borders, (self.left[i], self.right[i], self.top[i], self.bottom[i]), col=255)

        self.bounding_rect = np.stack([self.x, self.y, self.width, self.height], axis=1)
        return borders

    def filter_indices(self, allowed):
        """Filters statistics by indices in allowed."""
        self.left = self.left[allowed]
        self.right = self.right[allowed]
        self.top = self.top[allowed]
        self.bottom = self.bottom[allowed]
        self.area = self.area[allowed]
        self.x = self.x[allowed]
        self.y = self.y[allowed]
        self.width = self.width[allowed]
        self.height = self.height[allowed]
        self.bounding_area = self.bounding_area[allowed]

    def filter1(self):
        """Filters components based on height, horizontal ratio, and small area."""
        # Find minimum area under which we can delete components
        self.min_area = mean_top_k(self.area, k=15)/1.8

        # Use bounding box area to get rid of noise/very small components
        allowed_area = np.argwhere(self.bounding_area > self.min_area)[:, 0]


        # The average paper has up to ~35 lines of text.
        # This divides the page into 35 lines, which implies that a text contour should have
        # height no more than img.height/35. To be generous, allow lines 3 times bigger than this
        allowed_height = np.argwhere(self.height <= (self.img.shape[0]/35)*3)[:, 0]
        allowed = np.intersect1d(allowed_area, allowed_height)

        # Getting rid of the remnants of horizontal lines can be done via a height to width ratio
        # If width/height is VERY high, it must be a horizontal line (and the area can't be too large)
        allowed_horizontal_ratio = np.argwhere(np.logical_or(self.width/self.height < 5, self.bounding_area >= self.min_area*4))[:, 0]
        self.allowed = np.intersect1d(allowed, allowed_horizontal_ratio)

        # Note: In order to draw the components from the original connected components output
        # we must track which components we're 'allowing', or keeping
        self.filter_indices(self.allowed)


    def filter_strays(self, y):
        """Filters 'stray' components by according to y-value/closeness to other components."""
        counts, boundaries = np.histogram(y, bins=40)

        # Find runs of non-zero counts
        non_zero = np.argwhere(counts != 0)[:, 0]
        consecutive = consecutive_groups(non_zero)

        check_boundaries = []
        indices = []
        for c in consecutive:
            # Check consequtive interval length
            if len(c) <= 3:
                # Check number of components in consequtive interval
                if counts[c].sum() <= 4:
                    for b in c:
                        indices.extend(np.argwhere(np.logical_and(y >= boundaries[b], y <= boundaries[b + 1]))[:, 0])
        
        return np.array(indices)


    def filter2(self):
        """Filters components based on closeness to other components (stray) and smaller area."""
        # Get components with 'small enough' area - (average of top k areas)/1.5
        small_area_indices = np.argwhere(self.area <= mean_top_k(self.area, k=15)/1.5)

        # Get stray components
        stray_indices = self.filter_strays(self.y)

        # Combine filtering - If small area and a stray, then get rid of it!
        remove = np.intersect1d(small_area_indices, stray_indices)

        # Get 'allowed' indices - compliment of removed
        allowed = np.setdiff1d(np.array([i for i in range(len(self.area))]), remove)
        self.allowed = self.allowed[allowed]

        # Note: In order to draw the components from the original connected components output
        # we must track which components we're 'allowing', or keeping
        self.filter_indices(allowed)


    def filter(self, config):
        """Filters components."""
        # Filters based on height, horizontal ratio, and very small area
        self.filter1()

        # Draw connected components after filter1
        self.filtered = np.zeros((self.img.shape))
        for i in range(len(self.allowed)):
            self.filtered[self.output == self.allowed[i] + 1] = 255

        # Draw bounding boxes after filter1
        self.borders = self.bounding_boxes()

        # Save intermediate images
        config['save_inter_func'](config, self.filtered, "components_filtered1")
        config['save_inter_func'](config, self.borders, "components_borders1")


        # Filters based on closeness to other components (stray) and smaller area
        self.filter2()
        
        # Draw connected components after filter2
        self.filtered = np.zeros((self.img.shape))
        for i in range(len(self.allowed)):
            self.filtered[self.output == self.allowed[i] + 1] = 255

        # Draw bounding boxes after filter2
        self.borders = self.bounding_boxes()

        config['save_inter_func'](config, self.filtered, "components_filtered1")
        config['save_inter_func'](config, self.borders, "components_borders1")


# Creates connected components
def connected_components(img, config):
    """Create, visualize, filter, and return connected components."""
    if config['inter']:
        # Save a connected components display image
        components_labeled_img = show_connected_components(img)
        config['save_inter_func'](config, components_labeled_img, "components_labeled")
    
    # Create, filter, and return connected components
    components = Components(img)
    components.filter(config)
    return components