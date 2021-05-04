import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

import pageBordering
import lineRemoval
import circleRemoval
import connectedComponentsProcessing
import lineClustering
import wordAnalysis

def img_resize(img, scale):
    """Resize an image, keeping aspect ratio, according to a decimal scale percentage."""
    if scale == 1:
        return img

    dsize = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, dsize)


def restricted_float(x):
    """
    Takes an argparsed argument x and ensures it is a float in the range (0, 1].
    Credit: https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def save_intermediate_img(config, img, name, plot=False):
    """Save an intermediate preprocessing image given the config fields."""
    if config['inter']:
        path = config['inter_path'] + '/' + str(config['inter_saved']) + name + ".png"

        if not plot:
            cv2.imwrite(path, img_resize(img, scale=config['inter_scale']))
        else:
            plt.savefig(path)
            plt.figure(config['inter_saved'])
        
        config['inter_saved'] += 1


def get_canny(img, c=(40, 50), apertureSize=3):
    """Perform canny edge detection."""
    return cv2.Canny(img, c[0], c[1], apertureSize=apertureSize)


class ProcessedPage:
    """
    A class to handle the preprocessing pipeline and final output.

    Attributes
    ----------
    config : dict
        A dictionary describing a data loading/saving configuration.
    img : np.ndarray
        The original image loaded from file.
    cleaned : np.ndarray
        A cleaned image of img.
    canny : np.ndarray
        A canny edges image of cleaned.
    lines : list[Line]
        A list of Line objects containing information about each line.

    Methods
    -------
    preprocess_page():
        Preprocess the image into a cleaned image and a canny image.
    
    clean_page():
        Cleans the image by bordering it and removing any page holes/lines.

    get_words():
        Creates lines and word classes and images.
    """

    def __init__(self, config):
        self.config = config
        self.img = cv2.imread(config['image'])
        self.config['save_inter_func'](self.config, self.img, "img")
        self.cleaned, self.canny = self.preprocess_page()
        self.get_words()


    def preprocess_page(self):
        """Preprocess the image into a cleaned image and a canny image."""
        
        # Clean the image
        cleaned = self.clean_page()
        
        # Convert to grayscale
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        self.config['save_inter_func'](self.config, gray, "gray")

        # Blur the gray-scale image
        blurred = cv2.medianBlur(gray, 5)
        self.config['save_inter_func'](self.config, blurred, "blurred")

        # Perform canny edge detection
        canny = get_canny(blurred)
        self.config['save_inter_func'](self.config, canny, "canny")
        
        return cleaned, canny


    def clean_page(self):
        """Cleans the image by bordering it and removing any page holes/lines."""
        # Border the image to page
        error, bordered = pageBordering.page_border(self.img.copy())
        if error:
            raise Exception("The image provided could not be bordered.")
        self.config['save_inter_func'](self.config, bordered, "bordered")

        # Removes page holes
        holes_removed = circleRemoval.page_hole_removal(bordered)
        self.config['save_inter_func'](self.config, holes_removed, "holes_removed")

        # Remove lines on lined paper (repeating for multiple iterations gives better results)
        lines_removed = holes_removed
        for i in range(3):
            lines_removed, gray = lineRemoval.lines_removal(lines_removed)
        self.config['save_inter_func'](self.config, lines_removed, "lines_removed")

        return lines_removed

    def get_words(self):
        """Creates and returns word images."""
        components = connectedComponentsProcessing.connected_components(self.canny, self.config)
        line_components = lineClustering.line_clustering(components, self.config)
        self.lines = wordAnalysis.get_words_in_line(self.cleaned, components, \
            line_components, self.config)

        # To save all word images, iterate lines, words in a line, and images corresponding to a word
        if self.config['words_path'] is not None:
            for i, line in enumerate(self.lines):
                for j, word in enumerate(line.words):
                    for k, img in enumerate(word.images):
                        cv2.imwrite(self.config['words_path'] + "/word{}_{}-{}.jpg".format(i, j, k), img)


def preprocess(image_path, words_path, intermediate_path, scale=1):
    """
    Preprocess an image and return the word images.

    Parameters:
        image_path (str): Path to the image to preprocess
        save_path (str): Path to save preprocessing-related images as well as word images
        intermediate_path (str), optional: Path to save intermediate preprocessing images

    Returns:
        (list): A list of lines, where each element is a list of word images in each line list
    """
    config = dict({'image': image_path, 'words_path': words_path, \
        'inter_path': intermediate_path, 'inter_saved': 0, \
        'inter': intermediate_path != None, 'inter_scale': scale, \
        'save_inter_func': save_intermediate_img})

    return ProcessedPage(config)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Load image from path')
    parser.add_argument('-w', '--words', type=str, help='Save path for word images', \
        default=None)
    parser.add_argument('-i', '--intermediate', type=str, \
        help='Save path for intermediate preprocessing images', default=None)
    
    parser.add_argument('-s', '--scale', type=restricted_float, \
        help='Scale of saved intermediate preprocessing images', default=1)

    args = parser.parse_args()

    # Get word images
    processed = preprocess(args.image, args.words, args.intermediate, scale=args.scale)