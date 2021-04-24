import argparse
import numpy as np
import cv2
import math

import pageBordering
import lineRemoval
import circleRemoval
import connectedComponentsProcessing
import lineClustering
import lineAnalysis



def preprocess(image_path, save_path):

    # Load the image
    img = cv2.imread(image_path)

    # Create ImageAnalysis class
    image_anlysis = ImageAnalysis(img)

    # Get word images
    return image_anlysis.get_words(save_path=save_path)

def img_resize(img, scale=0.5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dsize = (width, height)
    return cv2.resize(img, dsize)

class ImageAnalysis:
    
    def __init__(self, img):
        self.img = img

    # Preprocess the image
    def preprocess(self):
        self.cleaned = self.clean_page()
        self.gray = cv2.cvtColor(self.cleaned, cv2.COLOR_BGR2GRAY)
        self.blurred = cv2.medianBlur(self.gray, 5)
        self.edges()
        self.dilate()

    # Demo the handwritting recogniction software
    def get_words(self, save_path=None):
        self.preprocess()
        self.components, self.components_img = connectedComponentsProcessing.connected_components(self.dilated)
        self.line_components, self.line_img = lineClustering.line_clustering(self.components)
        self.word_imgs, self.line_imgs = lineAnalysis.get_words_in_line(self.cleaned, self.dilated,
                                                                        self.components, self.line_components)

        if save_path is not None:
            self.save_images(save_path)

        return self.word_imgs

    # Performs canny
    def edges(self):
        c = (40, 50)
        self.edges = cv2.Canny(self.blurred, c[0], c[1], apertureSize=3)

    # Performs dilation
    def dilate(self, save=None):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(4, 4))
        self.dilated = cv2.dilate(self.edges, kernel)

    # Performs page cleaning
    def clean_page(self):
        # Border the image to page
        status, self.bordered = pageBordering.page_border(self.img.copy())

        # Removes page holes
        self.circles_removed = circleRemoval.circle_removal(self.bordered.copy())

        # Remove lines on lined paper (to an extent)
        # Repeating line removal gives better results up to 3 iterations
        self.lines_removed = self.circles_removed
        for i in range(3):
            self.lines_removed, self.gray = lineRemoval.lines_removal(self.lines_removed)
        
        return self.lines_removed


    # Saves pre-processing images
    def save_images(self, save_path):
        cv2.imwrite(save_path + "/0img.jpg", img_resize(self.img))
        cv2.imwrite(save_path + "/1bordered.jpg", img_resize(self.bordered))
        cv2.imwrite(save_path + "/2circles_removed.jpg", img_resize(self.circles_removed))
        cv2.imwrite(save_path + "/3lines_removed.jpg", img_resize(self.lines_removed))
        cv2.imwrite(save_path + "/4gray.jpg", img_resize(self.gray))
        cv2.imwrite(save_path + "/5blurred.jpg", img_resize(self.blurred))
        cv2.imwrite(save_path + "/6edges.jpg", img_resize(self.edges))
        cv2.imwrite(save_path + "/7dilated.jpg", img_resize(self.dilated))
        cv2.imwrite(save_path + "/8components.jpg", img_resize(self.components_img))
        cv2.imwrite(save_path + "/9components_filtered.jpg", img_resize(self.components.filtered))
        cv2.imwrite(save_path + "/10components_borders.jpg", img_resize(self.components.borders))
        cv2.imwrite(save_path + "/11lines_img.jpg", img_resize(self.line_img))

        for i, l in enumerate(self.line_imgs):
            cv2.imwrite(save_path + "/line{}.jpg".format(i), self.line_imgs[i])

        for i, l in enumerate(self.word_imgs):
            for j, w in enumerate(l):
                cv2.imwrite(save_path + "/word{}_{}.jpg".format(i, j), w[0])
                cv2.imwrite(save_path + "/word{}_{}_backup.jpg".format(i, j), w[1])



if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Load image from path')
    parser.add_argument('-s', '--save', type=str, help='Save path for images', default=None)
    args = parser.parse_args()

    # Get word images
    word_imgs = preprocess(args.image, args.save)
