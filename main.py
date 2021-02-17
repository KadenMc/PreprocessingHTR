import os
import sys
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
    def demo(self, save=False, save_path=None, predict=True):
        self.preprocess()

        self.components, self.components_img = connectedComponentsProcessing.connected_components(self.dilated)
        self.line_components, self.line_img = lineClustering.line_clustering(self.components)
        self.word_imgs, self.line_imgs = lineAnalysis.get_words_in_line(self.cleaned, self.dilated, self.components, self.line_components)

        if save:
            self.save_images(save_path)

        if predict:
            text = self.get_text()
            return text

        return None

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

    # NOTE: Change to work with your word prediction model
    def get_model(self):
        #curr_path = os.getcwd()
        #sys.path.append(curr_path + '\..\SimpleHTR-master\src')
        #import customInfer
        #return customInfer.get_model(curr_path)
        raise NotImplementedError

    # NOTE: Change to work with your word prediction model
    def predict_img(self, model, img):
        #sys.path.append(os.getcwd() + '../SimpleHTR-master/src')
        #import customInfer
        #return customInfer.infer(model, img)
        raise NotImplementedError

    # NOTE: Change to work with your word prediction model
    def get_text(self):
        # Load word prediction model
        model = self.get_model()
        text = ''
        for i, l in enumerate(self.word_imgs):
            for j, w in enumerate(l):
                # The model fails on some images for some unknown reason, so we can
                # use an image which was thresholded differently which sometimes works
                try:
                    word, prob = self.predict_img(model, w[0])
                    text += word + ' '
                except:
                    try:
                        word, prob = self.predict_img(model, w[1])
                        text += word + ' '
                    except:
                        print("Unable to predict word {0} on line {1}".format(i, j))
            text += '\n'
        return text[:-2]

    # Saves an image
    def save_image(self, img, label):
        cv2.imwrite(label, img)

    # Saves pre-processing images
    def save_images(self, save_path):
        cv2.imwrite(save_path + "/0img.jpg", self.img)
        cv2.imwrite(save_path + "/1bordered.jpg", self.bordered)
        cv2.imwrite(save_path + "/2circles_removed.jpg", self.circles_removed)
        cv2.imwrite(save_path + "/3lines_removed.jpg", self.lines_removed)
        cv2.imwrite(save_path + "/4gray.jpg", self.gray)
        cv2.imwrite(save_path + "/5blurred.jpg", self.blurred)
        cv2.imwrite(save_path + "/6edges.jpg", self.edges)
        cv2.imwrite(save_path + "/7dilated.jpg", self.dilated)
        cv2.imwrite(save_path + "/8components.jpg", self.components_img)
        cv2.imwrite(save_path + "/9components_filtered.jpg", self.components.filtered)
        cv2.imwrite(save_path + "/10components_borders.jpg", self.components.borders)
        cv2.imwrite(save_path + "/11lines_img.jpg", self.line_img)

        for i, l in enumerate(self.line_imgs):
            cv2.imwrite(save_path + "/line{}.jpg".format(i), self.line_imgs[i])

        for i, l in enumerate(self.word_imgs):
            for j, w in enumerate(l):
                cv2.imwrite(save_path + "/word{}_{}.jpg".format(i, j), w[0])
                cv2.imwrite(save_path + "/word{}_{}_backup.jpg".format(i, j), w[1])

def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('image', type=str, help='Load image from path')

    # Optional arguments
    parser.add_argument('-s', '--save', type=str, help='Save path for images', default=None)
    parser.add_argument('-p', '--predict', help='Predict words in image (if implemented)',
                        action='store_true', default=False)

    args = parser.parse_args()

    # Load the image
    img = cv2.imread(args.image)
    image_anlysis = ImageAnalysis(img)

    pred = image_anlysis.demo(save=True, save_path='processed', predict=args.predict)
    if args.predict:
        print("Predicted:\n")
        print(pred)

if __name__ == "__main__":
    main()
