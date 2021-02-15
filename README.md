# PreprocessingHTR
Pre-processing handwritten pages into words for Handwritten Text Recognition (HTR).

Pre-processing system takes the image of a full, handwritten page and returns cleaned images of individual words. These individual word images can then be fed into a Handwritten Text Recognition (HTR) system, which often prefers individual words.

![img](doc/w11.jpg)

## Run pre-processing
```
> python main.py test.jpg --save processed
```
### Command line arguments
* `image`: the path to the input image
* `--save`: the path to which images of pre-processing steps will be saved.
* `--predict`: a boolean representing whether to predict the images when calling demo - not currently implemented.


## Pre-processing Walkthrough

## Original image

![img](doc/0img.jpg)

## Bordered image

![bordered](doc/1bordered.jpg)

## Page holes removed

![circles_removed](doc/2circles_removed.jpg)

## Lines removed

![lines_removed](./doc/3lines_removed.jpg)

## Grayscale

![gray](./doc/4gray.jpg)

## Blurred

![blurred](./doc/5blurred.jpg)

## Edges

![edges](./doc/6edges.jpg)

## Dilated edges

![dilated](./doc/7dilated.jpg)

## Connected components

![components](./doc/8components.jpg)

## Connected components filtered

![components_filtered](./doc/9components_filtered.jpg)

## Connected component borders

![components_borders](./doc/10components_borders.jpg)

## Text lines

![lines_img](./doc/11lines_img.jpg)


## Individual text lines

![lines0](./doc/line0.jpg)
![lines1](./doc/line1.jpg)
![lines2](./doc/line2.jpg)
![lines3](./doc/line3.jpg)
![lines4](./doc/line4.jpg)
![lines5](./doc/line5.jpg)
![lines6](./doc/line6.jpg)


## Individual words




## Short-comings

Colored pen could be detected and extracted much more easily than pencil, however this isn't currently being taken advantage of.