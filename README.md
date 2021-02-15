# PreprocessingHTR
Pre-processing handwritten pages into words for Handwritten Text Recognition (HTR).

Pre-processing system takes the image of a full, handwritten page and returns cleaned images of individual words. These individual word images can then be fed into a Handwritten Text Recognition (HTR) system, which often prefers individual words.

## Run pre-processing
```
> python main.py test.jpg --save processed
```
### Command line arguments
* `image`: the path to the input image
* `--save`: the path to which images of pre-processing steps will be saved.
* `--predict`: a boolean representing whether to predict the images when calling demo - not currently implemented.




![img](./doc/0img.png)

![bordered](./doc/1bordered.png)

![circles_removed](./doc/2circles_removed.png)

![lines_removed](./doc/3lines_removed.png)

![gray](./doc/4gray.png)

![blurred](./doc/5blurred.png)

![edges](./doc/6edges.png)

![dilated](./doc/7dilated.png)

![components](./doc/8components.png)

![components_filtered](./doc/9components_filtered.png)

![components_borders](./doc/10components_borders.png)

![lines_img](./doc/11lines_img.png)