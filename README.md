# Cancer-Cell-Segmentation
Segmentation and Counting of Fibroblast(green) and Colon Cancer Cells(blue) in Confocal 3D image dataset using Image Processing Techniques (OPENCV)
There are 4 folders of images consisting of an amalgamation of Fibroblast and Colon Cancer Cells at various ratios  ( 1_2,1_1,2_0.5,2_1)
at horizontal and vertical cross-sections.

For getting access to the images - ping me on ritikxman@gmail.com
The code is written in Python and OpenCV has been used .

Image Processing Algorithm - 
1) Pre-Processing - Noise Removal , Opening .
2) Convert of Image Color Space from BGR to HSV for color segmentation .
3) Forming of Mask and applying the mask on original image using AND operation.
4) Convert Image back to RGB and then to Binary B/W.
5) Use Dilation for background Exctraction.
6) Use Distance Transform with thresholding for foreground extraction.
7) Mark the various objects with labels.
8) Apply Watershed Algorithms for counting of cells.

green.py - counts and segments green fibroblast cells for all the images in a folder so that we can get a distribution pattern among various image slices.
blue.py - counts and segments green fibroblast cells for all the images in a folder so that we can get a distribution pattern among various image slices.
