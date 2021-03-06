# CITS4402 Project - LRC for Face Recognition & Classification

This **README** file addresses the details behind the face recognition and classification project algorithm, as well as the GUI and supporting files submitted for this project.

## Folder Contents

The submission folder should contain the following, with the exception of the `FaceDataset` folder:

```
CITS4402-Project
├── README.md                           <-- The file you are currently reading.                     
├── project_app.m                       <-- The project application '.m' file, containing all the project code.
├── s41/                                <-- Our additional, custom class, containing 10 image files.
└── FaceDataset/                        <-- The folder containing all the facial images.
     ├── s1/                            <-- The name of the first class.
     │    ├── 1.pgm                     <-- The first image in class "s1".
     │    ├── 2.pgm                     <-- The second image in class "s1".
     │    ├── 3.pgm                     <-- The third image in class "s1".
     │    ├── 4.pgm                     <-- The fourth image in class "s1".
     │    ├── 5.pgm                     <-- The fifth image in class "s1".
     │    ├── 6.pgm                     <-- The sixth image in class "s1".
     │    ├── 7.pgm                     <-- The seventh image in class "s1".
     │    ├── 8.pgm                     <-- The eighth image in class "s1".
     │    ├── 9.pgm                     <-- The ninth image in class "s1".
     │    └── 10.pgm                     <-- The tenth image in class "s1".
     │
     ├── s2/                            <-- The name of the second class.
     │    ├── 1.pgm                     <-- The first image in class "s2".
     │    ├── 2.pgm                     <-- The second image in class "s2".
     │    ├── 3.pgm                     <-- The third image in class "s2".
     │    └── ...                       <-- All other images in class "s2", of which there should be 10 total.
     │
     ├── s3/                            <-- The name of the third class.
     │    ├── ...                       <--- The 10 images which fall into class "s3".
     ├── ...                            <-- As many classes as there are unique faces in the dataset.
```

## Execution Procedure

To run this GUI...

1. Load and execute the `project_app.m` file in MATLAB.
2. Click on `Load Classes` (the left-hand button) to load the directory containing all the images, organized into classes as per the **Folder Contents** section given above (the `FaceDataset` folder).
3. Set the `Training Images per Class` and the `Delay Timeout for Display (Seconds)` sliders, then click on `Classify Test Images` to run the program.

## Notes

Some important information needs to be touched on.

1. All the images are greyscale, with 112 rows and 92 columns of pixels, and are of the `.pgm` file type. When adding new images, please ensure that they are **also** greyscale and conform to the specified dimensions and filetype.
2. Each class has 10 images. When introducing a new class, please ensure that it has **at least** 10 images.
3. Please follow the naming convention given in the `FaceDataset` folder. Each class has the name `s<num>` where `<num>` is the next consecutive number from the previous class. The program will assign these numbers to the corresponding classes.

## The Method

This application implements the Linear Regression for Face Recognition algorithm found in section 2.1 of \[1\]. We refer readers to this paper for the full details of the algorithm and limit ourselves to a discussion of the implementation specifics.

- We used a very simple downsampling procedure. We sampled every 4th row and column of each image to create a downsized image with 28 rows and 23 columns of pixels. Furthermore, the values of each pixel in the downsized (and stacked) image column vector were normalized by simply dividing them by 255, the maximum possible value any pixel could have before normalization. These transformations were performed in a function so as to promote reusable code between training and test image processing.
- We opt not to derive the column vector of parameters for generating the projection of each test image onto the relevant class subspsace, instead opting to generate the projection directly using equation 4 of \[1\]. We do not store the projection for each subspace either, instead, we calculate the distance from that projection to the test image and store that value in a distance array with indices corresponding to the class number.
- The 10 images in each class are partitioned into training and test splits based on a value specified by the user via a slider. If, for example, the split is 5 training - 5 test (the default values), then the first 5 images in each class form the training split, and the last 5 images form the test split. We disallow the user from specifying any less than **1** image in the training set for each class.

## The Graphical User Interface (GUI)

<image src="https://cdn.discordapp.com/attachments/823141543195050017/843376775266172958/gui_screenshot.png">

The above is an image of my application in the middle of executing the LRC algorithm. Observe the following:

1. I can see that 41 classes have been loaded, comprising a total of 410 images. Each image is of size 112 x 92, and after downsampling, 28 x 23.
2. I have specified that there should be only **5** images per class in the training split for this particular execution. Because of this, my application is telling me that out of the 410 images loaded, 205 of them are in the training set, and 205 are in the test set.
3. I've specified the delay timeout as **1 second**. This means that every time the GUI loads a new image, it will wait 1 second after the execution of the algorithm, image and metrics display, before it moves onto the next test image.
4. This particular test image belonged to class `s16` and was correctly classified to `s16` based on the provided *minimum* class distance. Based on the result of this classification, the current accuracy (of all images up to and including this one), as well as the count of images classified, has been updated.

## Conclusion & Results

As is to be expected, the LRC algorithm for face classification implemented in this program performed better the greater the number of training instances per class. This is most likely due to the fact that there are more images with which a linear combination can be formed which resembles the test image.

Out of the 40 default classes provided for this project, class `s1` was the most frequently misclassed; in fact, regardless of whether there were 1 or 9 training images per class, images from `s1` were very rarely classed correctly, and the first image displayed from this class was consistently incorrectly assigned a different class label.

The most frequent incorrect label was class `s9`. This means that the images that were incorrectly classed were most likely assigned the `s9` class label. This label was most prominently incorrectly assigned to class `s1` and to other classes when there were less than 7 images in the training set.

We experimented with one additional class using face images from a member of our group. The additional of this extra class did not substantially alter the accuracy of our classifier. The new class can be found in folder `s41`.

## Project Members

- [Lachlan D Whang](https://github.com/ForsakenIdol)
- [Seamus Mulholland-Patterson](https://github.com/Seamooo)
- [Haolin Wu](https://github.com/Dragonite)

## References

\[1\] Naseem, Imran, Roberto Togneri, and Mohammed Bennamoun. "Linear regression for face recognition." IEEE transactions on pattern analysis and machine intelligence 32.11 (2010): 2106-2112.
