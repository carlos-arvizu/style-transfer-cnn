# Style Transfer using Convolutional Neural Networks

This is a **Python3** / **PyTorch** implementation of [Image Style Transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), based on the paper:

**Image Style Transfer Using Convolutional Neural Networks**
by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.

The purpose is to transfer the style of one image (a) to another (b), while preserving the content structure of (b) but adopting the visual style of (a).

We leverage the pre-trained algorithm VGG19 model (available in PyTorch and trained on ImageNet) and extract outputs layers to define a cost function that merges content and style features.

**Diagram of the model**

![Diagram of the modified model](images/diagram.png)

*Image taken from the paper*

## Result

**Image used for content**

<img src="examples/content/bellas_artes.jpg" alt="Content image" title="Content image" width="600" height="auto">

**Image used for style**

<img src="examples/style/starry_night.jpg" alt="Style image" title="Style image" width="600" height="auto">

**Output**

<img src="output/output_1.jpg" alt="Output image" title="Output image" width="600" height="auto">

## Setup
To run this code you need the following:
* **Python3**
* A machine without a GPU is sufficient, though using a GPU will speed up the process.
* Install libraries -- install them using:

```pip install -r requirements.txt```

* Configure the paths to your images in main.py (for the target image, use the same as the content image).
* Define hyperparameters in main.py as needed.

## Run the model
To train the model and generate the output image, run:

```python main.py```


## Notes about hyperparameters and conclusions
* **Learning Rate:** Smaller values work best (ideally, less than 0.1).
* **Steps:** Experiment with the number of steps to achieve the desired results. Remember that minimizing the cost function is only one aspect—visual inspection of the output is key.
* **Alpha and Beta Values:** It is advisable to use a small value for alpha (1 or smaller) and a higher value for beta (at least 10,000). A higher beta can provide more flexibility in minimizing the cost function, potentially leading to better results.
* Feel free to modify the model, add or remove features in the cost function, and adjust the hyperparameters based on the desired outcome.
* This is a fun project to experiment with—try various settings and enjoy creating different styles!





