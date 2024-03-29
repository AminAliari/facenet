<div align="center">

# Real Face Detector - COMP6321

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/AminAliari/facenet/blob/main/tests/notebook_real_face_detector.ipynb)

![](https://github.com/AminAliari/facenet/blob/main/figs/1.png)

</div>

## Project Structure
Here is our project folder structure and files with their descriptions.

```
root
│
└───.github: GitHub workflow.
│
└───data: contains sample data for tests.
|
└───facenet: facenet package.
│
└───recipes: contains training codes for different models.
|
└───tests: contains unit tests and test scripts.
|
└───trained: contains different trained models such as FaceCNN, MLP, etc.
```


## Installation
In the root folder, run the following command:
```
pip install .
```
or
```
 pip install . --use-feature=in-tree-build
```

## Dataset
The dataset is not publicly available and can be requested from [here](https://github.com/EricGzq/Hybrid-Fake-Face-Dataset).

## Training

Our training scripts reside under the `recipes` folder. For training, we recommend using the same `conda` environment that has been defined for **COMP6321**. This means using `python 3.8` and having the basic required packages like `torch`, `numpy`, `scikit-learn`, `pandas`, `seaborn` for the heatmap, and `Pillow` for image loading. Also, for faster training, you should use a `cuda` GPU with at least `8 GB` VRAM and related torch packages installed. Should no GPU is not available, the CPU will be picked for training automatically.

### Example of How to Run `train.py` :
In this example, we are launching the training process for *FaceCNN*.  Run the following command in the `root` of the project:
```
python.exe "recipes\FaceCNN\train.py"
```

The trained model along with training and validation loss histories would be saved under the `trained` folder.
At the end of each epoch, if the new weights achieve a lower validation loss, we would save that model. Each epoch should take about *15* minutes with `RTX3070`. 

## Testing and Evaluation
The project notebook can be accessed from [here](https://github.com/AminAliari/facenet/blob/main/tests/notebook_real_face_detector.ipynb). It contains all of the testing and evaluation codes along with various figures such as confusion matrices.  Running this notebook should give the expected result and is pretty much self-explanatory.

## References
We follow the [SpeachBrain](https://github.com/speechbrain/speechbrain) guidelines for project structure, unit tests, documentation, formatting, and GitHub CI. *SpeachBrain* deployment code has also been **directly** used in this project.

**Note**: All other parts of the project are completely original and written by *Mohammadamin Aliari*. Also, all the academic references and other inspirations have been acknowledged in the project report.
