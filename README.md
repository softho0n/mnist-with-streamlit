# mnist-with-streamlit
> This is prototype for predicting MNIST dataset using awesome streamlit library and PyTorch

## What is MNIST Dataset?
The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments.
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png"></p>

## Training Part
### Tools
* Python
* PyTorch
* Ubuntu 18.04 LTS

### Model Structure
<p align="center"><img width="502" alt="image" src="https://user-images.githubusercontent.com/42256738/173335304-cb5d62d1-c5ba-4fbf-8f7f-e336f1cf1a94.png"></p>

### Train
* `Batch Size    : 100`
* `Epochs        : 30`
* `Optimizer     : Adam Optimizer (Default Setting)`
* `Learning Rate : Default lr`
* Refer this code

## How to run?
### Requirements
* `PyTorch`
* `Streamlit`
* `OpenCV`
* `Python`
### Instructions
```console
foo@bar:~$ pip install streamlit torch torchsummary opencv-python
foo@bar:~$ streamlit run app.py
```
### Example
<p align="center"><img width="40%" src="https://user-images.githubusercontent.com/42256738/173337128-a19095a5-cc39-4cc5-b5f1-64dc0b373c4e.gif"></p>
