# Recognizing Captcha with Deep Learning

In this repository, multiple deep learning models have been explored to recognize random captcha images generated by the [captcha](https://github.com/lepture/captcha "captcha") library. This demo is based on the popular PyTorch package.

## Usage

Generate captcha images
```bash
$ python captcha_generation.py
```
This generates capcha images and stores them into a training and test folder according to the values in **config.py**. By default, 100,000 training images and 10,000 testing images are created.

Train a model
```bash
$ python main.py --arch Model1 --model_file model1.pt
```
This trains the preset **Model1** which is Convolutional Neural Network(CNN) consisting of two convolutional layers and two fully-connected layers. The parameters of the trained model is saved to **model1.pt**. The training and testing history are also saved, by default, to **Model1.json**
