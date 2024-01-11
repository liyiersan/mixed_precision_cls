# mixed_precision_cls
Exploring mixed precision training for resnet on tiny imagenet.
### prapare the dataset
You can download Tiny-imagenet through http://cs231n.stanford.edu/tiny-imagenet-200.zip, and run preprocess.py to prepare the val dataset.
### train and test
Run main.py to see the results of mixed precision and FP32 training.
### notes
Currently, the classification performance is very bad on tiny-imagenet, the results is consistent with https://github.com/tjmoon0104/Tiny-ImageNet-Classifier.

I guess the main reason is the low image resolution and complex model. You can try to upsample the image as well as simple models like LeNet5 and ResNet_cifar.

For ResNet_cifar, you can refer to https://github.com/microsoft/snca.pytorch/blob/master/models/resnet_cifar.py.

