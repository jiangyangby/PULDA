from .modelForCIFAR10 import CNN

def create_model(dataset):
    if dataset.startswith('cifar'):
        return CNN()
    else:
        raise NotImplementedError("The model: {} is not defined!".format(dataset))