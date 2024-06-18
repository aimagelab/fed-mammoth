from datasets import register_dataset
import torchvision.transforms as transforms
from datasets.seq_cifar100 import SequentialCifar100


@register_dataset("seq-cifar100_224")
class SequentialCifar100_224(SequentialCifar100):
    MEAN_NORM = (0.5, 0.5, 0.5)
    STD_NORM = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(MEAN_NORM, STD_NORM)
    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    INPUT_SHAPE = (224, 224, 3)
