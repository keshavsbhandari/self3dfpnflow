from torchvision.transforms import *
size = (128, 128)
rand_transform = Compose([
    RandomApply([ColorJitter(),
                  RandomGrayscale(),
                  RandomAffine(degrees=(-90, +90)),
                  RandomCrop(size),
                  RandomHorizontalFlip(0.1),
                  RandomPerspective(p=0.1),
                  ]),
    ToTensor()])