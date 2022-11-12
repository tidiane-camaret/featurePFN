import lightly
nb_features = 512
resnet = lightly.models.ResNetGenerator('resnet-18', num_classes=nb_features)
print(resnet)