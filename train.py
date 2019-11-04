import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms



dataset = datasets.ImageFolder(
    'dataset_blocked_free',
    transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 75, 75])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=16
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=16
)


#Alexnet, Googlenet, SqueezeNet, ResNext, ResNet, MnasNet

modelSet = {1: models.alexnet(pretrained=True),
            2: models.googlenet(pretrained=True),
            3: models.squeezenet1_1(pretrained=True),
            4: models.resnet152(pretrained=True),
            5: models.resnext101_32x8d(pretrained=True),
            6: models.mnasnet1_0(pretrained=True)}

print(len(modelSet))

modelSet.get(1).classifier[6] = torch.nn.Linear(modelSet.get(1).classifier[6].in_features, 2)

for idx in {2}:
    #model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

    print(f"Loading model: {idx}")
    device = torch.device('cuda')
    model = modelSet.get(idx).to(device)
    print("Finished loading model")

    NUM_EPOCHS = 2
    #BEST_MODEL_PATH = 'best_model.pth'
    best_accuracy = 0.0

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NUM_EPOCHS):

        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        test_error_count = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

        test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
        print('%d: %f' % (epoch, test_accuracy))
        if test_accuracy > best_accuracy:
            # torch.save(model.state_dict(), BEST_MODEL_PATH)
            # print("Saved best model")
            best_accuracy = test_accuracy
