import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import csv
import os
import datetime

torch.manual_seed(42)

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
    batch_size=32,
    shuffle=True,
    num_workers=32
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=32
)


def get_model(index):
    switcher = {
        1: models.googlenet(pretrained=True),
        2: models.squeezenet1_1(pretrained=True),
        3: models.mnasnet1_0(pretrained=True),
    }
    return switcher.get(index, alex_net())


def alex_net():
    m = models.alexnet(pretrained=True)
    m.classifier[6] = torch.nn.Linear(m.classifier[6].in_features, 2)
    return m


NUM_EPOCHS = 20

results = [[0 for x in range(4)] for y in range(NUM_EPOCHS)]
duration = [[0 for x in range(4)]for y in range(2)]

for idx in {0, 1, 2, 3}:
    torch.manual_seed(42)
    model = get_model(idx)
    best_accuracy = 0.0

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NUM_EPOCHS):

        train_timer = datetime.datetime.now()

        for images, labels in iter(train_loader):

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        print(datetime.datetime.now() - train_timer)
        duration[0][idx] = duration[0][idx] + (datetime.datetime.now()-train_timer).total_seconds()
        test_timer = datetime.datetime.now()
        test_error_count = 0.0
        for images, labels in iter(test_loader):
            outputs = model(images)
            test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

        duration[1][idx] = duration[1][idx] + (datetime.datetime.now() - test_timer).total_seconds()

        test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))

        results[epoch][idx-1] = test_accuracy

        print('%d: %f' % (epoch, test_accuracy))
        if test_accuracy > best_accuracy:
            # torch.save(model.state_dict(), BEST_MODEL_PATH)
            # print("Saved best model")
            best_accuracy = test_accuracy
    del model

for i in range(len(duration)):
    for j in range(len(duration[i])):
        print(duration[i][j], end=' ')
    print()

with open("duration.csv", "w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(duration)

with open("results.csv", "w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(results)