import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot

from customSOM import customSOM
from EmGD import EmGD
from EmSOM import EmSOM

data_dir = '../data/orl'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Data Transformations / Train-Test split

image_transforms = transforms.Compose([transforms.Grayscale(num_output_channels = 1),
                                       transforms.Resize((46,56)), #scaled by factor 2 original size (112,92)
                                       transforms.ToTensor(),
                                       transforms.Normalize([0],[10],inplace=True), #normalize [0:1] -> [0:0.1]
                                       transforms.Lambda(lambda x: x.view(-1))])

faceDataset = datasets.ImageFolder(data_dir, transform=image_transforms)

train_idx, test_idx = train_test_split(np.arange(len(faceDataset.targets)),
                                      test_size=0.5,
                                      shuffle=True,
                                      stratify=faceDataset.targets)

train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = DataLoader(faceDataset,
                          batch_size=10,
                          sampler=train_sampler)
test_loader = DataLoader(faceDataset,
                        batch_size=10,
                        sampler=test_sampler)


#SOM training
m, n = 10, 10
niter = 100
batch_size = 1

sigma = 5
alpha = 0.5

#input batch data
X = [data[0].to(device) for data in train_loader]
X = torch.cat(X, dim=0)
dim = X.shape[1]

som = customSOM(m, n, dim, niter=niter, device=device, sigma=5, alpha=0.5, sched='linear')
som_hidd = customSOM(m, n, 60, niter=niter, device=device, sigma=5, alpha=0.5, sched='linear')

#Model initilization
model = EmSOM(som, som_hidd)

#hidden output data
_, hidd_out = model(X.cpu())
X_hidd = hidd_out.to(device)

learning_error = som.fit(X, batch_size=batch_size, print_each=10)
learning_hidd_error = som_hidd.fit(X_hidd.detach(), batch_size=batch_size, print_each=10)
#Model Training
#hyperparameteres
lr = 21e-2
momentum = 45e-2
epochs = 350

criterion = nn.MSELoss(reduction='sum')
optimizer = EmGD(model.parameters(), lr=lr, momentum=momentum, m=1, k=0)

print('START of TRAINING PERIOD')

for epoch in range(epochs):

    acc = running_loss = 0.0
    corrects = total = 0

    #emotional parameters initilization
    batch = torch.cat([torch.flatten(data[0], start_dim=1) for data in train_loader])

    out_0, _ = model(batch)
    labels = torch.cat([data[1] for data in train_loader])
    hot_labels_0 = one_hot(labels, num_classes=40)

    sum_sqr = (out_0 - hot_labels_0.float()) ** 2
    sr_sqr = torch.sqrt(sum_sqr.sum(dim=1))
    e_0 = torch.mean(sr_sqr)

    yall_pat = batch.mean()

    axt_0 = yall_pat + e_0
    mt = axt_0
    k = torch.tensor(0.0)

    for data in train_loader:
        samples, labels = data

        out, _ = model(samples)
        hot_labels = one_hot(labels, num_classes=40)

        optimizer.zero_grad()

        loss = 0.5 * criterion(out, hot_labels.float())

        loss.backward()
        optimizer.step()

        et = 0.5 * loss
        mt = yall_pat + (et / model.outputlayer.out_features)
        k = axt_0 - mt

        pred = torch.argmax(out, dim=1)

        corrects += (pred == labels).sum().item()
        total += len(pred)

        acc = corrects / total
        running_loss += loss.data

    print('[%d] loss: %.3f corrects: %d' %
          (epoch + 1, running_loss / len(train_loader), corrects), "Train Accuracy: %.2f %%" % (acc * 100))

print('END of TRAINING PERIOD')

acc = 0.0
total = corrects = 0
for index, data in enumerate(test_loader, 0):
    samples, labels = data
    out, _ = model(samples)
    hot_labels = one_hot(labels, num_classes=40)
    pred = torch.argmax(out, dim=1)

    corrects += (pred == labels).sum().item()
    total += len(pred)

acc = corrects / total
print('Test Accuracy %d %%' % (acc * 100))


