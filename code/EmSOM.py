import torch
import torch.nn as nn

class EmSOM(nn.Module):
    def __init__(self, som, som_hidd):
        super(EmSOM, self).__init__()

        self.hiddenlayer = nn.Linear(2576 + 1, 60) #60 hidden neurons
        self.outputlayer = nn.Linear(60 + 1, 40) #40 classes

        self.activation = torch.sigmoid

        self.som = som
        self.som_hidd = som_hidd
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def find_bmus(self, som, x):
        x = x.to(self.device)

        emotion, _ = som.predict(x)
        centr_topology = som.centroids.reshape(som.m, som.n, -1)

        index = [em.astype(int).tolist() for em in emotion]
        bmus = []

        for i in index:
            [x, y] = i
            bmus.append(centr_topology[x, y])

        bmus = list(map(torch.mean, bmus))
        return (torch.Tensor(bmus).to(self.device))

    def forward(self, x):
        x = x.to(self.device)
        bmus = self.find_bmus(self.som, x)
        x = torch.cat((x, bmus.unsqueeze(1)), dim=1)

        x = self.hiddenlayer(x)
        hidd_out = self.activation(x)

        bmus_hidd = self.find_bmus(self.som_hidd, hidd_out.detach())
        x = torch.cat((hidd_out, bmus_hidd.unsqueeze(1)), dim=1)

        x = self.outputlayer(x)
        x = self.activation(x)

        return x, hidd_out

    if __name__ == '__main__':
        pass