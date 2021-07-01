import torch.optim
import torch.utils.data
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.utils.data

#########################NET##############################

#The backbone for the CNNS with shared weights
def backbone(**kwargs):
    """
    Construct a ResNet-101 model.
    Returns:
        Embeddingnet(model): The CNN with the specified model as its backbone is instantiated
    """
    #model = torch.hub.load('pytorch/vision:v1.7.1', 'resnet101', pretrained=True)
    model = models.resnet18(pretrained=True)
    #model = models.resnet34(pretrained=True)
    #model = models.vgg11_bn()
    #model = torch.hub.load('pytorch/vision:v0.8.2', 'alexnet', pretrained=True)
    #model = models.alexnet(pretrained=True)            #used in the paper
    #print('Layers',model.children)
    #model = models.resnet50(pretrained=True)
    #model = models.inception_v3(pretrained=True)
    #model = torchvision.models.resnet.ResNet(
        #torchvision.models.resnet.BasicBlock, [2, 1, 1, 1])

    return EmbeddingNet(model)

#The overall network consisting of three embedding nets with shared weights
class TripletNet(nn.Module):
    """Triplet Network."""

    def __init__(self, embeddingnet):
        """Triplet Network Builder."""
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet
        #print(self.embeddingnet.children())

    def forward(self, a, p, n):
        """Forward pass."""
        # anchor
        embedded_a = self.embeddingnet(a)

        # positive examples
        embedded_p = self.embeddingnet(p)

        # negative examples
        embedded_n = self.embeddingnet(n)

        return embedded_a, embedded_p, embedded_n

#The CNN used by Triplet Net with 'model' as its backbone and a final fully connected Layer
class EmbeddingNet(nn.Module):
    """EmbeddingNet using the specified model in backbone()."""

    def __init__(self, resnet):
        """Initialize EmbeddingNet model."""
        super(EmbeddingNet, self).__init__()
        # Everything excluding the last linear layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs =  resnet.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 1024)

    def forward(self, x):
        """Forward pass of EmbeddingNet."""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out