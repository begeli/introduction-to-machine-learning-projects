"""
Implementation adapted from 'https://github.com/Zhenye-Na/image-similarity-using-deep-ranking'

"""

import torch.backends.cudnn as cudnn
from net import *
from loader import DatasetImageLoader
from train import train
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# For reproducbility
cudnn.benchmark = False
cudnn.deterministic = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic=True

#Validation set split and write train/validation txt files # not necesssary for submission but used for model evaluation
with open('train_triplets.txt') as f:
    train_triplets = f.readlines()
train_triplets = [x.strip() for x in train_triplets]

train_triplets_splitted = np.array([['00000'] * 3] * len(train_triplets))  # TURN EACH ROW OF READ LISTS TO SEPARATE STRING NUMPY ARRAY ELEMENTS
#train_triplets_splitted = np.zeros((len(train_triplets),3))

for i in range(len(train_triplets)):
        train_triplets_splitted[i, :] = np.array(train_triplets[i].split())


train_triplets,val_triplets = train_test_split(train_triplets_splitted, test_size=0.001, random_state=42, shuffle=True)

train_triplets_list = []
val_triplets_list = []

for i in range(train_triplets.shape[0]):
    train_triplets_list.append(train_triplets[i,0]+' '+train_triplets[i,1]+' '+train_triplets[i,2])

for i in range(val_triplets.shape[0]):
    val_triplets_list.append(val_triplets[i,0]+' '+val_triplets[i,1]+' '+val_triplets[i,2])

with open('train_triplets_splitted.txt', 'w') as f:
    for item in train_triplets_list:
        f.write("%s\n" % item)

with open('validation_triplets_splitted.txt', 'w') as f:
    for item in val_triplets_list:
        f.write("%s\n" % item)


#######################################MAIN###################################

#Initialize the model
net = TripletNet(backbone())

#Move the net to GPU for training
print("==> Initialize CUDA support for TripletNet model ...")
net = torch.nn.DataParallel(net).cuda()
cudnn.benchmark = True

# Loss function, optimizer and scheduler
criterion = nn.TripletMarginLoss(margin=5.0, p=2)

optimizer = torch.optim.SGD(net.parameters(),
                            lr=0.0005,
                            momentum=0.9,
                            weight_decay=2e-3,#The value used in the paper is 1e-3
                            nesterov=True)

#These batch sizes were used as we ran the code in Colab and not our local machine which worked flawlessly.
#Subject to change if run on a local machine
val_batch_size = 120
train_batch_size = 120
test_batch_size = 120
# load triplet dataset - MUST ENTER THE PATH OF THE FOOD FOLDER!!!!!!!!!!!!!!!!!!!!
path = "C:\\Users\\user\\PycharmProjects\\IML TASK 4 Working\\food"

#Load train,test,validation triplets
trainloader,testloader, valloader = DatasetImageLoader(path.rstrip("\n"),train_batch_size,test_batch_size,val_batch_size)#TRAIN BATCH SIZE, TEST BATCH SIZE, VAL BATCH SİZE

#train model
trained_net = train(net, criterion, optimizer, 0, 1, True,trainloader,valloader,train_batch_size,val_batch_size)#START EPOCH AND FINAL EPOCH

######################################PREDICTION ON TEST SET#################################

#Switch the net to evaluation mode
trained_net.eval()

predicted_labels = np.zeros(59544)
pred_test=[]

#Predict labels 1 or 0 for each test triplet
for batch_idx, (data1, data2, data3) in enumerate(testloader):

    data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

    # wrap in torch.autograd.Variable
    data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

    with torch.no_grad():
        # compute output and loss
        embedded_a, embedded_p, embedded_n = trained_net(data1, data2, data3)

    #print(np.squeeze(embedded_a.cpu().detach().numpy()).shape)
    dist_ap = np.linalg.norm(np.squeeze((embedded_a-embedded_p).cpu().detach().numpy()),ord=2, axis=-1)
    dist_an = np.linalg.norm(np.squeeze((embedded_a-embedded_n).cpu().detach().numpy()), ord=2, axis=-1)

    pred_test.append(1*(dist_ap <= dist_an))

    if batch_idx%1000 == 0:
        print(batch_idx)


predicted_labels = np.hstack(pred_test)
print(predicted_labels)

#Write submisison file
df = pd.DataFrame(predicted_labels)
df.to_csv('submission.txt', index=False, header=None) #write CSV
