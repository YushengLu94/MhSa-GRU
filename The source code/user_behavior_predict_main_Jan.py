# -*- coding = utf-8 -*-
# @Time : 2021/11/9 10:28
# @Software: PyCharm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import _pickle as pickle
from visdom import Visdom
import torch.optim as optim
import torch.nn.functional as F
from shuffle_data import shuffle
from GRUModel_jan import GRU_model
from torch.autograd import Variable
from self_attention import selfattention

#Free up cuda memory to ensure enough space for training
torch.cuda.empty_cache()
#Define a function to solve Hit and MRR
def evalute(model,test_dataset):
    MRR = 0
    correct=0
    total=len(test_dataset)
    for idx in range(1,len(test_dataset)):
        product, category, behavior, price, product_label, users_id = test_dataset.iloc[idx]

        #tensor representation of various variables
        x_mask_product = torch.tril(torch.ones(len(product),len(product)),diagonal=0).to(device)
        x_mask_category = torch.tril(torch.ones(len(category),len(category)),diagonal=0).to(device)
        product = Variable(torch.LongTensor(product)).to(device)
        category = Variable(torch.LongTensor(category)).to(device)
        behavior = Variable(torch.FloatTensor(behavior).unsqueeze(0)).to(device)
        price = Variable(torch.FloatTensor(price).unsqueeze(0)).to(device)
        #Normalize behavior and price
        behavior = F.normalize(behavior, p=2, dim=1)
        price = F.normalize(price, p=2, dim=1)

        product_label = Variable(torch.from_numpy(np.array(product_label))).to(device)
        product_label = product_label.to(device)

        #No back-propagation involved, no need to derive parameters, reducing complexity
        with torch.no_grad():
            pred = model(product, category, behavior, price, x_mask_product, x_mask_category)#get the predicted value
            pred = pred[:,-1,:]
            pred = F.softmax(pred,dim=1)
            sorted_pred,sorted_index = torch.sort(pred,descending=True,dim=1)#Sort the output results according to the softmax value from major to minor
            recommend_product_20 = sorted_index[:,0:20].squeeze(dim=0)#Pick the top twenty as recommendations
            # Compute Hit@20
            if product_label in recommend_product_20:
                correct+=1
            #  Compute MRR@20
            recommend_product_list=sorted_index.squeeze(dim=0).tolist()
            id=recommend_product_list.index(product_label.item())+1
            if id > 20:
                rank_value = 0
                MRR += rank_value
            else:
                rank_value = 1/id
                MRR += rank_value
    return (correct/total)*100, (MRR/total)

#Define an L2 regularization
def L2_regularization(model,alpha):
    L2_loss=torch.tensor(0.0,requires_grad=True)
    for name,parma in model.named_parameters():
        if 'bias' not in name:
            L2_loss=L2_loss+(0.5*alpha*torch.sum(torch.pow(parma,2)))
    return L2_loss

device=torch.device('cuda')
iter=0
#best_acc=43.986089459167765
best_acc=0
epoch_iter=0
best_epoch=0
learn_rating=0.01
torch.manual_seed(1234)
#draw a curve diagram

viz=Visdom()
viz.line([0],[-1],win='val_acc',opts=dict(title='val_acc'))
viz.line([0],[-1],win='train_loss',opts=dict(title='train_loss'))
viz.line([0],[-1],win='total_loss',opts=dict(title='total_loss'))

#Import train data
with open('userdata_train_all.pkl', 'rb') as train:
    data_train = pickle.load(train)

#Import test data
with open('userdata_test_all.pkl', 'rb') as test:
    data_test = pickle.load(test)

model = GRU_model(vector_size=522,hidden_size=200,num_layers=2,output_size=45484).to(device)
model.load_state_dict(torch.load("best-4387-125wei-jan.mdl"))#Load the last optimal parameter result and use it directly
optimizer=optim.SGD(model.parameters(),lr=learn_rating)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)#Set the optimizer update schedule
criterion=nn.CrossEntropyLoss()
print(model.parameters)

for epoch in range(100):
    scheduler.step()#Update the learning rate before each epoch
    total_loss=0
    data_train=shuffle(data_train)#Randomly shuffle the data in data_train to ensure that the order of training data obtained for each epoch is different
    for idx in range(len(data_train)):#len(data_train)
        product, category, behavior, price, product_label, users_id = data_train.iloc[idx]

        # tensor representation of various variables
        x_mask_product = torch.tril(torch.ones(len(product),len(product)),diagonal=0).to(device)
        x_mask_category = torch.tril(torch.ones(len(category),len(category)),diagonal=0).to(device)
        product = Variable(torch.LongTensor(product)).to(device)
        category = Variable(torch.LongTensor(category)).to(device)
        behavior = Variable(torch.FloatTensor(behavior).unsqueeze(0)).to(device)
        price = Variable(torch.FloatTensor(price).unsqueeze(0)).to(device)
        # Normalize behavior and price
        behavior = F.normalize(behavior, p=2,dim=1)
        price = F.normalize(price, p=2,dim=1)
        product_label = Variable(torch.from_numpy(np.array(product_label,dtype=np.int64))).to(device)#sequence_len,1]
        product_label = product_label.to(device)

        pred = model(product, category, behavior, price,x_mask_product,x_mask_category)#get the predicted value, [batch_size,sequence_len,45484]
        pred = pred.squeeze(dim=0)#[sequence_len,45484]
        #print(product_label.shape)
        #print(pred.shape)
        loss = criterion(pred, product_label)# Calculate the loss function value
        #L2_loss = L2_regularization(model,0.5)# add regularization
        #loss += L2_loss

        # backpropagation derivation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        if iter % 1000 == 0:
            viz.line([loss.item()],[iter],win='train_loss',update='append')
            print('The {}th round, the {}th idx, the loss function is: {:.6f}'.format(epoch,idx,loss.item()))
        iter += 1
    total_loss=total_loss/len(data_train)
    file='the'+str(epoch)+'train.mdl'
    torch.save(model.state_dict(),file)
    epoch_iter+=1
    viz.line([total_loss.item()], [epoch_iter], win='total_loss', update='append')
    print('total_loss:',total_loss.item())
    #Do the validation set test and save the best set of parameter values to prevent overfitting
    if epoch <=30:
        num=2
    elif 30<epoch<=100:
        num = 2
    if epoch % num == 0:
        print('the train loss:',total_loss.item())
        Hit,MRR = evalute(model,data_test)#Get the accuracy on the validation set
        viz.line([Hit], [iter], win='val_acc', update='append')
        print('the Hit@20', Hit)
        print('the MRR@20', MRR)
        if Hit > best_acc:
            best_epoch = epoch
            best_acc = Hit
            #Save the current best result
            torch.save(model.state_dict(),'best.mdl')
print('best_acc',best_acc,'best_epoch:',best_epoch)
print('train is over!')
#Free memory again after running
torch.cuda.empty_cache()
'''
#test set test
Hit,MRR = evalute(model,data_test)#Get the accuracy on the validation set
print('the Hit@20',Hit)
print('the MRR@20',MRR)
'''

