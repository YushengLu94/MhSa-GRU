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
from LSTMModel_tafeng import GRU_model
from torch.autograd import Variable
from self_attention import selfattention

#释放cuda内存，确保有足够空间进行训练
torch.cuda.empty_cache()
#定义一个函数求解Hit和MRR
def evalute(model,test_dataset):
    MRR = 0
    correct=0
    total=len(test_dataset)
    for idx in range(1,len(test_dataset)):
        product, category, behavior, price, product_label, users_id = test_dataset.iloc[idx]

        #各种变量的tensor表示
        x_mask_product = torch.tril(torch.ones(len(product),len(product)),diagonal=0).to(device)
        x_mask_category = torch.tril(torch.ones(len(category),len(category)),diagonal=0).to(device)
        product = Variable(torch.LongTensor(product)).to(device)
        category = Variable(torch.LongTensor(category)).to(device)
        behavior = Variable(torch.FloatTensor(behavior).unsqueeze(0)).to(device)
        price = Variable(torch.FloatTensor(price).unsqueeze(0)).to(device)
        #对behavior和price归一化
        behavior = F.normalize(behavior, p=2, dim=1)
        price = F.normalize(price, p=2, dim=1)

        product_label = Variable(torch.from_numpy(np.array(product_label))).to(device)
        product_label = product_label.to(device)

        #不涉及反向传播，不需要对参数进行求导，降低复杂度
        with torch.no_grad():
            pred = model(product, category, behavior, price, x_mask_product, x_mask_category)#得到预测值
            pred = pred[:,-1,:]
            pred = F.softmax(pred,dim=1)
            sorted_pred,sorted_index = torch.sort(pred,descending=True,dim=1)#对输出的结果按照softmax值进行重大到小的排序
            recommend_product_20 = sorted_index[:,0:20].squeeze(dim=0)#选取前二十个作为推荐
            # 计算Hit@20
            if product_label in recommend_product_20:
                correct+=1
            #计算MRR@20
            recommend_product_list=sorted_index.squeeze(dim=0).tolist()
            id=recommend_product_list.index(product_label.item())+1
            if id > 20:
                rank_value = 0
                MRR += rank_value
            else:
                rank_value = 1/id
                MRR += rank_value
    return (correct/total)*100, (MRR/total)

#定义一个L2正则化
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
#绘制曲线图
viz=Visdom()
viz.line([0],[-1],win='val_acc',opts=dict(title='val_acc'))
viz.line([0],[-1],win='train_loss',opts=dict(title='train_loss'))
viz.line([0],[-1],win='total_loss',opts=dict(title='total_loss'))

#导入train数据
with open('userdata_train_all_tafeng.pkl', 'rb') as train:
    data_train = pickle.load(train)

#导入test数据
with open('userdata_test_all_tafeng.pkl', 'rb') as test:
    data_test = pickle.load(test)

model = GRU_model(vector_size=902,hidden_size=300,num_layers=2,output_size=23812).to(device)
#model.load_state_dict(torch.load("best_100wei_tafeng.mdl"))#加载上一次的最优参数结果，直接使用
optimizer=optim.SGD(model.parameters(),lr=learn_rating)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)#设定优化器更新时刻表
criterion=nn.CrossEntropyLoss()
print(model.parameters)

for epoch in range(200):
    scheduler.step()#每轮epoch之前更新学习率
    total_loss=0
    data_train=shuffle(data_train)#对data_train中的数据进行随机洗牌，以保证每个epoch得到的训练数据顺序不一样
    for idx in range(len(data_train)):#len(data_train)
        product, category, behavior, price, product_label, users_id = data_train.iloc[idx]

        # 各种变量的tensor表示
        x_mask_product = torch.tril(torch.ones(len(product),len(product)),diagonal=0).to(device)
        x_mask_category = torch.tril(torch.ones(len(category),len(category)),diagonal=0).to(device)
        product = Variable(torch.LongTensor(product)).to(device)
        category = Variable(torch.LongTensor(category)).to(device)
        behavior = Variable(torch.FloatTensor(behavior).unsqueeze(0)).to(device)
        price = Variable(torch.FloatTensor(price).unsqueeze(0)).to(device)
        # 对behavior和price归一化
        behavior = F.normalize(behavior, p=2,dim=1)
        price = F.normalize(price, p=2,dim=1)
        product_label = Variable(torch.from_numpy(np.array(product_label,dtype=np.int64))).to(device)#sequence_len,1]
        product_label = product_label.to(device)

        pred = model(product, category, behavior, price,x_mask_product,x_mask_category)#得到预测值[batch_size,sequence_len,45484]
        pred = pred.squeeze(dim=0)#[sequence_len,45484]
        #print(product_label.shape)
        #print(pred.shape)
        loss = criterion(pred, product_label)#计算损失函数值
        #L2_loss = L2_regularization(model,0.5)#加入正则化
        #loss += L2_loss

        #反向传播求导
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        if iter % 1000 == 0:
            viz.line([loss.item()],[iter],win='train_loss',update='append')
            print('第{}轮，第{}个idx,损失函数为：{:.6f}'.format(epoch,idx,loss.item()))
        iter += 1
    total_loss=total_loss/len(data_train)
    file='the'+str(epoch)+'train.mdl'
    torch.save(model.state_dict(),file)
    epoch_iter+=1
    viz.line([total_loss.item()], [epoch_iter], win='total_loss', update='append')
    print('total_loss:',total_loss.item())
    #做验证集测试，将效果最好的一组参数值保存，防止过拟合
    if epoch <=30:
        num=1
    elif 30<epoch<=80:
        num = 1
    if epoch % num == 0:
        print('the train loss:',total_loss.item())
        Hit,MRR = evalute(model,data_test)#得到验证集上的准确率
        viz.line([Hit], [iter], win='val_acc', update='append')
        print('the Hit@20', Hit)
        print('the MRR@20', MRR)
        if Hit > best_acc:
            best_epoch = epoch
            best_acc = Hit
            #保存当前最优结果
            torch.save(model.state_dict(),'best.mdl')
print('best_acc',best_acc,'best_epoch:',best_epoch)
print('train is over!')
#运行完后再一次释放内存
torch.cuda.empty_cache()
'''
#测试集检验
Hit,MRR = evalute(model,data_test)#得到验证集上的准确率
print('the Hit@20',Hit)
print('the MRR@20',MRR)
'''

