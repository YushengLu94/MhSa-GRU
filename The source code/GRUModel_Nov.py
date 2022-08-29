# -*- coding = utf-8 -*-
# @Time : 2021/11/4 15:44
# @Software: PyCharm

import torch
import torch.nn as nn
from self_attention import selfattention

device=torch.device('cuda')
class GRU_model(nn.Module):
    def __init__(self, vector_size, hidden_size, num_layers, output_size):
        super(GRU_model, self).__init__()

        #parameter settings
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        #embedding operations
        self.embedding_work_product = nn.Embedding(43419, 500)
        self.embedding_work_category = nn.Embedding(491, 20)
       
        #self_attention operations
        self.attention_product = selfattention(500, 50, 0.1)
        self.attention_category = selfattention(20, 2, 0.1).to(device)
        self.gru=nn.GRU(vector_size, hidden_size, num_layers, batch_first=True,dropout=0.2)
        self.fc1 = nn.Linear(hidden_size,2048)
        self.fc2 = nn.Linear(2048, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)


    def forward(self,product,category,behavior,price,x_mask_product,x_mask_category):
        #input:(squence_len,1)

        #embedding operations
        embed_product = self.embedding_work_product(product)#[len(product),embedding]
        embed_category = self.embedding_work_category(category)#[len(category),embedding]

        #self-attention operations
        embed_product = self.attention_product(embed_product,x_mask_product)#[len(product),embedding]
        embed_product = embed_product.unsqueeze(dim=0)#[batach_size,len(product),embedding]
        embed_category = self.attention_category(embed_category,x_mask_category)#[len(category),embedding]
        embed_category = embed_category.unsqueeze(dim=0)#[batach_size,len(category),embedding]

        # behavior, price unsqueeze operations,  to change a dimension
        behavior = behavior.unsqueeze(dim=2)#[batach_size,len(behavior),_]
        price = price.unsqueeze(dim=2)#[batach_size,len(price),_]

        # tensors concatenation operations
        embeds = torch.cat((embed_product, embed_category), 2)
        embeds = torch.cat((embeds, behavior), 2)
        embeds = torch.cat((embeds, price), 2)#size:[batch_size,squen_len,embedding(embedding:122)]

        # GRU network settings
        inithidden=torch.zeros(self.num_layers,1,self.hidden_size).to(device)
        #initcell=torch.zeros(self.num_layers,1,self.hidden_size).to(device)
        output,hidden=self.gru(embeds,inithidden)#[batch_size,sequence_len,embedding_size]
        output = self.relu(self.fc1(output))#[batch_size,sequence_len,1024]
        output = self.relu(self.fc2(output))#[batch_size,sequence_len,45484]
        return output#[batch_size,sequence_len,45484]
