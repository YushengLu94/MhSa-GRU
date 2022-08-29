# -*- coding = utf-8 -*-
# @Time : 2021/11/2 16:21
# @Software: PyCharm
import math
import torch
import torch.nn as nn

#Add position information
class positionencoding(nn.Module):
    def __init__(self,embedding_size,max_len=50000):
        super(positionencoding, self).__init__()
        pe=torch.zeros(max_len,embedding_size)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)#two dimensions [squence_len,]
        div_term=torch.exp(torch.arange(0,embedding_size,2).float()*(-math.log(10000.0)/embedding_size))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        #pe=pe.transpose(0,1)#[sqience_len,embedding_size]->[embedding_size,squence_len]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0),:]

class selfattention(nn.Module):
    def __init__(self,embedding_size,num_attention_heads,dropout_pro):
        """
        :param squence_len: Suppose the input is the product sequence，len(product)
        :param embedding_size: The dimension of the input vector, here the product is compressed into 100 dimensions
        :param num_attention_heads: Divide 100 dimensions into 5 heads
        :param dropout_pro: dropout=0.2
        """
        super(selfattention, self).__init__()
        if embedding_size % num_attention_heads != 0: # Divide is required, otherwise an error is reported
            raise ValueError(
                "the embeddning size(%d) id not multiple of the number of attention"
                "heads (%d)" %(embedding_size,num_attention_heads)
            )
        #parameter definition
        self.num_attention_heads = num_attention_heads # 5 heads
        self.attention_head_size=int(embedding_size/num_attention_heads)#(embedding/5)  # Dimensions of each attention head: 20 dimensions
        self.all_head_size=int(self.num_attention_heads*self.attention_head_size) #all_head_size =embedding_size, # Generally, the dimension of self-attention input and output does not change before and after

        #query,key,value , linear change
        #self.weight=nn.Linear(embedding_size,embedding_size) #embedding->embedding  # weight
        self.pose_encoder = positionencoding(embedding_size)# add position information
        self.tanh = nn.Tanh()
        self.fc=nn.Linear(embedding_size,embedding_size)

        #dropout
        self.dropout = nn.Dropout(dropout_pro)

    # Define the multi-head transpose function
    def transpose_for_scores(self,x):
        #input: x'shape=[sequence_len,embedding_size]
        new_x_shape=x.size()[0:1]+(self.num_attention_heads,self.attention_head_size) #[sequence_len]+[self.num_attention_heads,self.all_head_size] ->[sequence_len,self.num_attention_heads,self.all_head_size]
        x=x.view(*new_x_shape)
        return x.permute(1,0,2)#[sequence_len,num_attention_heads,all_head_size]->[num_attention_heads,sequence_len, attention_head_size]
        # In this step, the self-attention multi-head operation is performed, and it is divided into num_attention_heads attention heads to calculate

    def forward(self,inputdata,attention_mask):
        #eg: inputdata = [squence_len,embedding_size] # the according operation is embedding_size
        #eg: attention_mask = [self.num_attention_heads, seqlen, seqlen]
        attention_mask = (1.0-attention_mask)*-10000.0 #attention_mask = (1.0-attention_mask)*-10000.0

        #linear transformation
        #mixed_query_layer=self.weight(inputdata) #[sequence_len,hidden_size]
        mixed_query_layer = self.tanh(inputdata)  #  operate tanh [sequence_len,embedding_size]
        #mixed_keys_layer = self.weight(inputdata) #[sequence_len,hidden_size]
        mixed_keys_layer = self.tanh(inputdata)  # operate tanh [sequence_len,embedding_size]
        #mixed_value_layer = self.weight(inputdata)#[sequence_len,embedding_size] # Must be the same as the original dimension
        # transpose operation
        query_layer = self.transpose_for_scores(mixed_query_layer) #[num_attention_heads,sequence_len,attention_head_size]
        keys_layer = self.transpose_for_scores(mixed_keys_layer)#[num_attention_heads,sequence_len,attention_head_size]
        value_layer = self.transpose_for_scores(inputdata)#[num_attention_heads,sequence_len,attention_head_size]

        # Do the dot product multiplication of query and keys, just do vector multiplication
        attention_scores=torch.matmul(query_layer,keys_layer.transpose(-1,-2))
        #[num_attention_heads,sequence_len,attention_head_size]*[num_attention_heads,attention_head_size,sequence_len]->[num_attention_heads,sequence_len,sequence_len]
        # To scale and divide by the number of attention heads, you can see the formula of the original paper to prevent the score from being too large. If the score is too large, it will cause softmax to be either 0 or 1.
        attention_scores=attention_scores/math.sqrt(self.attention_head_size)# [num_attention_heads, seqlen, seqlen]
        attention_scores+=attention_mask#[num_attention_heads, seqlen, seqlen]
        # Add the mask, and the representation of the padding is directly -10000
        attention_probs=nn.Softmax(dim=-1)(attention_scores)#[num_attention_heads, seqlen, seqlen] -1 position becomes probability


        attention_probs = self.dropout(attention_probs)# prevent overfitting, [num_attention_heads, seqlen, seqlen]

        context_layer=torch.matmul(attention_probs,value_layer)#[num_attention_heads, seqlen, seqlen]*[num_attention_heads,sequence_len,attention_head_size]->[num_attention_heads, sequence_len,attention_head_size]
        # The next step is to stitch back the multi-head attention
        context_layer=context_layer.permute(1,0,2).contiguous()#[num_attention_heads, sequence_len,attention_head_size]->[sequence_len,num_attention_heads,attention_head_size]
        new_context_layer_shape=context_layer.size()[0:1]+(self.all_head_size,)
        context_layer=context_layer.view(*new_context_layer_shape)
        context_layer = self.pose_encoder(context_layer)  # add postion information
        context_layer = self.fc(context_layer)#[sequence_len, hidden_size]->[sequence_len, embedding_size]
        return context_layer # [sequence_len, embedding_size] # get the outcome


