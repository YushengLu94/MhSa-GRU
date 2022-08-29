# -*- coding = utf-8 -*-
# @Time : 2021/11/2 16:21
# @Software: PyCharm
import math
import torch
import torch.nn as nn

#添加位置信息
class positionencoding(nn.Module):
    def __init__(self,embedding_size,max_len=50000):
        super(positionencoding, self).__init__()
        pe=torch.zeros(max_len,embedding_size)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)#二维[squence_len,]
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
        :param squence_len:假设输入的是product序列，len(product)
        :param embedding_size: 输入向量的维度,这里将product压缩成100维
        :param num_attention_heads: 将100维分成5个部分，即5个多头
        :param dropout_pro: dropout=0.2
        """
        super(selfattention, self).__init__()
        if embedding_size % num_attention_heads != 0: #要求整除，否则报错
            raise ValueError(
                "the embeddning size(%d) id not multiple of the number of attention"
                "heads (%d)" %(embedding_size,num_attention_heads)
            )
        #参数定义
        self.num_attention_heads = num_attention_heads #5个多头
        self.attention_head_size=int(embedding_size/num_attention_heads)#(embedding/5)每个注意力头的维度：20维
        self.all_head_size=int(self.num_attention_heads*self.attention_head_size) #all_head_size =embedding_size, 一般自注意力输入输出前后维度不变

        #query,key,value 的线性变化
        #self.weight=nn.Linear(embedding_size,embedding_size) #embedding->embedding 当然可以指定，这里只是用作权重
        self.pose_encoder = positionencoding(embedding_size)#加入位置信息
        self.tanh = nn.Tanh()
        self.fc=nn.Linear(embedding_size,embedding_size)

        #dropout
        self.dropout = nn.Dropout(dropout_pro)

    #定义多头拆解函数
    def transpose_for_scores(self,x):
        #input: x'shape=[sequence_len,embedding_size]
        new_x_shape=x.size()[0:1]+(self.num_attention_heads,self.attention_head_size) #[sequence_len]+[self.num_attention_heads,self.all_head_size] ->[sequence_len,self.num_attention_heads,self.all_head_size]
        x=x.view(*new_x_shape)
        return x.permute(1,0,2)#[sequence_len,num_attention_heads,all_head_size]->[num_attention_heads,sequence_len, attention_head_size]
        #这一步进行self-attention的多头操作，将其分为num_attention_heads个注意头来计算

    def forward(self,inputdata,attention_mask):
        #eg: inputdata = [squence_len,embedding_size]对应操作为embedding_size
        #eg: attention_mask = [self.num_attention_heads, seqlen, seqlen]
        attention_mask = (1.0-attention_mask)*-10000.0 #attention_mask = (1.0-attention_mask)*-10000.0

        #线性变换
        #mixed_query_layer=self.weight(inputdata) #[sequence_len,hidden_size]
        mixed_query_layer = self.tanh(inputdata)  # 进行tanh [sequence_len,embedding_size]
        #mixed_keys_layer = self.weight(inputdata) #[sequence_len,hidden_size]
        mixed_keys_layer = self.tanh(inputdata)  # 进行tanh [sequence_len,embedding_size]
        #mixed_value_layer = self.weight(inputdata)#[sequence_len,embedding_size] 必须与原来的维度一样
        #进行拆分
        query_layer = self.transpose_for_scores(mixed_query_layer) #[num_attention_heads,sequence_len,attention_head_size]
        keys_layer = self.transpose_for_scores(mixed_keys_layer)#[num_attention_heads,sequence_len,attention_head_size]
        value_layer = self.transpose_for_scores(inputdata)#[num_attention_heads,sequence_len,attention_head_size]

        #将query和keys进行点积相乘，只是做向量的相乘
        attention_scores=torch.matmul(query_layer,keys_layer.transpose(-1,-2))
        #[num_attention_heads,sequence_len,attention_head_size]*[num_attention_heads,attention_head_size,sequence_len]->[num_attention_heads,sequence_len,sequence_len]
        #进行缩放 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        attention_scores=attention_scores/math.sqrt(self.attention_head_size)# [num_attention_heads, seqlen, seqlen]
        attention_scores+=attention_mask#[num_attention_heads, seqlen, seqlen]
        # 加上mask，将padding所在的表示直接-10000
        attention_probs=nn.Softmax(dim=-1)(attention_scores)#[num_attention_heads, seqlen, seqlen] -1位置变为概率


        attention_probs = self.dropout(attention_probs)#防止过拟合 [num_attention_heads, seqlen, seqlen]

        context_layer=torch.matmul(attention_probs,value_layer)#[num_attention_heads, seqlen, seqlen]*[num_attention_heads,sequence_len,attention_head_size]->[num_attention_heads, sequence_len,attention_head_size]
        #接下去的步骤是将多头注意力拼接回去
        context_layer=context_layer.permute(1,0,2).contiguous()#[num_attention_heads, sequence_len,attention_head_size]->[sequence_len,num_attention_heads,attention_head_size]
        new_context_layer_shape=context_layer.size()[0:1]+(self.all_head_size,)
        context_layer=context_layer.view(*new_context_layer_shape)
        context_layer = self.pose_encoder(context_layer)  # 加入位置信息
        context_layer = self.fc(context_layer)#[sequence_len, hidden_size]->[sequence_len, embedding_size]
        return context_layer # [sequence_len, embedding_size] 得到输出


