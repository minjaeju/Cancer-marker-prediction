#import torch_geometric.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv,GATConv,GINConv, APPNP
from torch_geometric.nn.pool.asap import ASAPooling
from sagpool import SAGPool
from torch_geometric.nn.pool import *
import pickle
import numpy as np


class simpleGCN(torch.nn.Module):
    def __init__(self,num_genes, num_class, num_layers=1):
        
        super(simpleGCN,self).__init__()
        print("simpleGCN:{} layers".format(num_layers))
        
        self.num_genes = num_genes
        self.num_class = num_class
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
        self.exp_embedding = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = self.num_class)
        self.exp_bias = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = 1)
        
        self.convs = torch.nn.ModuleList()
        self.convs.extend([
            GCNConv(self.num_class,self.num_class)
            for i in range(num_layers)
        ])
        
        self.lin1 = torch.nn.Linear(self.num_genes, 1024)
        self.lin2 = torch.nn.Linear(1024, self.num_class)

        
    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index, data.batch
        
        #.item() gets scalr
        batch_size = data.batch[-1].item() + 1
        
        #get node embedding
        #node_index = torch.LongTensor(list(range(self.num_genes))*batch_size).to(self.device)
        #exp_emb = self.exp_embedding(node_index)
        #exp_emb = exp_emb * x
        
        #exp_bias = self.exp_bias(node_index)
        #exp_emb = exp_emb+exp_bias
        #x = exp_emb
        
#         x = self.conv1(exp_emb, edge_index)
#         x = F.tanh(x)
        #for i,conv in enumerate(self.convs):
            #x = conv(x,edge_index)
#             x = F.tanh(x)
            #x = F.relu(x)
        
#         x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = torch.reshape(x, (1,6016))
        #x = torch.cat([gap(x, batch)], dim=1)
        #x = self.lin1(x)
        #print(x)
        x = F.relu(self.lin1(x))
        #print(x)
        x = self.lin2(x)

        #print(x.shape)
        return x 

    
    
class simpleGCN2(torch.nn.Module):
    def __init__(self,num_genes, num_class):
        
        super(simpleGCN2,self).__init__()
        print("simpleGCN2")
        
        self.num_genes = num_genes
        self.num_class = num_class
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
        self.exp_embedding = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = self.num_class)
        self.exp_bias = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = 1)
        
        self.conv1 = GCNConv(self.num_class,self.num_class)
        self.conv2 = GCNConv(self.num_class,self.num_class)
        
        self.dropout = nn.Dropout(0.3)
        
        self.lin1 = torch.nn.Linear(self.num_class, self.num_class)
        self.lin2 = torch.nn.Linear(self.num_class, self.num_class)
        
        
    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index, data.batch
        
        #.item() gets scalr
        batch_size = data.batch[-1].item() + 1
        
        #get node embedding
        node_index = torch.LongTensor(list(range(self.num_genes))*batch_size).to(self.device)
        exp_emb = self.exp_embedding(node_index)
        exp_emb = exp_emb * x
        
        exp_bias = self.exp_bias(node_index)
        exp_emb = exp_emb+exp_bias
        
        x = exp_emb
        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = self.conv2(x,edge_index)
        x = F.tanh(x)
        
        x = torch.cat([gap(x, batch)], dim=1)
        x = self.dropout(x)
        x = self.lin2(x)
        return x 
        
class simpleGCN_ASAP(torch.nn.Module):
    def __init__(self,num_genes, num_class):
        
        super(simpleGCN_ASAP,self).__init__()
        print("simpleGCN_ASAP")
        
        self.num_genes = num_genes
        self.num_class = num_class
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
        #Expression Embedding
        self.exp_embedding = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = self.num_class)
        self.exp_bias = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = 1)
        
        #GNN layers
        self.conv1 = GCNConv(self.num_class,self.num_class)
        
        #Pooling Layers
        self.pool1 = ASAPooling(self.num_class, ratio=0.3)
        
        #Final FC layer 
        self.lin1 = torch.nn.Linear(self.num_class, 1024)
        self.lin2 = torch.nn.Linear(1024, self.num_class)
        
    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index, data.batch
        
        #.item() gets scalr
        batch_size = data.batch[-1].item() + 1
        
        #get node embedding
        node_index = torch.LongTensor(list(range(self.num_genes))*batch_size).to(self.device)
        exp_emb = self.exp_embedding(node_index)
        exp_emb = exp_emb * x
        
        exp_bias = self.exp_bias(node_index)
        exp_emb = exp_emb+exp_bias
        
        x = self.conv1(exp_emb, edge_index)
        x = F.tanh(x)
        
#         print("pre pooled size:", x.size(), batch)
        new_x, edge_index ,_, batch ,_ = self.pool1(exp_emb, edge_index, batch=batch)
        del(edge_index)
        del(x)
        x = new_x
#         print("pooled size:", x.size(), batch)
        x = F.tanh(x)
        
#         x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = torch.cat([gap(x, batch)], dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x 
        
# class simpleDiffPool(torch.nn.Module):
#     def __init__(self,num_genes, num_class):
        
#         super(simpleGCN,self).__init__()
#         print("simpleGCN")
        
#         self.num_genes = num_genes
#         self.num_class = num_class
        
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(self.device)
        
#         self.exp_embedding = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = self.num_class)
#         self.exp_bias = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = 1)
        
#         self.conv1 = GCNConv(self.num_class,self.num_class)
#         self.lin1 = torch.nn.Linear(self.num_class, self.num_class)
        
#     def forward(self, data):
#         x, edge_index,batch = data.x, data.edge_index, data.batch
        
#         #.item() gets scalr
#         batch_size = data.batch[-1].item() + 1
        
#         #get node embedding
#         node_index = torch.LongTensor(list(range(self.num_genes))*batch_size).to(self.device)
#         exp_emb = self.exp_embedding(node_index)
#         exp_emb = exp_emb * x
        
#         exp_bias = self.exp_bias(node_index)
#         exp_emb = exp_emb+exp_bias
        
#         x = self.conv1(exp_emb, edge_index)
#         x = F.tanh(x)
        
# #         x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
#         x = torch.cat([gap(x, batch)], dim=1)
#         x = self.lin1(x)
#         return x 


class simpleGCN_SAGPOOL(torch.nn.Module):
    def __init__(self,num_genes, num_class, k=1, alpha=0.1, num_layers=1):
        
        super(simpleGCN_SAGPOOL,self).__init__()
        print("simpleGCN_SAGPOOL:{} layers".format(num_layers))
        
        self.num_genes = num_genes
        self.num_class = num_class
        self.k = k
        self.alpha = alpha
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
        self.exp_embedding = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = self.num_class)
        self.exp_bias = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = 1)
        
        self.convs = torch.nn.ModuleList()
        self.convs.extend([
            GCNConv(self.num_class,self.num_class)
            for i in range(num_layers)
        ])
        
        self.pool1 = SAGPool(self.num_class, ratio=0.5)
        self.lin1 = torch.nn.Linear(self.num_class, self.num_class)
        
    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index, data.batch
        
        #with open('./data/preprocessed/netics_adj_norm.pkl', 'rb') as f:
            #netics_adj_norm = pickle.load(f)
        
        
        #.item() gets scalr
        batch_size = data.batch[-1].item() + 1
        
        #h = x
        #for k in range(self.k):
            #x = x*(1-self.alpha)
            #x = torch.sparse.mm(netics_adj_norm.cuda(), x)
            
        #x += h*self.alpha
        
        #get node embedding
        node_index = torch.LongTensor(list(range(self.num_genes))*batch_size).to(self.device)
        exp_emb = self.exp_embedding(node_index)
        exp_emb = exp_emb * x
        
        exp_bias = self.exp_bias(node_index)
        exp_emb = exp_emb+exp_bias
        
        x = exp_emb
        
#         x = self.conv1(exp_emb, edge_index)
#         x = F.tanh(x)
        #for i,conv in enumerate(self.convs):
            #x = conv(x,edge_index)
            #x = torch.tanh(x)
        #print(x)
        x,edge_index,_,batch,_= self.pool1(x, edge_index,None, batch) 

#         x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #print(x)

        x = torch.cat([gap(x, batch)], dim=1)
        x = self.lin1(x)
        return x 
    
    
class simpleGCN_APPNP(torch.nn.Module):
    def __init__(self,num_genes, num_class, k=10, alpha=0.1,dropout=0.5):
        
        super(simpleGCN_APPNP,self).__init__()
        print("simpleGCN_APPNP: K:{}, alpha:{}, dropout:{}".format(k,alpha,dropout))
        
        self.num_genes = num_genes
        self.num_class = num_class
        self.k = k
        self.alpha = alpha
        self.dropout = dropout
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
        self.exp_embedding = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = self.num_class)
        self.exp_bias = nn.Embedding(num_embeddings = self.num_genes , embedding_dim  = 1)
        
        self.lin1 = torch.nn.Linear(self.num_class, self.num_class)
        self.lin2 = torch.nn.Linear(self.num_class, self.num_class)
        self.lin3 = torch.nn.Linear(self.num_class, self.num_class)
        self.prop1 = APPNP(self.k, self.alpha)
    
#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#         self.lin3.reset_parameters()
        
    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index, data.batch
        
        #.item() gets scalr
        batch_size = data.batch[-1].item() + 1
        
        #get node embedding
        node_index = torch.LongTensor(list(range(self.num_genes))*batch_size).to(self.device)
        exp_emb = self.exp_embedding(node_index)
        exp_emb = exp_emb * x
        
        exp_bias = self.exp_bias(node_index)
        exp_emb = exp_emb+exp_bias
        
        x = exp_emb
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        #print(x)

#         x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = torch.cat([gap(x, batch)], dim=1)
        #x = self.lin3(x)
        return x 

class pagerank_pool(torch.nn.Module):
    def __init__(self, num_genes, num_class, k=10, alpha=0.1,dropout=0.1):
        
        super(pagerank_pool,self).__init__()
        print("pagerank_pool: K:{}, alpha:{}, dropout:{}".format(k,alpha,dropout))
        
        torch.manual_seed(0)

        self.num_genes = num_genes
        self.num_genes2 = num_genes*2
        self.num_class = num_class
        self.k = k
        self.alpha = alpha
        self.dropout = dropout
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
        self.emb_mut = nn.Parameter(torch.rand(1))
        self.bias_mut = nn.Parameter(torch.rand(1))
        self.emb_exp = nn.Parameter(torch.rand(1))
        self.bias_exp = nn.Parameter(torch.rand(1))

        self.pool = TopKPooling(2, ratio=0.5)
        self.lin1 = torch.nn.Linear(self.num_genes2, 1024)
        self.lin2 = torch.nn.Linear(1024, 128)
        self.lin3 = torch.nn.Linear(128, self.num_class)
        
        self.att_q = torch.nn.Linear(2, 2)
        self.att_k = torch.nn.Linear(2, 2)

   
        
    def forward(self, data_sample, data_TF):
        #get mut exp data
        x_sample, edge_index_sample, batch_sample = data_sample.x, data_sample.edge_index, data_sample.batch
        x_TF, edge_index_TF, batch_TF = data_TF.x, data_TF.edge_index, data_TF.batch
        
        #load adj_norm
        with open('./data/preprocessed/netics_adj_norm_sp.pkl', 'rb') as f:
            netics_adj_norm = pickle.load(f)
        with open('./data/preprocessed/netics_adj_trans_norm_sp.pkl', 'rb') as f:
            netics_adj_trans_norm = pickle.load(f)
        
        batch_size = data_sample.batch[-1].item() + 1
        print(batch_size)
        #get emb_mut emb_exp
        emb_mut = self.emb_mut*x_sample+self.bias_mut        
        emb_exp = self.emb_exp*x_TF+self.bias_exp
        emb_mut = F.relu(emb_mut)
        emb_exp = F.relu(emb_exp)
        #print("mut_a",self.emb_mut)
        #print("mut_b",self.bias_mut)
        #print("exp_a",self.emb_exp)
        #print("exp_b",self.bias_exp)
        #match tensor dim
        x_sample = torch.reshape(emb_mut, (batch_size, self.num_genes)).T
        x_TF = torch.reshape(emb_exp, (batch_size, self.num_genes)).T
        
        h_sample = x_sample
        h_TF = x_TF
        
        #pagerank
        for k in range(self.k):
            x_sample = x_sample*(1-self.alpha)
            x_sample = torch.sparse.mm(netics_adj_norm.to(self.device), x_sample)
            x_TF = x_TF*(1-self.alpha)
            x_TF = torch.sparse.mm(netics_adj_trans_norm.to(self.device), x_TF)
            
        x_sample += h_sample*self.alpha
        x_TF += h_TF*self.alpha

        #concat for original prediction batch by 6016*2
        x_all = torch.cat((x_sample.T, x_TF.T),1)
        """
        #for attention        
        x_sample = torch.reshape(x_sample.T,(batch_size, self.num_genes, 1))
        x_TF = torch.reshape(x_TF.T,(batch_size, self.num_genes, 1))
        ori = torch.cat((x_sample, x_TF),2)
        query = self.att_q(ori)
        key = self.att_k(ori)
        key = torch.transpose(key,1,2)
        att = torch.matmul(query,key)
        #masking? dim**0.5?
        score = nn.Softmax(dim=-1)(att)
        
        value, index = torch.topk(score,5,dim=-1)
        y = []
        z = []
        for i in range(batch_size):
            w = score[i]
            for j in range(self.num_genes):
                x = w[j]
                for k in range(5):
                    x[index[i][j][k]]=1
                y.append(x)
                q = torch.stack(y,dim=0)
            z.append(q)
            #v = torch.stack(z,dim=1)
        print(q.shape)
        
        for i in range(batch_size):
            for j in range(self.num_genes):
                for k in range(self.num_genes):
                    if score[i][j][k] > 0.18:
                        score[i][j][k] = 1
                    else:
                        score[i][j][k] = 0
        print(score)
                        
        #score1 = score[0].cpu().detach().numpy()
        #with open ('scoremat.pkl', 'wb') as f:
            #pickle.dump(score1,f)
        x_att = torch.matmul(score, query)
        #shape for prediction
        x_att = torch.transpose(x_att,1,2)
        x_split = torch.split(x_att,1,1)
        a = torch.squeeze(x_split[0],1)
        b = torch.squeeze(x_split[1],1)
        x_all = torch.cat((a,b),1)    
        
        # for random experiments
        #x_sampler = torch.reshape(x_sample, (self.num_genes, 1, batch_size))
        #x_TFr = torch.reshape(x_TF, (self.num_genes, 1, batch_size))
        #randp = torch.cat((x_sampler, x_TFr),1)
        #np.random.seed(8)
        #rand = np.random.randint(self.num_genes, size=3008)
        #randpp = randp[rand]
        #randppp = torch.split(randpp, 1, 1)
        #a = torch.squeeze(randppp[0], 1).T
        #b = torch.reshape(randppp[1], 1).T
        #randl = torch.cat((a,b),1)
        #print(rand)
        #print(x_TF.shape)
        
        #data dim for pooling
        #x_sample_p = torch.flatten(x_sample, start_dim=0)
        #x_sample_p = torch.reshape(x_sample_p,(self.num_genes*batch_size,1))
        #x_TF_p = torch.flatten(x_TF, start_dim=0)
        #x_TF_p = torch.reshape(x_TF_p,(self.num_genes*batch_size,1))
        #x = torch.cat((x_sample_p, x_TF_p),1)
        #x = x.T
        #print(x.shape)

        #x,_,_,batch,_,_= self.pool(x, edge_index_sample,None,batch_sample) 
        #print(x.shape)
        #x = torch.cat([gap(x, batch)], dim=1)
        #x = torch.split(x, 1, 1)
        #a = x[0]
        #b = x[1]
        #a = torch.flatten(a, start_dim=0)
        #a = torch.reshape(a, (batch_size, 3008))
        #b = torch.flatten(b, start_dim=0)
        #b = torch.reshape(b, (batch_size, 3008))
        #x = torch.cat((a,b),1)
        #print(x.shape)
        #xt = torch.cat((randl, x_sample.T, x_TF.T),1)
        
        #x = torch.cat([x],dim=1)
        #print(x.shape)
        """
        
        #MLP
        xt = F.relu(self.lin1(x_all))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        xt = F.relu(self.lin2(xt))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        xt = self.lin3(xt)
        #print(x.shape)
        return xt 
    
class pagerank_explain(torch.nn.Module):
    def __init__(self, num_genes, num_class,k=1, alpha=0.1,dropout=0.5):
        
        super(pagerank_explain,self).__init__()
        print("pagerank_explain: K:{}, alpha:{}, dropout:{}".format(k,alpha,dropout))
        
        torch.manual_seed(0)

        self.num_genes = num_genes
        self.num_genes2 = num_genes*2
        self.num_class = num_class
        self.k = k
        self.alpha = alpha
        self.dropout = dropout
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
        self.emb_mut = nn.Parameter(torch.rand(1))
        self.bias_mut = nn.Parameter(torch.rand(1))
        self.emb_exp = nn.Parameter(torch.rand(1))
        self.bias_exp = nn.Parameter(torch.rand(1))

        #self.pool = TopKPooling(2, ratio=0.5)
        self.lin1 = torch.nn.Linear(self.num_genes, 1024)
        self.lin2 = torch.nn.Linear(1024, 128)
        self.lin3 = torch.nn.Linear(128, self.num_class)
        
    def forward(self, data_sample,data_TF,batch=True,mask=None,i=None):
        #get mut exp data
        data_sample = data_sample.to(self.device)
        data_TF = data_TF.to(self.device)
        x_sample, edge_index_sample= data_sample.x, data_sample.edge_index
        x_TF, edge_index_TF= data_TF.x, data_TF.edge_index
        
        #load adj_norm
        with open('./data/preprocessed/netics_adj_norm_sp.pkl', 'rb') as f:
            netics_adj_norm = pickle.load(f)
                
        with open('./data/preprocessed/netics_adj_trans_norm_sp.pkl', 'rb') as f:
            netics_adj_trans_norm = pickle.load(f)
        
        if batch:
            batch_size = data_sample.batch[-1].item() + 1
        else:
            batch_size = 1
            
        #get emb_mut emb_exp
        emb_mut = self.emb_mut*x_sample+self.bias_mut        
        emb_exp = self.emb_exp*x_TF+self.bias_exp
        emb_mut = F.relu(emb_mut)
        emb_exp = F.relu(emb_exp)
        x_sample = torch.reshape(emb_mut, (batch_size, self.num_genes)).T
        x_TF = torch.reshape(emb_exp, (batch_size, self.num_genes)).T
        h_sample = x_sample
        h_TF = x_TF
        
        #pagerank
        for k in range(self.k):
            x_sample = x_sample*(1-self.alpha)
            x_sample = torch.sparse.mm(netics_adj_norm.to(self.device), x_sample)
            x_TF = x_TF*(1-self.alpha)
            x_TF = torch.sparse.mm(netics_adj_trans_norm.to(self.device), x_TF)

        x_sample += h_sample*self.alpha
        x_TF += h_TF*self.alpha
        
        #hardamard product
        hardamard = torch.mul(x_sample,x_TF).T
        #print(hardamard.shape)

        if mask is not None:
            #mask = mask.to(self.device)
            #x_ori = torch.cat((x_sample.T,x_TF.T),0).T
            x_mask = torch.mul(hardamard.T, mask)
            #x_all = torch.cat((x_mask.T[0],x_mask.T[1]),0)
            #x_all = torch.unsqueeze(x_all,0)
            x_all = x_mask.T
            act = x_all.cpu().detach().numpy()
            with open('./her2.pkl'.format(i),'wb') as f:
                pickle.dump(act,f)
            
            #act_top10 = torch.topk(x_all,10,dim=-1).indices.cpu().detach().tolist()[0]
            #with open('./result/class0/act_top10/{}'.format(i),'wb') as f:
                #pickle.dump(act_top10,f)
            #x_all = torch.mul(hardamard.T,mask).T
            
        #concat for original prediction batch by 6016*2
        else:
            #x_all = torch.cat((x_sample.T,x_TF.T),1)
            x_all = hardamard
        #x_all = F.dropout(x_all,p=self.dropout,training=self.training)
        #MLP
        xt = F.relu(self.lin1(x_all))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        xt = F.relu(self.lin2(xt))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        xt = self.lin3(xt)
        #
        #print('output:',xt)
        return xt