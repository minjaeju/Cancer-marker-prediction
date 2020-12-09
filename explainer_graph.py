from math import sqrt

import torch
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph, to_networkx
import numpy as np
import pickle

EPS = 1e-15

class GNNExplainer(torch.nn.Module):
    
    coeffs = {
        'edge_size': 0.001,
        'edge_ent': 1.0,
    }
    
    
    def __init__(self, model, epochs=100, lr=0.001, log=True):
        super(GNNExplainer,self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
        self.adj_mask = torch.nn.Parameter(torch.randn(6016,6016,device=self.device))
        self.adj_trans_mask = torch.nn.Parameter(torch.randn(6016,6016,device=self.device))
        print(self.adj_mask.device)
    def __subgraph__(self, adj, adj_trans):
        adj = adj.to(self.device)
        adj_trans = adj_trans.to(self.device)
        
        masked_adj = torch.sparse.mm(adj,self.adj_mask)
        masked_adj_trans = torch.sparse.mm(adj_trans,self.adj_trans_mask)
        return masked_adj, masked_adj_trans
    
    def __graph_loss__(self, log_logits, pred_label):
        loss = -torch.log(log_logits[0,pred_label])

        r_adj = self.adj_mask.sigmoid()
        r_adj_trans = self.adj_trans_mask.sigmoid()

        loss = loss + self.coeffs['edge_size'] * r_adj.sum() + self.coeffs['edge_size']*r_adj_trans.sum()
        #print('second:',loss)
        
        loss[torch.isnan(loss)]=0.

        ent = -(r_adj+r_adj_trans)*torch.log(r_adj+r_adj_trans+EPS)-(1-r_adj-r_adj_trans)*torch.log(1-r_adj-r_adj_trans+EPS)
        ent[torch.isnan(ent)]=0.
        #print('mean:',ent.mean())
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        print('third:',loss)
        return loss
    
    def explain_graph(self, mut, exp, adj, adj_trans):

        self.model.eval()
        #self.__clear_masks__()

        # Only operate on a k-hop subgraph around `node_idx`.
        adj = adj.to(self.device)
        adj_trans = adj_trans.to(self.device)
                
        #masked_adj,masked_adj_trans=self.__subgraph__(adj, adj_trans)
        # Get the initial prediction.
        with torch.no_grad():
            log_logits= self.model(mut,exp,False)
            probs_Y = torch.softmax(log_logits, 1)
            print('softmax:',probs_Y)
            pred_label = probs_Y.argmax(dim=-1)
            print('predlabel:',pred_label)

        #print(self.mask_adj)
        #print(self.mask_adj_trans)
        #         optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
#                                      lr=self.lr)
        optimizer = torch.optim.Adam([self.adj_mask,self.adj_trans_mask],lr=self.lr)
        epoch_losses=[]
        for epoch in range(1, self.epochs + 1):
            epoch_loss=0
            optimizer.zero_grad()

            self.masked_adj = torch.matmul(adj,self.adj_mask)
            self.masked_adj_trans = torch.matmul(adj_trans,self.adj_trans_mask)
            
            self.masked_adj[torch.isnan(self.masked_adj)]=0
            self.masked_adj_trans[torch.isnan(self.masked_adj_trans)]=0
            print('masked_adj:',self.adj_mask)
            print('masked_adj_trans:',self.adj_trans_mask)
            log_logits= self.model(mut,exp,False,self.masked_adj.sigmoid(),self.masked_adj_trans.sigmoid())
            #print("log_logits:",log_logits)
            pred= torch.softmax(log_logits, 1)
            print("pred",pred)
            loss = self.__graph_loss__(pred, pred_label)
            print("loss:",loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
#             print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)

        #edge_mask = self.edge_mask.new_zeros(num_edges)
        mask_adj= self.masked_adj.detach().sigmoid()
        mask_adj_trans=self.masked_adj_trans.detach().sigmoid()
        masking_adj = mask_adj.cpu().detach().numpy()
        masking_adj_trans = mask_adj_trans.cpu().detach().numpy()
        with open('masking_adj.pkl', 'wb') as f:
            pickle.dump(masking_adj,f)
        with open('masking_adj_trans.pkl', 'wb') as f:
            pickle.dump(masking_adj_trans,f)
        #self.__clear_masks__()

        return mask_adj, mask_adj_trans,epoch_losses
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'    
    
    
class GNNExplainer_lite(torch.nn.Module):        
    
    
    coeffs = {
        'edge_size': 0.0001,
        'edge_ent': 1.0,
    }
    
    
    def __init__(self, model, epochs=500, lr=0.001, log=True):
        super(GNNExplainer_lite,self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
    def __set_masks__(self, x, init="normal"):
        N = x.x.size(0)
        self.mask = torch.nn.Parameter((torch.randn(N,1,device=self.device)*0.1).uniform_(0,1))
        #std = torch.nn.init.calculate_gain('relu')*sqrt(2.0/(2*N))

    def __graph_loss__(self, log_logits, pred_label):
        loss = -torch.log(log_logits[0,pred_label])
        #print("loss:",loss)
        r_mask = self.mask.sigmoid()
        #print("r_mask:",r_mask)
        loss = loss + self.coeffs['edge_size'] * r_mask.sum()

        ent = -(r_mask)*torch.log(r_mask+EPS)-(1-r_mask)*torch.log(1-r_mask+EPS)

        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        #print('third:',loss)
        return loss
    
    def explain_graph(self, mut, exp, i):

        self.model.eval()
        #self.__clear_masks__()

        # Only operate on a k-hop subgraph around `node_idx`.
        self.__set_masks__(mut)
        #print("mask:",self.mask)
        #masked_adj,masked_adj_trans=self.__subgraph__(adj, adj_trans)
        # Get the initial prediction.
        #target = mut.y
        #print(target)
        with torch.no_grad():
            log_logits= self.model(mut,exp,False)
            probs_Y = torch.softmax(log_logits, 1)
            #print('softmax:',probs_Y)
            pred_label = probs_Y.argmax(dim=-1)
            print('predlabel:',pred_label)
        #if pred_label == target:
            #print("True")
        #print(self.mask_adj)
        #print(self.mask_adj_trans)
        #         optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
#                                      lr=self.lr)
        optimizer = torch.optim.Adam([self.mask],lr=self.lr)
        epoch_losses=[]
        for epoch in range(1, self.epochs + 1):
            epoch_loss=0
            optimizer.zero_grad()

            log_logits= self.model(mut,exp,False,self.mask,i)
            #print("log_logits:",log_logits)
            pred= torch.softmax(log_logits, 1)
            #print("pred",pred)
            loss = self.__graph_loss__(pred, pred_label)
            #print("loss:",loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
#             print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)

        #edge_mask = self.edge_mask.new_zeros(num_edges)
        mask_adj= F.relu(self.mask)
        masking_adj = mask_adj.cpu().detach().numpy()
        with open('masking_adj_rand.pkl', 'wb') as f:
            pickle.dump(masking_adj,f)

        return mask_adj,epoch_losses
    
    def visualize_subgraph(self, mask_adj, adj, k, target,num,**kwargs):
        mask_adj = torch.squeeze(mask_adj)
        #print(mask_adj.shape)
        subset = torch.topk(mask_adj,k,dim=0).indices.tolist()

        with open('./data/preprocessed/netics_node_mapping.pkl', 'rb') as f:
            mapping = pickle.load(f)
        
        ori_gene= []
        for i in subset:
            ori_gene.append(list(mapping.items())[list(mapping.values()).index(i)])
            
        with open('./result/class{}/ori_gene{}.pkl'.format(target,num), 'wb') as f:
            pickle.dump(ori_gene, f)        
            
        edge_index,_ = subgraph(subset,adj,relabel_nodes=True,num_nodes=6016)
        
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 700
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'
        
        if edge_index.size()[1] == 0:
            print("Nodes are not connected")
            subset,edge_index,_,_= k_hop_subgraph(subset,1,adj,relabel_nodes=True,num_nodes=6016)
            data = Data(edge_index=edge_index,num_nodes=len(subset))
            label_mapping = {k: i for k, i in enumerate(subset.tolist())}            
            sub_gene= []
            for i in subset:
                sub_gene.append(list(mapping.items())[list(mapping.values()).index(i)])
            with open('./result/class{}/sub_gene{}.pkl'.format(target,num), 'wb') as f:
                pickle.dump(sub_gene, f)
        else:
            data = Data(edge_index=edge_index,num_nodes=len(subset))
            label_mapping = {k: i for k, i in enumerate(subset)}
            sub_gene= []
            for i in subset:
                sub_gene.append(list(mapping.items())[list(mapping.values()).index(i)])
        G = to_networkx(data)
        G = nx.relabel_nodes(G,label_mapping)
                    
        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="-",
                        
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                    ))
        nx.draw_networkx_nodes(G,pos,**kwargs)
        nx.draw_networkx_labels(G, pos, **kwargs)
        plt.savefig('subgraph.png')
            

            
        return ori_gene,sub_gene
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'    