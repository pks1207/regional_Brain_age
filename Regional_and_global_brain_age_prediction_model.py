#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:36:36 2023
@author: Gilsoon Park, PhD

USC Stevens Neuroimaging and Informatics Institute
Keck School of Medicine of USC
University of Southern California
2025 Zonal Ave.
Los Angeles, CA 90033
Email: gp_446@usc.edu
"""


import tensorflow as tf
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from lib import graph, coarsening

# test dataset

test_list = [
'./test_dataset/civet_01_features_20k.txt',
'./test_dataset/civet_02_features_20k.txt',
'./test_dataset/civet_03_features_20k.txt']


all_target_features20k_list = [pd.read_table(f'{i}',header=None).values for i in test_list]

all_target_features20k = abs(np.stack(all_target_features20k_list,axis=0))
all_target_features5k = np.concatenate((all_target_features20k[:,0:2562,:], all_target_features20k[:,10242:10242+2562,:]), axis=1)
all_target_features1k = np.concatenate((all_target_features20k[:,0:642,:], all_target_features20k[:,10242:10242+642,:]), axis=1)

### data normalization

x_test_1k = all_target_features1k
x_test_5k = all_target_features5k
x_test_20k = all_target_features20k
y_test= test_age

scaler = StandardScaler()

bn, vn, n_features = x_test_1k.shape

x_test_1k = x_test_1k.reshape([-1,n_features])

scaler.mean_= [3.12048644, 0.25537666]
scaler.scale_ = [0.44333759, 0.05053439]

x_test_1k = scaler.transform(x_test_1k)
x_test_1k = x_test_1k.reshape([bn, vn, n_features])

bn, vn, n_features = x_test_5k.shape

x_test_5k = x_test_5k.reshape([-1,n_features])

scaler.mean_= [3.12057338, 0.25527562]
scaler.scale_ = [0.44316179, 0.05075688]

x_test_5k = scaler.transform(x_test_5k)
x_test_5k = x_test_5k.reshape([bn, vn, n_features])

bn, vn, n_features = x_test_20k.shape

x_test_20k = x_test_20k.reshape([-1,n_features])

scaler.mean_= [3.12061775, 0.2552776 ]
scaler.scale_ = [0.44311141, 0.05075485]

x_test_20k = scaler.transform(x_test_20k)
x_test_20k = x_test_20k.reshape([bn, vn, n_features])

### networks definition

networks_labels = ['sns', 'fpn', 'drs', 'vnt', 'dft', 'slt', 'lng', 'adt', 'vsl', 'lmb', 'AD', 'global']
networks_edgesk = [5, 5, 20, 5, 5, 20, 20, 20, 5, 5, 5, 1]

aal_atlas_5k = pd.read_table(f'./surface_information/aal_atlas_5k.txt',header=None)
print(aal_atlas_5k.shape)

aal_atlas_20k = pd.read_table(f'./surface_information/aal_atlas_20k.txt',header=None)
print(aal_atlas_20k.shape)

# ROIs
sensorimotor = [1,2,19,20,57,58,69,70];
fpn = [5,6,7,8,9,10,65,66];
dorsal = [3,4,59,60];
ventral = [11,12,13,14,63,64];
default = [21,22,25,26,27,28,35,36,65,66,67,68,85,86];
salient = [29,30,31,32];
language = [11,12,13,17,63];
auditory = [79,80,81,82];
visual = [43,44,45,46,47,48,49,50,51,52,53,54,55,56,89,90];
limbic = [15,16,23,24,33,34,39,40,83,84,87,88];

roi_n = 10 # The number of ROIs

networks_10rois = [sensorimotor, fpn, dorsal, ventral, default, salient, language, auditory, visual, limbic]

networks_5k_lr = np.zeros((aal_atlas_5k.shape[0], roi_n))
networks_20k_lr = np.zeros((aal_atlas_20k.shape[0], roi_n))

for i in range(len(networks_10rois)):
    
    target_network = networks_10rois[i]
    
    for j in range(len(target_network)):
        
        networks_5k_lr[(aal_atlas_5k==target_network[j])[0], i] = i+1
        networks_20k_lr[(aal_atlas_20k==target_network[j])[0], i] = i+1
        
    
AD_networks_5k = pd.read_table(f'./surface_information/AD_signature_region_5k.txt',header=None)
AD_networks_5k = AD_networks_5k[0]

### brain age prediction

test_n = x_test_1k.shape[0]
y_pred_all_net = np.zeros((12, test_n))

k_fold=5

for net in range(12):

    net_n = net

    if networks_labels[net] == 'global':
        graph_edges = pd.read_table('./surface_information/edges1k.txt',names =['node1','node2'],encoding='UTF-8')
        num_nodes = graph_edges.iloc[:,0].max()+1
        x_test = x_test_1k

    elif networks_labels[net] == 'AD':
        networks_lr = AD_networks_5k
        graph_edges = pd.read_table('./surface_information/edges5k.txt',names =['node1','node2'],encoding='UTF-8')
        num_nodes = np.sum(networks_lr==1)
        x_test = x_test_5k[:,(networks_lr==1),:]
        
    elif networks_edgesk[net] == 5:
        networks_lr = networks_5k_lr[:,net_n]
        graph_edges = pd.read_table('./surface_information/edges5k.txt',names =['node1','node2'],encoding='UTF-8')
        num_nodes = np.sum(networks_lr==net_n+1)
        x_test = x_test_5k[:,(networks_lr==net_n+1),:]
    
    else:
        networks_lr = networks_20k_lr[:,net_n]
        graph_edges = pd.read_table('./surface_information/edges20k.txt',names =['node1','node2'],encoding='UTF-8')
        num_nodes = np.sum(networks_lr==net_n+1)
        x_test = x_test_20k[:,(networks_lr==net_n+1),:]
        
    
    print('network vertex number: ', num_nodes)
    print('final shape')
    print('test: ', x_test.shape)

    y_delta_all = 0
    y_pred_all = 0

    for fold_num in range(1,6):
    
        if networks_edgesk[net] == 1:
            filename='train_age_korbb_global_brain_age_f%d'%(fold_num)
        else:
            filename='train_age_korbb_regional_brain_age_%s_f%d'%(networks_labels[net], fold_num)
            
        print('----- ', filename, ' -----')

        # edge information

        # make some random dataadd column
        temp = np.load('./network_information/' + filename + ".npz", allow_pickle=True)

        gg = temp['gg']
        adj_matrix = temp['adj_matrix']
        cfg = temp['cfg']

        graphs = cfg[0];
        perm = cfg[1]

        L = [graph.laplacian(A, normalized=True).astype('float32') for A in graphs]  # Laplacian graph

        findex = [0,1] # feature index
        
        x_test_data = coarsening.perm_data(x_test[:,:,findex], perm)
        
        x_test_data = x_test_data.astype('float32')

        # optimal weights load
        weights_path = './weights/' + filename

        #
        optimal_list = !ls {weights_path}/model_optimal*
        print(optimal_list[-1])

        #
        tf.reset_default_graph()

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True)))
        new_saver = tf.train.import_meta_graph(optimal_list[-1])
        new_saver.restore(sess, optimal_list[-1][:-5])

        graphs = tf.get_default_graph()
        graphs.get_operations()

        input_data = graphs.get_tensor_by_name("inputs/data:0")
        input_data_shape = input_data.shape

        #Now, access the op that you want to run. 
        op_to_restore = graphs.get_tensor_by_name("logits/add:0")

        #
        from sklearn.linear_model import HuberRegressor

        model = HuberRegressor()

        #
        test_size = len(x_test_data)
        y_test_pred = np.zeros((test_size, ))
        
        for i in range(test_size):

            test_data = x_test_data[i:(i+1)]

            if input_data_shape[0] == 1:
                test_data_final = test_data
            else:
                for s in range(input_data_shape[0]):
                    if s == 0:
                        test_data_final = test_data
                    else:
                        test_data_final = np.concatenate((test_data_final, test_data), axis=0)

            feed_dict = {input_data: test_data_final}
            y_test_pred_temp = sess.run(op_to_restore, feed_dict)
            y_test_pred[i:(i+1)] = y_test_pred_temp[0]
                        
        y_pred_all += y_test_pred

    y_pred_all = y_pred_all/5
    y_pred_all_net[net, :] = y_pred_all
    
### brain ages save
save_table = pd.DataFrame(np.transpose(y_pred_all_net))
save_table.columns = networks_labels
save_table.to_csv('Regional_and_global_brain_ages.csv')
