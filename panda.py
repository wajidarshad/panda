# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:12:09 2020

@author: Wajid Abbasi
"""
import numpy as np
from Bio.Data import IUPACData
from sklearn.externals import joblib as sk_joblib

# code to genrate 2-mer features
def genrate_dict_mers(alphbets,mers):
    import itertools
    li = []
    for i in itertools.product(alphbets, repeat=mers):
        li.append(''.join(map(str, i)))
    prot_dic = dict((k, 0) for k in li)
    return prot_dic
    
def count_full_mers(seq,mers):
    prot_dic = genrate_dict_mers(IUPACData.protein_letters,mers)
    for aa in prot_dic:
        prot_dic[aa] =seq.count(aa)
    return prot_dic
def k_mers_features_prot_level(seq,k):#Simply Compute Amino Acid Composition
    feature=count_full_mers(seq,k).values()
    #feature=feature/np.linalg.norm(feature)
    return list(feature)
#********************************
#Code to normalize features
def mean_varrianace_normalization_single(examples):
    feats_mean=np.load('2mer_ungoup_feats_mean.npy')
    feats_std=np.load('2mer_ungoup_feats_std.npy')
    if np.all((examples == 0)):
        pass
    else:
        examples=(examples-feats_mean)/(feats_std)
        examples=(examples/np.linalg.norm(examples))
    return examples
#****************
    

def predict_affinity(WT_seq1,WT_seq2,M_seq1,M_seq2):
    feats_1=k_mers_features_prot_level(WT_seq1,2)
    feats_2=k_mers_features_prot_level(WT_seq2,2)
    feats_3=k_mers_features_prot_level(M_seq1,2)
    feats_4=k_mers_features_prot_level(M_seq2,2)
    feats_WT=np.concatenate((feats_1,feats_2), axis=0)
    feats_M=np.concatenate((feats_3,feats_4), axis=0)
    final_feats=feats_WT-feats_M
    if np.all((final_feats == 0)):
        return 0
    else:
        feats=mean_varrianace_normalization_single(final_feats).reshape(1,-1)
        panda=sk_joblib.load('trained_model_affinity_change_2mer_full.pkl')
        return panda.predict(feats)
