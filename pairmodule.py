# coding=utf-8
# Copyright 2020 Xubin ZHENG xbzheng@cse.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Commonly re-used modules including pair p-value"""

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as sm
from fisher import pvalue_npy
from tqdm import tqdm, trange

def pair_pvalue(x_train, y_train, diff, multitest="fdr"):
    
    """Prepare pair two-sided p-value by fisher's exact test.
    Args:
        x_train: a matrix [n_samples,n_genes].
        y_train: a boolean vecter, 1 for case, 0 for control.
        diff: value of size effect
        multitest: "fdr" or "bonferroni"
    Returns:
        pvalue: a matrix [n_genes,n_genes], p value
        a: gene1<gene2 in control
        b: gene1>gene2 in control
        c: gene1<gene2 in case
        d: gene1>gene2 in case
    """

    length = len(x_train[0,:])
    n_sm = len(x_train[:,0])
    
    # create tensor [n_sampels,length,length] that restores the subtraction between two genes
    result1 = np.zeros([length,length,n_sm],dtype=np.bool)
    result1_ = np.zeros([length,length,n_sm],dtype=np.bool)
    for i in tqdm(range(0,n_sm)):
        x = np.tile(x_train[i,:], (length,1))
        sub = x - x.T
        result1[:,:,i] = (sub > diff)
        result1_[:,:,i] = (sub < -diff)

    # prepare data for fisher exact test, a,c is greater, b,d is smaller
    # Control
    con_label = 1 - y_train
    a = np.sum(result1 * con_label, axis = 2)
    b = np.sum(result1_* con_label, axis = 2)
    # Case
    c = np.sum(result1 * y_train, axis = 2)
    d = np.sum(result1_* y_train, axis = 2)

    a_ = a.reshape((-1,1))
    a_ = np.squeeze(a_)
    a_ = a_.astype(np.uint)
    b_ = b.reshape((-1,1))
    b_ = np.squeeze(b_)
    b_ = b_.astype(np.uint)
    c_ = c.reshape((-1,1))
    c_ = np.squeeze(c_)
    c_ = c_.astype(np.uint)
    d_ = d.reshape((-1,1))
    d_ = np.squeeze(d_)
    d_ = d_.astype(np.uint)

    # fisher exact test
    _, _, twosided = pvalue_npy(a_, b_, c_, d_)

    # fdr or bonferroni
    if multitest == "bonferroni":
        rejected, pvalue_Bonf, alphacSidak, alphacBonf = sm.multipletests(twosided, alpha=0.05, method='bonferroni', 
                                                                      is_sorted=False, returnsorted=False)
        pvalue = pvalue_Bonf.reshape((length,length))
        
    elif multitest == "fdr":
        rejected, pvalue_fdr = sm.fdrcorrection(twosided, method='indep', is_sorted=False)
        pvalue = pvalue_fdr.reshape((length,length))
    
    else:
        pvalue = twosided.reshape((length,length))
        
    return pvalue,a,b,c,d


def p_screen(pvalue,threshold):
    
    """Prepare pair screening module using p-value.
    Args:
        pvalue: a matrix [n_genes,n_genes].
        threshold: a value.
    Returns:
        index of reversal pairs.
        length of reversal pairs.
    """
        
    # find indices of pvalue < threshold
    length = len(pvalue[0,:])
    # add 1 in triangle matrix to remove duplicated index
    pvalue_matrix = pvalue + np.triu(np.ones([length,length]))
    j,k = np.where(pvalue_matrix < threshold)
    idx = np.array([j,k],dtype = np.uint16).T
#     print ("pvalue_threshold = ", threshold)
#     print ("number of pairs = ",len(idx))
#     print ("number of genes = ", len(np.unique(idx)))
    return idx,len(idx)

def rev_screen(a,c,num_disease,num_control,threshold):
    """Prepare pair screening module using reversal 
        percentage in normal and disease.
    Args:
        a: no. of control samples with gene A>B,
            a vector [n_genes_pairs].
        c: no. of case samples with gene A>B,
            a vector [n_genes_pairs].
    Returns:
        index of reversal pairs.
        length of reversal pairs.
    """
    at = ((a.astype(float)/num_control) >= threshold)
    ct = ((c.astype(float)/num_disease) <= (1-threshold))
    idx, = np.where(at * ct)
    return idx,len(idx)

def nor_screen(a,c,num_disease,num_control,threshold):
    """Prepare pair screening module using reversal 
        percentage in normal and disease.
    Args:
        a: no. of control samples with gene A>B,
            a vector [n_genes_pairs].
        c: no. of case samples with gene A>B,
            a vector [n_genes_pairs].
    Returns:
        index of reversal pairs.
        length of reversal pairs.
    """
    at = ((a.astype(float)/num_control) >= threshold)
    ct = ((c.astype(float)/num_disease) <= (1-threshold))
    idx, = np.where(at * ct)
    return idx,len(idx)

def dftopair(df_data,idx1,idx2, diff):
    """Convert gene value to pair value.
    Args:
        df_data: dataframe [n_genes,n_samples],
                index:gene symbol.
        idx1: list [genes1 in pairs].
        idx2: list [genes2 in pairs].
        diff: effect size
    Returns:
        pair value: np.array of [n_samples,n_genes].
    """
    g1 = df_data.loc[idx1].to_numpy()
    g2 = df_data.loc[idx2].to_numpy()
    sub1 = g1 - g2 > diff
    sub2 = -1*(g1 - g2 < -diff)
    sub = sub1+sub2
    return sub.T

def nptopair(np_data,idx1,idx2, diff):
    """Convert gene value to pair value.
    Args:
        df_data: dataframe [n_samples,n_genes],
                index:gene symbol.
        idx1: list [genes1 in pairs].
        idx2: list [genes2 in pairs].
        diff: effect size
    Returns:
        pair value: np.array of [n_samples,n_genes].
    """
    g1 = np_data[:,idx1]
    g2 = np_data[:,idx2]
    sub1 = g1 - g2 > diff
    sub2 = -1*(g1 - g2 < -diff)
    sub = sub1+sub2
    return sub

def load_data(directory,zero_per):
    """
        load TCGA data, add gene symbol
    Args: 
        directory: directory of TCGA data *.tsv file.
    Returns:
        dftcga: pandas dataframe
    
    """
    dftcga = pd.read_csv(directory,sep="\t")
    # remove Ensembl ID version
    dftcga['Ensembl_ID'] = dftcga['Ensembl_ID'].str.split('.').str[0]
    
    # load ID and symbol from Ensembl database
    geneID = pd.read_csv('./data/ENSEMBL_SYMBOL_1.csv',usecols=[1,2])
    geneID.columns = ['Ensembl_ID','symbol']
    
    # left join with gene ID to obtain symbol and rearrange columns
    dftcga = pd.merge(dftcga, geneID, on='Ensembl_ID', how='left')
    dftcga.insert(loc = 1, 
              column = 'SYMBOL', 
              value = dftcga['symbol'])
    dftcga.drop(columns='symbol',inplace=True)
    # remove genes with more than 80% zero
    dftcga = dftcga.loc[(dftcga == 0).astype(int).sum(axis=1)<zero_per*len(dftcga.columns)]
    return dftcga

def pair_select(name,param,df,lab,diff,multitest,enhan,enname,pthreshold):
    df_res = pd.DataFrame(columns=['enhancer','gene1','gene2','p','a','b','c','d'])
    rev_pairs = []
    for i in tqdm(param):
#         try:
            # retrieve target genes
            lst = enhan[i]
            df_group = df[df['SYMBOL'].isin(lst)]
            glst = df_group['SYMBOL'].values.tolist()
            group = df_group.to_numpy()[:,2:].T
            if len(group[0])>1:
                pvalue,a,b,c,d = pair_pvalue(group, lab, diff, multitest)

                idx_0,idx_1 = np.where(pvalue==np.amin(pvalue))
#                 idx_0,idx_1 = np.where((a>=b) & (a!=0)) # to select all pair in each enhancer
                idx = np.array([idx_0,idx_1],dtype = np.uint16).T
                num = len(idx_0)
    
                if num>0:
                    for n in range(0,num):
                        rev_pairs.append([glst[idx[n,0]],glst[idx[n,1]]])
                        newrow = {'enhancer':enname[i,0],
                                  'gene1':glst[idx[n,0]],
                                  'gene2':glst[idx[n,1]],
                                  'p':pvalue[idx[n,0],idx[n,1]],
                                  'a':a[idx[n,0],idx[n,1]],
                                  'b':b[idx[n,0],idx[n,1]],
                                  'c':c[idx[n,0],idx[n,1]],
                                  'd':d[idx[n,0],idx[n,1]]}
                        df_res = df_res.append(newrow, ignore_index=True)
                        
#                 num = len(pvalue)
#                 if num>0:
#                     for k in range(0,num):
#                         for j in range(0,k):
#                             rev_pairs.append([glst[k],glst[j]])
#                             newrow = {'enhancer':enname[i,0],
#                                       'gene1':glst[k],
#                                       'gene2':glst[j],
#                                       'p':pvalue[k,j],
#                                       'a':a[k,j],
#                                       'b':b[k,j],
#                                       'c':c[k,j],
#                                       'd':d[k,j]}
#                             df_res = df_res.append(newrow, ignore_index=True)
                
                
    #         print('\r', "{:.2%}".format((i+1)*1.0/n_enhan), end='', flush=True)
            
#         except:
#             print("error i = ", i)
    return df_res
