import sys
import gzip
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, grid_search
import numpy as np
import pdb
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
#plt.rcParams['font.size'] = 15.0
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn2, venn2_circles
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
import random
import cPickle
import argparse
import stringrnautils
from math import*
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn import cross_validation
#import xgboost as xgb
import pandas as pd
import math
from sklearn.neighbors.kde import KernelDensity
from sklearn.cross_validation import train_test_split
from sklearn import mixture
from collections import Counter
from sklearn.cross_validation import KFold
from parseobo import obo_object
import scipy.spatial.distance as ssd
from scipy.stats import spearmanr as spearman
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
import venn


CUTOFF = 0.001#np.log2(1+0.5)
TRAIN_NUM = 50
SCALEUP = 1.0
SCALEDOWN = 1.0
WINDOW_SIZE = 35
DisGeNET_cutoff = 0.20
parser = argparse.ArgumentParser(description="""infer disease-associated lncRNAs based on lncRNA and mRNA co-expression from RNAseq""")

parser.add_argument('-ratio',
                    type=int, help='ratio is number of negative subsets to ensemble learning',
                    default=5)  # LJJ said to use this number instead :D

parser.add_argument('-file', help='input expression file', default='')
parser.add_argument('-outfile', help='outfile for infered disease associated lncRNAs')  
parser.add_argument('-type', default='mRNA', help='cross-validation using mRNA or predict for lncRNA')
parser.add_argument('-data', default=0, type=int, help='0: gencode; 1: GSE43520; 2:GSE30352; 3:GTEx')
parser.add_argument('-conf', help='extract mRNA-disease pair with at least this confidence', type=int, default=2) 
args = parser.parse_args()

'''
def allindices(string, sub, listindex=[], offset=0):
    i = string.find(sub, offset)
    while i >= 0:
        listindex.append(i)
        i = string.find(sub, i + 1)
    return listindex

import re
starts = [match.start() for match in re.finditer(re.escape('GG'), sss)]
'''
""" return euclidean distance between two lists """

def point_overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))

def euclidean_distance(x,y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp)
    sensitivity = float(tp)/ (tp+fn)
    specificity = float(tn)/(tn + fp)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 

def get_normalized_values_by_column(array, fea_length):
    max_col =[-100000] * fea_length
    min_col = [100000] * fea_length
    #for key in array.keys():
    #    indvidual_fea =  array[key]
    for values in array:
        for index in range(len(values)):
            if values[index] > max_col[index]:
                max_col[index] = values[index]
            if values[index] < min_col[index]:
                min_col[index] = values[index]
    for values in array:
        for index in range(len(values)):
            #print values[index],min_col[index], max_col[index]   
            values[index] = float(values[index] - min_col[index])/(max_col[index] - min_col[index]) 
    fw = open('saved_min_max', 'w')
    for val in min_col:
        fw.write('%f\t' %val)
    fw.write('\n')
    for val in max_col:
        fw.write('%f\t' %val)
    fw.write('\n')
    fw.close() 
    
def get_normalized_given_max_min(array):
    normalized_data = np.zeros(array.shape)
    tmp_data = np.loadtxt('saved_min_max')
    min_col = tmp_data[0, :]
    max_col = tmp_data[1, :]
    for x in xrange(array.shape[0]):
        for y in xrange(array.shape[1]):
            #print values[index],min_col[index], max_col[index]   
            normalized_data[x][y] = float(array[x][y] - min_col[y])/(max_col[y] - min_col[y])
    return normalized_data

def transfer_probability_class(result):
    y_pred = []
    for val in result:
        if val >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

def train_model(train_data, train_label, save_model_file, SVM = False):
    if SVM:
        parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 2, 3, 4, 5, 6, 10], 'gamma': [0.5,1,2,4, 6, 8]}
        svr = svm.SVC(probability = True)
        clf = grid_search.GridSearchCV(svr, parameters, cv=3)
    else:
        clf = RandomForestClassifier(n_estimators=10)
    clf.fit(train_data, train_label)
    with open(save_model_file, 'wb') as f:
        cPickle.dump(clf, f)
    
def predict_new_data(test_data, save_model_file, SVM = False):
    get_normalized_given_max_min(test_data)
    with open(save_model_file, 'rb') as f:
        clf = cPickle.load(f)  
    preds = clf.predict_proba(test_data)
    return preds[:, 1]

def get_banlanced_data(data, label, ratio = 1):
    inner_data = []
    inner_label = []
    posi_itemindex = np.where(label==1)[0]
    nega_item_index = np.where(label==0)[0]
    posi_size = len(posi_itemindex)
    random.shuffle(nega_item_index)
    nega_size = int(ratio * posi_size)
    extrated_ind = np.append(posi_itemindex, nega_item_index[:nega_size])
    random.shuffle(extrated_ind)
    inner_label = label[extrated_ind]
    inner_data = data[extrated_ind]    
    
    return inner_data, inner_label

def get_multiple_data(data, label, ratio = 5):
    inner_data = []
    inner_label = []
    posi_itemindex = np.where(label==1)[0]
    nega_item_index = np.where(label==0)[0]
    posi_size = len(posi_itemindex)
    random.shuffle(nega_item_index)
    nega_size = int(ratio * posi_size)
    #nega_id = nega_item_index[:nega_size]
    extrated_ind = np.append(posi_itemindex, nega_item_index[:nega_size])
    #random.shuffle(extrated_ind)
    inner_label = label[extrated_ind]
    inner_data = data[extrated_ind]    
    
    return inner_data, inner_label, posi_size  


def element_count(a):
    results = {}
    for x in a:
        if x not in results:
            results[x] = 1
        else:
            results[x] += 1
    return results

def check_lncRNA_mRNA_position(mRNA_list, lncRNA_list, gene_coordinate_dict, DIS_GAP = 500000, strand_speific = True):
    #pdb.set_trace()
    dist = 0
    nearby_flag = False
    locate_next_to_mRNA_list = []
    for lncRNA in lncRNA_list:
        lncRNA_chr_name, lncRNA_strand, lncRNA_start, lncRNA_end = gene_coordinate_dict[lncRNA]
        for mRNA in mRNA_list:
            mRNA_chr_name, mRNA_strand, mRNA_start, mRNA_end = gene_coordinate_dict[mRNA]
            if strand_speific:
                if lncRNA_chr_name + lncRNA_strand != mRNA_chr_name + mRNA_strand:
                    continue
            else:
                if lncRNA_chr_name  != mRNA_chr_name:
                    continue                
            start_gap = mRNA_start - lncRNA_start
            end_gap = lncRNA_end - mRNA_end
            if  (start_gap >= 0 and start_gap <= DIS_GAP) or (end_gap >= 0 and end_gap <= DIS_GAP):
                #pdb.set_trace()
                locate_next_to_mRNA_list.append(lncRNA)
                break
            
    return locate_next_to_mRNA_list            

def read_snp_dataset(snp_file = 'SNP/snp142Common.txt.gz'):
    snp_dict = {}
    with gzip.open(snp_file, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            key = values[1] + values[6]
            snp_dict.setdefault(key, set()).add(int(values[2]))
            
    return snp_dict

def read_snp_coordinate(snp_file = 'SNP/snp142Common.txt.gz'):
    snp_dict = {}
    with gzip.open(snp_file, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            #key = values[1] + values[6]
            snp_dict[values[4]] = values[1] + '_' + values[2] + '_' + values[6]
            
    return snp_dict

def read_GWAS_catalog(gwas_file = 'SNP/gwascatalog.txt', down_up_stream = False, cutoff=5000):
    gwas_snp = {}
    with open(gwas_file, 'r') as fp:
        head = True
        for line in fp:
            if head:
                head = False
                continue
            values = line.rstrip('\r\n').split('\t')
            chr_name = 'chr' + values[11]
            snp_id = values[21]
            coor = int(values[12])
            if down_up_stream: # upstream and downstream 500kb
                new_start = coor - cutoff
                new_end = coor + cutoff
                for val in range(new_start, new_end):
                    gwas_snp.setdefault(chr_name, set()).add(val)
            else:
                gwas_snp.setdefault(chr_name, set()).add(coor)
    
    return gwas_snp

def read_GWAS_catalog_disease(gwas_file = 'SNP/gwas_catalog_ensembl_mapping_v1.0-downloaded_2015-10-09.tsv'):
    gwas_dis = {}
    with open(gwas_file, 'r') as fp:
        head = True
        for line in fp:
            if head:
                head = False
                continue

            values = line.rstrip('\r\n').split('\t') 
            if values[7] == '' or values[-13] == '':
                continue
            disease = values[7]
            #except:
            #    pdb.set_trace()
            #if values[21] == '' or values[11] == '' or values[12] == '':
            #    continue
            #snp_id = values[21]
            #chr_name = 'chr' + values[11]
            #coor = int(values[12])
            #except:
            #pdb.set_trace()
            gwas_dis.setdefault(disease.upper(), set()).add(values[-13])
            #gwas_snp_coor[snp_id] = chr_name + '_' + values[12]
    #pdb.set_trace()
    return gwas_dis
            
def read_gwas_ld_region(gwas_ld_file = 'SNP/GWAS-LD-region-snps.csv'):
    #gwas_dis ={}
    snp_dis = {}
    with open(gwas_ld_file, 'r') as fp:
        head = True
        for line in fp:
            if head:
                head = False
                continue
            try:
                if '"' in line:
                    values = line.rstrip('\r\n').split('"')
                    SNP,GWAS_SNP,PMID = values[0][:-1].split(',')
                    diseases = values[1].split(',')
                    for dis in diseases:
                        snp_dis.setdefault(dis.upper(), []).append(SNP)
                        #gwas_dis.setdefault(dis.upper(), set()).add(GWAS_SNP)                        
                else:
                    SNP,GWAS_SNP,PMID,disease = line.rstrip('\r\n').split(',')
                    snp_dis.setdefault(disease.upper(), []).append(SNP)
                    #gwas_dis.setdefault(disease.upper(), set()).add(GWAS_SNP)
            except:
                pdb.set_trace()
            
    return snp_dis

def get_all_snp_disease_assoc():
    all_disease_snp = {}
    snp_coor_dict = read_snp_coordinate()
    ld_snp_dis = read_gwas_ld_region()
    gwas_dis = read_GWAS_catalog_disease()
    #pdb.set_trace()
    for key, val in gwas_dis.iteritems():
        for snp in val:
            if snp_coor_dict.has_key(snp):
                all_disease_snp.setdefault(key, set()).add(snp_coor_dict[snp])

    for key, val in ld_snp_dis.iteritems():
        for snp in val:
            if snp_coor_dict.has_key(snp):
                all_disease_snp.setdefault(key, set()).add(snp_coor_dict[snp])
                
    return all_disease_snp

def get_lcnRNA_disease_snp_assoc(gene_coordinate_dict, lncRNA_list, disease, dis_cutff = 250000):
    disease_snp_lncRNA = []
    all_disease_snp = get_all_snp_disease_assoc()
    #pdb.set_trace()
    disease = disease.upper()
    if all_disease_snp.has_key(disease):
        snp_coors = all_disease_snp[disease]
        for lncRNA in lncRNA_list:
            lncRNA_chr_name, lncRNA_strand, lncRNA_start, lncRNA_end = gene_coordinate_dict[lncRNA]
            for snp in snp_coors:
                chr_name, posi, strand = snp.split('_')
                #pdb.set_trace()
                lncRNA_chr_name = 'chr' + lncRNA_chr_name
                if chr_name != lncRNA_chr_name:
                    continue
                #pdb.set_trace()
                posi_start, posi_end = int(posi) - dis_cutff, int(posi) + dis_cutff
                if point_overlap(posi_start, posi_end, lncRNA_start, lncRNA_end) > 0:
                    print 'overlpa snps'
                    disease_snp_lncRNA.append(lncRNA)
    
    return disease_snp_lncRNA


def exist_snp_in_lncRNA(gene_coordinate_dict, lncRNA_list, gwas = False, down_up_stream = False):
    if gwas:
        snp_dict = read_GWAS_catalog(down_up_stream = down_up_stream)
    else:
        snp_dict = read_snp_dataset()
    snp_exist_in_lncRNA_list = []
    for lncRNA in lncRNA_list:
        lncRNA_chr_name, lncRNA_strand, lncRNA_start, lncRNA_end = gene_coordinate_dict[lncRNA] 
        if gwas:
            key = lncRNA_chr_name
        else:    
            key = lncRNA_chr_name + lncRNA_strand
       
        coor_set = set(range(lncRNA_start, lncRNA_end + 1))
        snp_coor_set = snp_dict[key]
        overlap_set = snp_coor_set & coor_set
        if len(overlap_set):
            snp_exist_in_lncRNA_list.append(1)
        else:
            snp_exist_in_lncRNA_list.append(0)
    
    return snp_exist_in_lncRNA_list


def get_kmeans_k_biggest_cluster(cluster_centers, labels, num_cluster):
    freq_dict = element_count(labels)
    freq_list  = []
    for key, val in freq_dict.iteritems():
        freq_list.append((val, key))
        
    freq_list.sort(reverse= True)
    new_array = []
    for val in freq_list[:num_cluster]:
        new_array.append(cluster_centers[val[1]])
    
    return np.array(new_array)

def get_multiple_data_based_clustering(data, label, ratio = 5):
    posi_itemindex = np.where(label==1)[0]
    nega_item_index = np.where(label==0)[0]
    posi_size = len(posi_itemindex)
    nega_data = data[nega_item_index]
    #nega_data = normalize(nega_data, axis=0)
    num_cluster = ratio*posi_size
    kmeans = KMeans(init='k-means++', n_clusters=num_cluster*4)
    kmeans.fit(nega_data)
    nega_new = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    #new_nega_ind = range(len(nega_new))
    #random.shuffle(new_nega_ind)
    #random.shuffle(nega_new)
    posi_data = data[posi_itemindex]
    nega_select = get_kmeans_k_biggest_cluster(nega_new, cluster_labels, num_cluster)
    #pdb.set_trace()
    inner_data = np.concatenate((posi_data, nega_select), axis=0)
    inner_label = [1]*posi_size + [0]*num_cluster
    #pdb.set_trace()
    return inner_data, inner_label, posi_size



def get_multiple_data_based_mRNAs(mRNA_list, data, label, ensg_ensp_map, ratio = 5, other_mRNA_list = None, negative_sampe_set = None,
                                  gene_position_dict = None):
    #print 'nultiuple mRNA uniq'
    inner_data = []
    inner_label = []
    posi_itemindex = np.where(label==1)[0]
    nega_item_index = np.where(label==0)[0]
    posi_size = len(posi_itemindex)
    random.shuffle(nega_item_index)
    #nega_size = int(2*ratio * posi_size) 
    #nega_index = nega_item_index[:nega_size]
    tmp_ensp = set()
    new_nega_ind = []
    if other_mRNA_list is None:
        print 'should excluding other mRNAs with low confidence related to disease'
        
    for ind in nega_item_index:
        mrna = mRNA_list[ind]
        mrna_ensp = ''
        if ensg_ensp_map.has_key(mrna):
            mrna_ensp = ensg_ensp_map[mrna]
        if mrna_ensp not in negative_sampe_set:
            continue
        if mrna_ensp in tmp_ensp:
            print mrna
            continue
        if mrna_ensp in other_mRNA_list:
            print mrna
            continue
        else:
            new_nega_ind.append(ind)
            if mrna_ensp != '':
                tmp_ensp.add(mrna_ensp)
                
        if len(new_nega_ind) >= ratio * posi_size:
            #print 'multiuple mRNA uniq'
            break
    
    extrated_ind = np.append(posi_itemindex, np.array(new_nega_ind))  
    inner_label = label[extrated_ind]
    inner_data = data[extrated_ind, :]  
    return inner_data, inner_label, posi_size 
'''
X -= np.mean(X, axis = 0) # zero-center 
X /= np.std(X, axis = 0)
'''    
def preprocess_data(X, scaler=None, minmax = True):
    if not scaler:
        if minmax:
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler                 

def preprocess_data_tissue(X):
    new_col = np.sum(X,1).reshape((X.shape[0],1))    
    X_new = X/X.sum(axis=1)[:, None] 
    X_new[np.isnan(X_new)] = 0
    X_new = np.append(X_new,new_col, axis=1)
    return X_new

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    '''KL divergence between GMM'''
    X = gmm_p.sample(n_samples)
    log_p_X, _ = gmm_p.score_samples(X)
    log_q_X, _ = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()

def fit_gmm(data):
    gmm = mixture.GMM(n_components=1)
    gmm.fit(data)
    
    return gmm

def plot_gaussian_distribution(h1, h2):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    h1.sort()
    hmean1 = np.mean(h1)
    hstd1 = np.std(h1)
    pdf1 = stats.norm.pdf(h1, hmean1, hstd1)
    ax.plot(h1, pdf1, label="lncRNA")

    h2.sort()
    hmean2 = np.mean(h2)
    hstd2 = np.std(h2)
    pdf2 = stats.norm.pdf(h2, hmean2, hstd2)
    ax.plot(h2, pdf2, label="PCG")
    legend = ax.legend(loc='upper right')
    
    plt.xlabel('FPKM')
    #plt.xlim(0,0.2)

    plt.show()

def get_gene_expression_in_tissue(expdata, tissues=None):
    '''
    cutoff value for RNA-seq    1    10    20    FPKMs
    
    '''
    fpkm_cutoff = 1
    result = []
    gmm = fit_gmm(expdata)
    
    #kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
def calculate_pcc_old(data, data_source):
    print 'calculating PCC distance'
    rows, cols = data.shape
    #data_pcc = np.zeros(rows, cols)
    pcc_list  = []
    #data = normalize(data, axis=0)
    scaler = StandardScaler()
        #scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    print data.shape
    for i in xrange(rows): # rows are the number of rows in the matrix. 
        pcc_list = pcc_list + [stats.pearsonr(data[i],data[j])[0] for j in range(i) if j != i]
        #pdb.set_trace()
        '''for j in xrange(i, rows):
            if i == j:
                continue
            rval = stats.pearsonr(data[i,:], data[j,:])[0]
            abs_rval = np.absolute(rval)
            pcc_list.append(abs_rval)
        '''
    #pdb.set_trace()
    print len(pcc_list)
    plot_hist_distance(pcc_list, 'PCC', data_source)        
    return pcc_list

def calculate_pcc_hist(mRNA_data, lncRNA_data):
    print 'calculating PCC distance'
    corr_pval = []
    corr_ind = []
    #pdb.set_trace()
    for i, mval in enumerate(lncRNA_data):
        tmp = []
        ind_j = []
        for j, lncval in enumerate(mRNA_data):
            rval, pval = stats.pearsonr(mval, lncval)
            abs_rval = np.absolute(rval)
            if  abs_rval > 0.3 and pval <= 0.01:
                corr_pval.append(abs_rval)
            
        #corr_pval.append(tmp)
    #print len(corr_pval)
    #plot_hist_distance(corr_pval, 'PCC', 'gencode')        #corr_ind.append(ind_j)
            
    return corr_pval

def calculate_cc(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices")

    XY = ssd.cdist(X, Y, 'correlation')

    return 1 - XY

def calculate_pcc_fast(A, B):
    #pdb.set_trace()
    A = A.T
    B = B.T
    N = B.shape[0]
    
    # Store columnw-wise in A and B, as they would be used at few places
    sA = A.sum(0)
    sB = B.sum(0)
    
    # Basically there are four parts in the formula. We would compute them one-by-one
    #p1 = N*np.einsum('ij,ik->kj',A,B)
    p1 = N*np.dot(B.T,A)
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)
    
    # Finally compute Pearson Correlation Coefficient as 2D array 
    pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None]))
    
    return pcorr


def calculate_pcc(mRNA_data, lncRNA_data):
    print 'calculating PCC distance'
    corr_pval = []
    corr_ind = []
    #pdb.set_trace()
    for i, mval in enumerate(lncRNA_data):
        tmp = []
        ind_j = []
        for j, lncval in enumerate(mRNA_data):
            rval, pval = stats.pearsonr(mval, lncval)
            abs_rval = np.absolute(rval)
            if  abs_rval > 0.3 and pval <= 0.01:
                tmp.append(abs_rval)
                ind_j.append(j)
            
        corr_pval.append(tmp)
        corr_ind.append(ind_j)
            
    return corr_pval, corr_ind

def coexpression_hist_fig(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw):
    posi_itemindex = np.where(mRNAlabels==1)[0]
    inner_data = disease_mRNA_data[posi_itemindex, :]  
    corr_pval = calculate_pcc_hist(inner_data, disease_lncRNA_data)
    return corr_pval

def coexpression_based_prediction(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw, k = 1):
    print 'k:', k
    posi_itemindex = np.where(mRNAlabels==1)[0]
    inner_data = disease_mRNA_data[posi_itemindex, :]  
    corr_pval, corr_ind = calculate_pcc(inner_data, disease_lncRNA_data)
    y_ensem_pred = []
    
    for val in corr_pval:
        if not len(val):
            y_ensem_pred.append(0)
        else:
            val.sort(reverse = True)
            sel_vals = val[:k]
            bigval = np.mean(sel_vals)
            y_ensem_pred.append(bigval)
    #pdb.set_trace()        
    fw.write('\t'.join(map(str, y_ensem_pred)))
    fw.write('\n')
    
    
def coexpression_knn_based_prediction(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw, k = 15):
    print 'k:', k 
    corr_pval= calculate_pcc_fast( disease_mRNA_data, disease_lncRNA_data)
    y_ensem_pred = []
    posi_itemindex = np.where(mRNAlabels==1)[0]
    num_len = len(corr_pval[0])
    #pdb.set_trace()
    for ind in range(len(corr_pval)):
        score = [abs(val) for val in corr_pval[ind]]
        num_inds = np.argsort(score)
        #val.sort(reverse = True)
        sel_vals = num_inds[num_len - k:]
        bigval = set(sel_vals) & set(posi_itemindex)
        knn_prob = len(bigval)
        y_ensem_pred.append(knn_prob)
    #pdb.set_trace()        
    fw.write('\t'.join(map(str, y_ensem_pred)))
    fw.write('\n')
'''
def predict_lncRNA_using_mRNA(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw, ensg_ensp_map={}, 
                            ratio = 5, SVM = False, roc_plot=False, weight =1, other_mRNA_list=None, gene_position_dict = None, 
                            overlap_disease_mRNA_list = None, negative_from_other_disease = None, f_imp = None):
    #data, labels, posi_size = get_multiple_data(disease_mRNA_data, mRNAlabels, ratio=ratio)
    posi_nega_ratio = 1
    #posi_data, nega_data, posi_size = get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, ratio=posi_nega_ratio*ratio)
    #data, labels, posi_size =  get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, ratio=posi_nega_ratio*ratio, other_mRNA_list = other_mRNA_list)
    if negative_from_other_disease is not None:
        data, labels, posi_size = get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, 
                        ratio=ratio*posi_nega_ratio, other_mRNA_list = other_mRNA_list, negative_sampe_set = negative_from_other_disease,
                        gene_position_dict = gene_position_dict)
    else:
        raise 'should select negative from other disease'
    #posi_data, nega_data, posi_size = get_multiple_data_based_clustering(disease_mRNA_data, mRNAlabels, ratio = ratio)
    #data = normalize(data, axis=0)
    fea_len = len(data[0])
    
    data, scaler = preprocess_data(data.transpose())
    data = data.transpose()
    disease_lncRNA_data, scaler = preprocess_data(disease_lncRNA_data.transpose())
    disease_lncRNA_data = disease_lncRNA_data.transpose()
    
    data = preprocess_data_tissue(data)
    disease_lncRNA_data = preprocess_data_tissue(disease_lncRNA_data)
    
    get_normalized_values_by_column(data, fea_len)
    #get_normalized_values_by_column(data, fea_len)
    normalized_data = get_normalized_given_max_min(disease_lncRNA_data)    
    #normalized_data = xgb.DMatrix( normalized_data, missing=-999)
    posi_data = data[:posi_size]
    nega_data = data[posi_size:]
    ntress = 10

    #y_ensem_pred = [0] * len(disease_lncRNA_data)
    y_impotance = np.zeros(fea_len)
    y_ensem_pred = np.zeros(len(disease_lncRNA_data))
    nega_size = posi_size * posi_nega_ratio
    #y_ensem_pred = []
    for ind in range(ratio): 
        #train = np.vstack((data[:posi_size], data[(ind + 1)*posi_size:(ind+2)*posi_size]))
        #pdb.set_trace() 
        train = np.vstack((posi_data, nega_data[ind*nega_size:(ind + 1)*nega_size])) 
        #print train.shape
        train_label = [1] *posi_size + [0] * nega_size
        
        if SVM:
            parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 2, 3, 4, 5, 6, 10], 'gamma': [0.5,1,2,4, 6, 8]}
            svr = svm.SVC(probability = True)
            clf = grid_search.GridSearchCV(svr, parameters, cv=3)
        else:
            clf = RandomForestClassifier(n_estimators=ntress)
        #loo = cross_validation.LeaveOneOut(len(train_label))
        #this_scores = cross_validation.cross_val_score(clf, train, np.array(train_label), cv = loo)
        #weight = np.mean(this_scores)
        
        clf.fit(train, train_label)
        y_impotance = y_impotance + clf.feature_importances_
        #if roc_plot:
        #    y_pred = clf.predict_proba(disease_lncRNA_data)[:, 1]    
        #else:
        y_pred = clf.predict_proba(normalized_data)[:, 1]
        #pdb.set_trace()
        y_ensem_pred = y_ensem_pred + y_pred/ratio
        #y_ensem_pred = [x + y/ratio for x,y in zip(y_ensem_pred, y_pred)]
        #y_ensem_pred.append(y_pred)

    if overlap_disease_mRNA_list is not None:
        overlap_lncRNA = check_lncRNA_mRNA_position(overlap_disease_mRNA_list, lncRNA_list, gene_position_dict, DIS_GAP = 500000, strand_speific = True)
        print len(overlap_lncRNA)
        y_ensem_pred_new  = []
        for score, lncRNA in zip(y_ensem_pred, lncRNA_list):
            if lncRNA in overlap_lncRNA:
                #y_ensem_pred_new.append(min(1.0, score*1.4))
                y_ensem_pred_new.append( score*1.5)
            else:
                y_ensem_pred_new.append(score*0.6)  
        y_ensem_pred =  y_ensem_pred_new      
                
    #snp_exist_in_lncRNA_list = exist_snp_in_lncRNA(gene_position_dict, lncRNA_list, gwas = True, down_up_stream = False)
    
    y_impotance = y_impotance/ratio  
      
    fw.write('\t'.join(map(str, y_ensem_pred)))
    fw.write('\n')
    if f_imp is not None:
        f_imp.write('\t'.join(map(str, y_impotance)))
        f_imp.write('\n')
'''
def rf_parameter_select(X,y):
    param_grid = {"min_samples_leaf": [1, 2, 3],
                'max_features': ['auto', 'sqrt', 'log2'],
              "n_estimators": [5, 10, 20, 50]}
    gs = grid_search.GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
    gs.fit(X, y)
    return gs    
    
def cross_validataion_lncRNA_using_mRNA(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw, ensg_ensp_map={}, 
                            ratio = 5, SVM = False, roc_plot=False, weight =1, other_mRNA_list=None, gene_position_dict = None, 
                            overlap_disease_mRNA_list = None, negative_from_other_disease = None, f_imp = None):
    #data, labels, posi_size = get_multiple_data(disease_mRNA_data, mRNAlabels, ratio=ratio)
    posi_nega_ratio = 1
    #posi_data, nega_data, posi_size = get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, ratio=posi_nega_ratio*ratio)
    #data, labels, posi_size =  get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, ratio=posi_nega_ratio*ratio, other_mRNA_list = other_mRNA_list)
    if negative_from_other_disease is not None:
        data, labels, posi_size = get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, 
                        ratio=ratio*posi_nega_ratio, other_mRNA_list = other_mRNA_list, negative_sampe_set = negative_from_other_disease,
                        gene_position_dict = gene_position_dict)
    else:
        raise 'should select negative from other disease'
    #posi_data, nega_data, posi_size = get_multiple_data_based_clustering(disease_mRNA_data, mRNAlabels, ratio = ratio)
    #data = normalize(data, axis=0)
    
    '''
    data, scaler = preprocess_data(data.transpose())
    data = data.transpose()
    disease_lncRNA_data, scaler = preprocess_data(disease_lncRNA_data.transpose())
    disease_lncRNA_data = disease_lncRNA_data.transpose()
    '''
    data = preprocess_data_tissue(data)
    disease_lncRNA_data = preprocess_data_tissue(disease_lncRNA_data)
    fea_len = len(data[0])
    #get_normalized_values_by_column(data, fea_len)
    #data, scaler = preprocess_data(data, minmax = False)
    #disease_lncRNA_data, scaler = preprocess_data(disease_lncRNA_data, scaler = scaler, minmax = False)
    
    get_normalized_values_by_column(data, fea_len)
    disease_lncRNA_data = get_normalized_given_max_min(disease_lncRNA_data)    

    posi_data = data[:posi_size]
    nega_data = data[posi_size:]
    #ntress = 10
    print len(posi_data), len(nega_data)
    #y_ensem_pred = [0] * len(disease_lncRNA_data)
    y_impotance = np.zeros(fea_len)
    y_ensem_pred = np.zeros(len(disease_lncRNA_data))
    nega_size = posi_size * posi_nega_ratio
    #y_ensem_pred = []
    for ind in range(ratio): 
        #train = np.vstack((data[:posi_size], data[(ind + 1)*posi_size:(ind+2)*posi_size]))
        #pdb.set_trace() 
        train = np.vstack((posi_data, nega_data[ind*nega_size:(ind + 1)*nega_size])) 
        #print train.shape
        train_label = [1] *posi_size + [0] * nega_size
        
        if SVM:
            parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 2, 3, 4, 5, 6, 10], 'gamma': [0.5,1,2,4, 6, 8]}
            svr = svm.SVC(probability = True)
            clf = grid_search.GridSearchCV(svr, parameters, cv=3)
        else:
            #clf = RandomForestClassifier(n_estimators=ntress)
            gs = rf_parameter_select(train, train_label)
            #clf = RandomForestClassifier().set_params(**clf.best_params_)
            clf = gs.best_estimator_
        #loo = cross_validation.LeaveOneOut(len(train_label))
        #this_scores = cross_validation.cross_val_score(clf, train, np.array(train_label), cv = loo)
        #weight = np.mean(this_scores)
        
        
        #pdb.set_trace()
        #clf.fit(train, train_label)
        y_impotance = y_impotance + clf.feature_importances_
        #if roc_plot:
        #    y_pred = clf.predict_proba(disease_lncRNA_data)[:, 1]    
        #else:
        y_pred = clf.predict_proba(disease_lncRNA_data)[:, 1]
        #pdb.set_trace()
        y_ensem_pred = y_ensem_pred + y_pred/ratio
        #y_ensem_pred = [x + y/ratio for x,y in zip(y_ensem_pred, y_pred)]
        '''y_ensem_pred = [x + (weight*y + (1-weight)*(1-y))/ratio for x,y in zip(y_ensem_pred, y_pred)]
        max_val = max(y_ensem_pred)
        min_val = min(y_ensem_pred)
        gap_Val = max_val - min_val
        y_ensem_pred = [(float(i) - min_val)/gap_Val for i in y_ensem_pred]
        '''
        #y_ensem_pred.append(y_pred)

    if overlap_disease_mRNA_list is not None:
        overlap_lncRNA = check_lncRNA_mRNA_position(overlap_disease_mRNA_list, lncRNA_list, gene_position_dict, DIS_GAP = 500000, strand_speific = True)
        print len(overlap_lncRNA)
        y_ensem_pred_new  = []
        for score, lncRNA in zip(y_ensem_pred, lncRNA_list):
            if lncRNA in overlap_lncRNA:
                #y_ensem_pred_new.append(min(1.0, score*1.6))
                new_score = score*SCALEUP
                #if new_score > 1:
                #    y_ensem_pred_new.append( 1 + (new_score -1)*0.5)
                #else:
                y_ensem_pred_new.append(new_score)
            else:
                y_ensem_pred_new.append(score*SCALEDOWN)  
        y_ensem_pred =  y_ensem_pred_new      
                
    #snp_exist_in_lncRNA_list = exist_snp_in_lncRNA(gene_position_dict, lncRNA_list, gwas = True, down_up_stream = False)
    
    y_impotance = y_impotance/ratio  
      
    fw.write('\t'.join(map(str, y_ensem_pred)))
    fw.write('\n')
    if f_imp is not None:
        f_imp.write('\t'.join(map(str, y_impotance)))
        f_imp.write('\n')
    #return y_real_all, y_pred_all     

def cross_validataion_lncRNA_using_mRNA_xgtboost(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, 
                                        mRNA_list, fw, ensg_ensp_map={}, ratio = 5, SVM = False, roc_plot=False, weight =1, other_mRNA_list=None):
    #data, labels, posi_size = get_multiple_data(disease_mRNA_data, mRNAlabels, ratio=ratio)
    posi_nega_ratio = 1
    #posi_data, nega_data, posi_size = get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, ratio=posi_nega_ratio*ratio)
    data, labels, posi_size =  get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, 
                                                             ratio=posi_nega_ratio*ratio, other_mRNA_list = other_mRNA_list)

    fea_len = len(data[0])
    get_normalized_values_by_column(data, fea_len)
    #get_normalized_values_by_column(data, fea_len)
    normalized_data = get_normalized_given_max_min(disease_lncRNA_data)
    #normalized_data = xgb.DMatrix( normalized_data, label=np.random.randint(2, size=normalized_data.shape[0]))
    normalized_data = xgb.DMatrix( normalized_data)
    posi_data = data[:posi_size]
    nega_data = data[posi_size:]
    
    param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
    param['nthread'] = 4
    plst = param.items()
    #plst += [('eval_metric', 'auc')] # Multiple evals can be handled in this way
    #y_ensem_pred = [0] * len(disease_lncRNA_data)
    y_ensem_pred = np.zeros(len(disease_lncRNA_data))
    nega_size = posi_size * posi_nega_ratio
    #y_ensem_pred = []
    for ind in range(ratio): 
        train = np.vstack((posi_data, nega_data[ind*nega_size:(ind + 1)*nega_size])) 
        #print train.shape
        train_label = [1] *posi_size + [0] * nega_size
        train = xgb.DMatrix( train, label=np.array(train_label))

        evallist  = [(train,'train')]
        num_round = 10
        clf = xgb.train( plst, train, num_round, evallist )

        y_pred = clf.predict(normalized_data)
        #y_pred = clf.predict_proba(normalized_data)[:, 1]
        #pdb.set_trace()
        y_ensem_pred = y_ensem_pred + y_pred/ratio

    fw.write('\t'.join(map(str, y_ensem_pred)))
    fw.write('\n')

def validataion_multiple_mRNA(disease_mRNA_data, mRNAlabels, mRNA_list, fw, ensg_ensp_map ={}, ratio = 5, SVM = False, roc_plot=False, 
                                    other_mRNA_list=None, negative_from_other_disease = None):
    posi_nega_ratio = 1
    if negative_from_other_disease is not None:
        data, labels, posi_size = get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, 
                        ratio=ratio*posi_nega_ratio, other_mRNA_list = other_mRNA_list, negative_sampe_set = negative_from_other_disease)
    else:
        raise 'should select negative from other disease'
    #data, labels, posi_size = get_multiple_data_based_clustering(disease_mRNA_data, mRNAlabels, ratio = ratio* posi_nega_ratio)
    
    print 'data size: ', data.shape
    '''
    data = data.transpose()
    data, scaler = preprocess_data_tissue(data)
    data = data.transpose()
    '''
    data = preprocess_data_tissue(data)
    #pdb.set_trace()
    fea_len = len(data[0])
    #pdb.set_trace()
    get_normalized_values_by_column(data, fea_len)

    posi_data = data[:posi_size]
    ntress = 10
    nega_data = data[posi_size:]
    nega_size = posi_size * ratio 
    #1/5 as testing, 4/5 as training
    test_posi_num = int(0.2*posi_size)
    test_nega_num = int(0.2*nega_size)
    
    random.shuffle(posi_data)
    random.shuffle(nega_data)
    test_posi = posi_data[:test_posi_num]
    test_nega = nega_data[:test_nega_num]
    
    
    train_posi = posi_data[test_posi_num:]
    train_nega = nega_data[test_nega_num:]
    
    len_nega = len(train_nega)/ratio
    #posi_kf = KFold(len(train_posi), n_folds=ratio)
    #nega_kf = KFold(len(train_nega), n_folds=ratio)
    #X_train, X_test, y_train, y_test = train_test_split(posi_data, y, test_size=0.2, random_state=42)
    
    y_pred_all = []
    #y_pred_prob = []
    
    #test_data = np.concatenate((posi_data, nega_data[:nega_size]), axis=0)
    #test_labels = [1] *posi_size + [0] * nega_size

    X_test = np.concatenate((test_posi, test_nega), axis=0) 
    print X_test.shape
    y_real_all = [1] *len(test_posi) + [0] * len(test_nega)
    y_pred_prob = [0] * len(y_real_all)
    print 'independent testing'
    # other ratio -1 negative subset data
    for ind in range(ratio):
        #nega_data = data[(ind + 2)*posi_size:(ind + 3)*posi_size, :]
        X_train = []
        if ind != ratio -1:
            x_nega_data = train_nega[ind*len_nega : len_nega*(ind + 1)]
        else:
            x_nega_data = train_nega[ind*len_nega:]            
        #tmp_nega = nega_data[ind*posi_size:(ind + 1)*posi_size]
        #test = []
        X_train = np.concatenate((train_posi, x_nega_data), axis=0) 
        y_train = [1] *len(train_posi) + [0] * len(x_nega_data)
        
        #print X_train.shape
        
        #sub_nega_data  = nega_data[(ind + 1)*nega_size:(ind + 2)*nega_size]
        #new_data = []
        #new_data = np.vstack((sub_posi_data, sub_nega_data))
        #print new_data.shape
        #new_data = np.vstack((posi_data, nega_data))
        #clf = RandomForestClassifier(n_estimators=ntress)
        #clf.fit(X_train, y_train)
        gs = rf_parameter_select(X_train, y_train)
        clf = gs.best_estimator_
        #clf = RandomForestClassifier().set_params(**clf.best_params_)
        #clf = gs.best_estimator_
        y_pred = clf.predict_proba(X_test)[:, 1]
        y_pred_prob = [val1 + val2/ratio for val1, val2 in zip(y_pred_prob, y_pred)] 
    
    y_pred_all = [ 1 if x>=0.5 else 0 for x in y_pred_prob]
    #if y_ensemb >= 0.5:
    #    y_pred_all.append(1)
    #else:
    #    y_pred_all.append(0)
    #y_pred_prob.append(y_ensemb)
    
    if not roc_plot:    
        acc, precision, sensitivity, specificity, MCC  = calculate_performace(len(y_real_all), y_pred_all,  y_real_all)
        fw.write('\t'.join(map(str, [acc, precision, sensitivity, specificity, MCC])))
        fw.write('\nROC_label\t')
        fw.write('\t'.join(map(str, y_real_all)))
        fw.write('\nROC_probability\t')
        fw.write('\t'.join(map(str, y_pred_prob)))
        fw.write('\n')

def cross_validataion_multiple_mRNA(disease_mRNA_data, mRNAlabels, mRNA_list, fw, ensg_ensp_map ={}, ratio = 5, SVM = False, roc_plot=False, 
                                    other_mRNA_list=None, negative_from_other_disease = None):
    #data, labels = get_banlanced_data(data, labels)
    #data, labels, posi_size = get_multiple_data(disease_mRNA_data, mRNAlabels, ratio=ratio)
    #data, labels, posi_size = get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, ratio=ratio)
    posi_nega_ratio = 1
    if negative_from_other_disease is not None:
        data, labels, posi_size = get_multiple_data_based_mRNAs(mRNA_list, disease_mRNA_data, mRNAlabels, ensg_ensp_map, 
                        ratio=ratio*posi_nega_ratio, other_mRNA_list = other_mRNA_list, negative_sampe_set = negative_from_other_disease)
    else:
        raise 'should select negative from other disease'
    #data, labels, posi_size = get_multiple_data_based_clustering(disease_mRNA_data, mRNAlabels, ratio = ratio* posi_nega_ratio)
    
    print 'data size: ', data.shape
    data = data.transpose()
    data, scaler = preprocess_data(data)
    data = data.transpose()
    fea_len = len(data[0])
    #pdb.set_trace()
    get_normalized_values_by_column(data, fea_len)

    
    posi_data = data[:posi_size]
    ntress = 10
    nega_data = data[posi_size:]
    '''std_scale = preprocessing.StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)
    '''
    '''get_normalized_values_by_column(np.vstack((posi_data, nega_data)), fea_len)

    normalized_data = get_normalized_given_max_min(disease_lncRNA_data)
    posi_data = get_normalized_given_max_min(posi_data)
    nega_data = get_normalized_given_max_min(nega_data)
    '''
    #data = normalize(data, axis=0)
    y_pred_all = []
    y_real_all = []
    y_pred_prob = []
    #test_data = data[:2*posi_size, :]
    #test_labels = labels[:2*posi_size]
    nega_size = posi_size * posi_nega_ratio 
    test_data = np.concatenate((posi_data, nega_data[:nega_size]), axis=0)
    test_labels = [1] *posi_size + [0] * nega_size
    for fold in range(len(test_data)):
        train = []
        test = []
        
        train = [x for i, x in enumerate(test_data) if i != fold]
        test = [x for i, x in enumerate(test_data) if i == fold]
        train_label = [x for i, x in enumerate(test_labels) if i != fold]
        test_label = [x for i, x in enumerate(test_labels) if i == fold]

        if SVM:
            parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 2, 3, 4, 5, 6, 10], 'gamma': [0.5,1,2,4, 6, 8]}
            svr = svm.SVC(probability = True)
            clf = grid_search.GridSearchCV(svr, parameters, cv=3)
        else:
            clf = RandomForestClassifier(n_estimators=ntress)
        #pdb.set_trace()
        clf.fit(train, train_label)
        y_pred = clf.predict_proba(test)[:, 1] 
        
        y_ensemb = y_pred[0]/ratio   
        
        if test_labels[fold] == 1:
            #posi_data = np.array(train[:posi_size-1])
            sub_posi_data = np.array(train[:posi_size-1])  
            posi_label = [1] *(posi_size - 1)
        else:
            #posi_data = np.array(train[:posi_size])  
            sub_posi_data = np.array(train[:posi_size])
            posi_label = [1] * posi_size 
            
        new_label = posi_label + [0] *nega_size   
        #  
        # other ratio -1 negative subset data
        for ind in range(ratio - 1):
            #nega_data = data[(ind + 2)*posi_size:(ind + 3)*posi_size, :]
            sub_nega_data  = nega_data[(ind + 1)*nega_size:(ind + 2)*nega_size]
            new_data = []
            new_data = np.vstack((sub_posi_data, sub_nega_data))
            #print new_data.shape
            #new_data = np.vstack((posi_data, nega_data))
            clf = RandomForestClassifier(n_estimators=ntress)
            clf.fit(new_data, new_label)
            y_pred = clf.predict_proba(test)[:, 1]  
            y_ensemb = y_ensemb + y_pred[0]/ratio 
        
        if y_ensemb >= 0.5:
            y_pred_all.append(1)
        else:
            y_pred_all.append(0)
        y_real_all.append(test_labels[fold])
        y_pred_prob.append(y_ensemb)
        
    if not roc_plot:    
        acc, precision, sensitivity, specificity, MCC  = calculate_performace(len(y_real_all), y_pred_all,  y_real_all)
        fw.write('\t'.join(map(str, [acc, precision, sensitivity, specificity, MCC])))
        fw.write('\nROC_label\t')
        fw.write('\t'.join(map(str, y_real_all)))
        fw.write('\nROC_probability\t')
        fw.write('\t'.join(map(str, y_pred_prob)))
        fw.write('\n')
    
    #return y_real_all, y_pred_all  

def read_GPL_file(GPL_file):
    gene_map_dict = {}
    fp = open(GPL_file, 'r')
    for line in fp:
        if line[0] == '#' or line[0] == 'I':
            continue
        values = line.rstrip('\n').split('\t')
        refID = values[0]
        probID = values[3]
        gene_symbol = values[9]
        ensembleID = values[12]
        gene_map_dict[refID] = (probID, gene_symbol, ensembleID)
    fp.close()
    return gene_map_dict

def read_normalized_series_file(series_file, take_median=True):
    gene_map_dict = read_GPL_file('data/GSE34894/GPL15094-7646.txt')
    fp = gzip.open(series_file, 'r')
    expression_dict = {}
    #fw = open('gene_expression_file', 'w')
    for line in fp:
        if line[0] == '!' or len(line) < 10:
            continue
        #pdb.set_trace()
        if 'ID_REF' in line:
            sampel_ids = line.rstrip('\r\n').split('\t')[1:]
            sampel_ids = ['probID', 'gene_symbol', 'ensembleID'] + sampel_ids[1:]
            #fw.write('\t'.join(sampel_ids))
            continue
        values = line.rstrip('\r\n').split('\t')
        refID = values[0]
        probID, gene_symbol, ensembleID = gene_map_dict[refID]
        if ensembleID == '':
            continue
        expression_dict.setdefault(ensembleID, []).append([probID] + values[1:] + [gene_symbol])
        #fw.write('\t'.join(values))
    fp.close() 

    merge_probe_expression_dict = {}
    #num_of_tissue = 31
    for key,vals in expression_dict.iteritems():
            new_vals = []
            num_dup = len(vals)
            for single_val in vals:
                exp_vals = []
                exp_vals = [float(val) for val in single_val[1:-1]]
                #for index in range(num_of_tissue):
                new_vals.append(exp_vals)
                #new_vals = [x.append(float(y)) for x,y in zip(new_vals, exp_vals)]
                prob = single_val[0]
                gene_symbol = single_val[-1]
            new_vals = np.array(new_vals)
            #pdb.set_trace()
            final_express_vals = []
            if take_median:
                final_express_vals = np.median(new_vals, axis=0)
            else:
                final_express_vals = np.mean(new_vals, axis=0) 
            try:    
                merge_probe_expression_dict[key] = [prob, gene_symbol] + [inside_val for inside_val in final_express_vals]
            except:
                pdb.set_trace()
                print final_express_vals
                print prob, gene_symbol
    return merge_probe_expression_dict

def get_mean_expression_for_tissue_multiple_sampels(samples, expression_dict, use_mean = False, log2 = True):
    sample_set = set(samples)
    sample_list = [samp for samp in sample_set]
    aver_expr_vals = {}
    for key, vallist in expression_dict.iteritems():
        for sam in sample_set:
            ave_inds =  [i for i,val in enumerate(samples) if val==sam]
            if use_mean:
                mean_val = np.mean(map(vallist.__getitem__, ave_inds)) 
            else:
                mean_val = np.median(map(vallist.__getitem__, ave_inds)) 
            if log2:
                mean_val = np.log2(1 + mean_val) 
                
            aver_expr_vals.setdefault(key, []).append(mean_val) 
    expression_dict.clear()
    new_aver_expr_vals = {}
    for key, val in aver_expr_vals.iteritems():
        #if max(val) < CUTOFF:
        #    continue
        new_aver_expr_vals[key] = val
    
    return new_aver_expr_vals, sample_list  

def read_human_RNAseq_expression(RNAseq_file = 'data/gencodev7/genes.fpkm_table', gene_name_ensg = None, log2 = True):
    print 'read expresion file: ', RNAseq_file
    data_dict  = {}
    fp = open(RNAseq_file, 'r')
    head = True
    for line in fp:
        if head:
            head = False
            values = line.rstrip('\r\n').split('\t')[1:]
            samples = [val.split('_')[0] for val in values]
            continue
        else:
            values = line.rstrip('\r\n').split('\t')
            if log2:
                expval_list = [np.log2(1 + float(tmp_val)) for tmp_val in values[1:]]
            else:
                expval_list = [float(tmp_val) for tmp_val in values[1:]]
                
            #if max(expval_list) < CUTOFF:
            #    continue
                
            key = values[0].split('.')[0]
            #if gene_name_ensg.has_key(key):
            #    data_dict[gene_name_ensg[key]] = expval_list
            #else:
            data_dict[key] = expval_list
    fp.close()
    
    return data_dict, samples

def read_gtex_gene_map(map_file = 'data/gtex/xrefs-human.tsv'):
    entrid_ensemid = {}
    head = True
    with open(map_file, 'r') as fp:
        for line in fp:
            if head:
                head = False
                continue
            if 'Ensembl' in line:
                values = line.rstrip().split()
                entrid_ensemid[values[0]] = values[-1]
    return entrid_ensemid       
    

def read_gtex_expression(RNAseq_file = 'data/gtex/GTEx_Analysis_v6p_RNA-seq_RNA-SeQCv1.1.8_gene_median_rpkm.gct.gz', gene_type_dict = None, log2 = True):
    print 'read expresion file: ', RNAseq_file
    #entrid_ensemid = read_gtex_gene_map()
    data_dict  = {}
    fp = gzip.open(RNAseq_file, 'r')
    head = True
    #pdb.set_trace()
    for line in fp:
        if line[0] == '#':
            continue
        if '56238\t53' in line:
            continue
        if head:
            head = False
            values = line.rstrip('\r\n').split('\t')[2:]
            samples = [val for val in values]
            print '# of tissues', len(samples)
            continue
        else:
            values = line.rstrip('\r\n').split('\t')
            if log2:
                expval_list = [np.log2(1 + float(tmp_val)) for tmp_val in values[2:]]
            else:
                expval_list = [float(tmp_val) for tmp_val in values[2:]]
            #expval_list = [float(tmp_val) for tmp_val in values[2:]]
                
            #if max(expval_list) < CUTOFF:
            #    continue
                
            key = values[0].split('.')[0]
            #if entrid_ensemid.has_key(key):
            #    ensem_id = entrid_ensemid[key]
            #if key in gene_type_dict:
            data_dict[key] = expval_list
            #else:
            #    data_dict[key] = expval_list
    fp.close()
    #pdb.set_trace()
    return data_dict, samples

def read_evolutionary_expression_data(input_file = 'data/GSE43520/genes.fpkm_table', use_mean = False, log2 = True):
    print 'read expresion file: ', input_file
    data_dict  = {}
    fp = open(input_file, 'r')
    head = True
    for line in fp:
        if head:
            head = False
            values = line.rstrip('\r\n').split('\t')[1:]
            samples = [val.split('-')[0].lower() for val in values]
            continue
        else:
            values = line.rstrip('\r\n').split('\t')
            key = values[0].split('.')[0]
            data_dict[key] = [float(tmp_val) for tmp_val in values[1:]]            
    fp.close()
    #pdb.set_trace()
    data_dict, sample_set = get_mean_expression_for_tissue_multiple_sampels(samples, data_dict, use_mean = use_mean, log2 = log2)

    return data_dict, sample_set

def read_evolutionary_expression_data_old(input_file, use_mean = False, log2 = True):
    print 'read expresion file: ', input_file
    data_dict  = {}
    fp = open(input_file, 'r')
    head = True
    for line in fp:
        if head:
            head = False
            values = line.rstrip('\r\n').split('\t')[1:]
            samples = [val.split('_')[0] for val in values]
            continue
        else:
            values = line.rstrip('\r\n').split('\t')
            key = values[0]
            if 'blastn' in key:
                continue
            data_dict[key] = [float(tmp_val) for tmp_val in values[1:]]            
    fp.close()
    #pdb.set_trace()
    data_dict, sample_set = get_mean_expression_for_tissue_multiple_sampels(samples, data_dict, use_mean = use_mean, log2 = log2)
    
    return data_dict, sample_set

def read_average_read_to_normalized_RPKM(input_file = 'data/GSE30352/genes.fpkm_table', readlength=75, use_mean = False, log2 = True):
    print 'read expresion file: ', input_file
    fp =open(input_file, 'r')
    head = True
    express_vals = {}
    #gene_list = []
    for line in fp:
        if head:
            values = line.rstrip('\r\n').split('\t')[1:]
            samples = [val.split('-')[0] for val in values]
            head =False
            continue
        values = line.rstrip('\r\n').split('\t')
        gene = values[0].split('.')[0]
        #gene_list.append(gene)
        express_vals[gene] = [float(val) for val in values[1:]]
    #pdb.set_trace()    
    final_vals, sample_set =  get_mean_expression_for_tissue_multiple_sampels(samples, express_vals, use_mean = use_mean, log2 = log2) 
       
    return final_vals, sample_set   
     
def read_average_read_to_normalized_RPKM_old(input_file, readlength=75, use_mean = False, log2 = True):
    print 'read expresion file: ', input_file
    fp =open(input_file, 'r')
    head = True
    total_exp_vals = []
    gene_list = []
    exon_len = []
    for line in fp:
        if head:
            values = line.rstrip('\r\n').split('\t')[6:]
            samples = [val[1:-1].split('_')[1] for val in values]
            head =False
            continue
        values = line.rstrip('\r\n').split('\t')
        gene = values[0][1:-1]
        gene_list.append(gene)
        gene_start = int(values[2])
        gene_end = int(values[3])
        ExonicLength = int(values[5])
        exon_len.append(ExonicLength)
        express_vals = [float(val)*ExonicLength/readlength for val in values[6:]]
        total_exp_vals.append(express_vals)
    
    final_vals = {}    
    total_exp_vals = np.array(total_exp_vals)
    sum_val = sum(total_exp_vals)/1000000000
    for x in xrange(total_exp_vals.shape[0]):
        final_vals[gene_list[x]] = []
        for y in xrange(total_exp_vals.shape[1]): 
            final_vals[gene_list[x]].append(total_exp_vals[x][y]/(exon_len[y]*sum_val[y]))
    fp.close()
    #pdb.set_trace()
    #average value for the same tissue from different samples
    final_vals, sample_set =  get_mean_expression_for_tissue_multiple_sampels(samples, final_vals, use_mean = use_mean, log2 = log2)
    
    return final_vals, sample_set


def remove_redudancy_expression_data(expression_data, cutoff):
    redudancy_data = []
    #new_list=[]
    for i in range(len(expression_data)):
        keep_flag = True
        for j in range(i):
            if euclidean_distance(expression_data[i, :],expression_data[j, :]) <= cutoff:
                keep_flag = False
                break
        if keep_flag:        
            redudancy_data.append(expression_data[i, :])
    
    return redudancy_data


def extract_gene_type_name(input_val):
    input_val = input_val.strip()
    split_gene_name = input_val.split()
    gene = split_gene_name[1][1:-1]
    return gene

def read_gencode_gene_type():
    gene_type_dict = {}
    gene_name_ensg = {}
    gene_id_position = {}
    fp = gzip.open('data/dict/gencode.v19.genes.v6p_model.patched_contigs.gtf.gz')
    for line in fp:
        if line[0] == '#':
            continue
        values = line.rstrip('\r\n').split('\t')
        if values[2] != 'gene':
            continue
        gene_ann = values[-1]
        split_gene_ann = gene_ann.split(';')
        gene_type_info = split_gene_ann[2]
        gene_type = extract_gene_type_name(gene_type_info)
            
        gene_name_info = split_gene_ann[4]
        gene_name = extract_gene_type_name(gene_name_info)
        '''gene_type_dict[gene_name] = gene_type'''
        
        gene_id_info = split_gene_ann[0]
        gene_id = extract_gene_type_name(gene_id_info)
        gene_id =  gene_id.split('.')[0]
        gene_type_dict[gene_id] = gene_type        
        
        gene_name_ensg[gene_name] = gene_id
        
        chr_name = 'chr' + values[0]
        strand = values[6]
        start = int(values[3])
        end = int(values[4])
        
        #if not gene_id_position.has_key(gene_id):
        #    gene_id_position[gene_id] = (chr_name, strand, start, end)
        # gene has bigger length compared to exon    
        
        gene_id_position[gene_id] = (chr_name, strand, start, end)
        
    fp.close()
    
    return gene_type_dict, gene_name_ensg, gene_id_position
    
def read_GRCh37_gene_type():
    gene_type_dict = {}
    gene_name_ensg = {}
    gene_id_position = {}
    fp = gzip.open('data/dict/Homo_sapiens.GRCh37.70.gtf.gz')
    for line in fp:
        if line[0] == '#':
            continue
        values = line.rstrip('\r\n').split('\t')
        gene_ann = values[-1]
        split_gene_ann = gene_ann.split(';')
        gene_type_info = split_gene_ann[4]
        gene_type = extract_gene_type_name(gene_type_info)
            
        gene_name_info = split_gene_ann[3]
        gene_name = extract_gene_type_name(gene_name_info)
        '''gene_type_dict[gene_name] = gene_type'''
        
        gene_id_info = split_gene_ann[0]
        gene_id = extract_gene_type_name(gene_id_info)
        gene_type_dict[gene_id] = gene_type        
        
        gene_name_ensg[gene_name] = gene_id
        
        chr_name = values[0]
        strand = values[6]
        start = int(values[3])
        end = int(values[4])
        
        if not gene_id_position.has_key(gene_id):
            gene_id_position[gene_id] = (chr_name, strand, start, end)
        # gene has bigger length compared to exon    
        if values[2] == 'gene':
            gene_id_position[gene_id] = (chr_name, strand, start, end)
        
    fp.close()
    
    return gene_type_dict, gene_name_ensg, gene_id_position

def get_ENSP_ENSG_map():
    ensg_to_ensp_map = {}
    fp = open('data/dict/string_9606___10_all_T.tsv')
    for line in fp:
        species, ensg, ensp = line.rstrip('\n').split('\t')
        ensg_to_ensp_map[ensg] = ensp
    fp.close()
    
    return ensg_to_ensp_map

def get_ensg_ensp_map():
    ensp_to_ensg_map = {}
    fp = open('data/dict/string_9606_ENSG_ENSP_10_all_T.tsv')
    for line in fp:
        species, ensg, ensp = line.rstrip('\n').split('\t')
        ensp_to_ensg_map[ensp] = ensg
    fp.close()
    
    return ensp_to_ensg_map    

'''def read_ncRNA_alias():
    ncRNA_alias_identifier_map ={}
    fp = gzip.open('/home/panxy/eclipse/string-rna/id_dictionaries/ncRNAaliasfile.tsv.gz')  
    for line in fp:
        species, identity, alias, source = line.rstrip('\r\n').split('\t')
        if '9606' != species:
            continue
        else:
            ncRNA_alias_identifier_map[alias] = identity
    fp.close()         
    
    return ncRNA_alias_identifier_map
''' 

def get_more_evidence_DisGeNET(whole_disease_gene_dict):
    ensg_ensp = get_ENSP_ENSG_map()
    gene_type_dict, gene_name_ensg, gene_id_position = read_gencode_gene_type()
    with open('data/disease/consolidated_all.tsv', 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            disease = values[0]
            gene = values[3]
            if not gene_name_ensg.has_key(gene):
                        continue
            new_gene = gene_name_ensg[gene]
            if not ensg_ensp.has_key(new_gene):
                continue
            ensp = ensg_ensp[new_gene]
            whole_disease_gene_dict.setdefault(disease, set()).add(ensp)

def read_DISEASE_database(include_textming = False, confidence=2):
    #ensp_to_ensg_map = get_ENSP_ENSG_map()
    print confidence
    disease_gene_dict = {}
    whole_disease_gene_dict = {}
    disease_name_map = {}

    fp = open('data/disease/human_disease_integrated_full.tsv', 'r')
    for line in fp:
        if 'DOID:' not in line or 'ENSP' not in line:
            continue
        values = line.rstrip('\r\n').split('\t')
        gene = values[0]
        disease = values[2]
        disease_name = values[3]
        disease_name_map[disease] = disease_name.upper()
        whole_disease_gene_dict.setdefault(disease, set()).add(gene)
        conf = float(values[-1])
        if conf < confidence:
            continue  
        disease_gene_dict.setdefault(disease, set()).add(gene)
                 
    fp.close()
                 
    return disease_gene_dict, disease_name_map, whole_disease_gene_dict

def read_DISEASE_database_old(include_textming = False, confidence=2, include_DisGeNET_curated = True):
    #ensp_to_ensg_map = get_ENSP_ENSG_map()
    disease_gene_dict = {}
    whole_disease_gene_dict = {}
    disease_name_map = {}
    database_files = ['data/disease/human_disease_experiments_full.tsv', 'data/disease/human_disease_knowledge_full.tsv']
    if include_textming:
        database_files = database_files + ['data/disease/human_disease_textmining_full.tsv']
        
    for datafile in database_files:
        fp = open(datafile, 'r')
        for line in fp:
            if 'DOID:' not in line or 'ENSP' not in line:
                continue
            values = line.rstrip('\r\n').split('\t')
            gene = values[0]
            disease = values[2]
            disease_name = values[3]
            disease_name_map[disease] = disease_name.upper()
            whole_disease_gene_dict.setdefault(disease, set()).add(gene)
            if 'textmining' in datafile:
                conf = float(values[-2])
                if conf < 2.5:
                    continue                
            else:
                conf = float(values[-1])
                if conf < confidence:
                    continue
            #if ensp_to_ensg_map.has_key(gene):
            disease_gene_dict.setdefault(disease, set()).add(gene)
            #else:
            #    print gene
        
        fp.close()
    if not include_textming:
        fp = open('data/disease/human_disease_textmining_full.tsv', 'r')
        for line in fp:
            if 'DOID:' not in line or 'ENSP' not in line:
                continue
            values = line.rstrip('\r\n').split('\t')
            gene = values[0]
            disease = values[2]
            disease_name = values[3]
            disease_name_map[disease] = disease_name.upper()
            whole_disease_gene_dict.setdefault(disease, set()).add(gene)
        fp.close()
                 
    if include_DisGeNET_curated:
        print 'integrating DisGeNET'
        with open('data/disease/DisGeNET_new.txt') as fp:
            for line in fp:
                values = line.rstrip().split('\t')
                gene = values[0]
                disease = values[1]
                disease_name = values[2]
                disease_name_map[disease] = disease_name
                score = float(values[-1])
                whole_disease_gene_dict.setdefault(disease, set()).add(gene)
                if score < DisGeNET_cutoff:
                    continue
                disease_gene_dict.setdefault(disease, set()).add(gene)
                       

    return disease_gene_dict, disease_name_map, whole_disease_gene_dict
   
def read_DISEASE_database_older(include_textming = True, confidence=2, include_DisGeNET_curated = True, more_evidence = False):
    #ensp_to_ensg_map = get_ENSP_ENSG_map()
    disease_gene_dict = {}
    whole_disease_gene_dict = {}
    disease_name_map = {}
    database_files = ['data/disease/human_disease_experiments_filtered.tsv', 'data/disease/human_disease_knowledge_filtered.tsv']
    if include_textming:
        database_files = database_files + ['data/disease/human_disease_textmining_filtered.tsv']
        
    for datafile in database_files:
        fp = open(datafile, 'r')
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            gene = values[0]
            disease = values[2]
            disease_name = values[3]
            disease_name_map[disease] = disease_name.upper()
            conf = float(values[-1])
            whole_disease_gene_dict.setdefault(disease, set()).add(gene)
            if conf < confidence:
                continue
            #if ensp_to_ensg_map.has_key(gene):
            disease_gene_dict.setdefault(disease, set()).add(gene)
            #else:
            #    print gene
        
        fp.close()
    
    if include_DisGeNET_curated:
        print 'integrating DisGeNET'
        with open('data/disease/DisGeNET_new.txt') as fp:
            for line in fp:
                values = line.rstrip().split('\t')
                gene = values[0]
                disease = values[1]
                disease_name = values[2]
                disease_name_map[disease] = disease_name
                score = float(values[-1])
                whole_disease_gene_dict.setdefault(disease, set()).add(gene)
                if score < DisGeNET_cutoff:
                    continue
                disease_gene_dict.setdefault(disease, set()).add(gene)
                
    if more_evidence:
        get_more_evidence_DisGeNET(whole_disease_gene_dict)         

       
    '''gene_counts = {}
    for dis_key in disease_gene_dict.keys():
        genes = disease_gene_dict[dis_key]
        gene_counts[dis_key] = len(genes)
    
    fig = plt.figure()  
    
    plt.bar(range(len(gene_counts)),gene_counts.values()) 
    plt.xlabel('Disease index')
    plt.ylabel('# of genes')
    plt.savefig('figure/dis_distribution.jpg')
    '''
    return disease_gene_dict, disease_name_map, whole_disease_gene_dict

def get_mRNA_lncRNA_expression_RNAseq_data(whole_data, disease_gene_list=[], ensg_ensp_map={}, gene_type_dict={}, mRNA = True, tissue=False):
    labels = []
    mrna_num = 0
    data = []
    lncRNA_list = []
    mRNA_list = []
    #http://vega.sanger.ac.uk/info/about/gene_and_transcript_types.html
    lncRNA_type_dicts = ['3prime_overlapping_ncrna', 'ambiguous_orf', 'antisense', 'antisense_RNA', 'lincRNA', 'ncrna_host', 'non_coding', 
                'non_stop_decay', 'processed_transcript', 'retained_intron', 'sense_intronic', 'sense_overlapping']
    curate_dict = add_more_lncRNAs()
    for key, val in whole_data.iteritems():
        if mRNA:
            ensp = ''
            if not tissue:
                if not gene_type_dict.has_key(key):
                    continue
    
                if gene_type_dict[key] == 'protein_coding':
                    mrna_num = mrna_num + 1
                    if ensg_ensp_map.has_key(key):
                        ensp = ensg_ensp_map[key]
                    else:
                        continue
                else:
                    continue
            else:
                ensp = key
            data.append(val)
            mRNA_list.append(key) 
            if ensp in disease_gene_list:
                labels.append(1)
            else:
                labels.append(0)   
            #except:
            #    pdb.set_trace()
        else:
            if key in curate_dict.values():
                data.append(val)
                lncRNA_list.append(key)        
                continue           
            if gene_type_dict.has_key(key):
                if gene_type_dict[key] == 'protein_coding':
                    continue
                #elif gene_type_dict[key] == 'lincRNA' or gene_type_dict[key] == 'antisense' or gene_type_dict[key] == 'pseudogene' or gene_type_dict[key] == 'sense_overlapping':
                #elif gene_type_dict[key] == 'lincRNA' or gene_type_dict[key] == 'antisense' or gene_type_dict[key] == 'sense_overlapping' or gene_type_dict[key] == 'processed_transcript':
                elif gene_type_dict[key] in lncRNA_type_dicts:
                #else:
                    data.append(val)
                    lncRNA_list.append(key)
                                    
            '''
            if not gene_type_dict.has_key(key):
                data.append(val)
                lncRNA_list.append(key)

            elif gene_type_dict[key] != 'protein_coding':
                data.append(val)
                lncRNA_list.append(key)
            '''        
    return np.array(data), np.array(labels), lncRNA_list, mRNA_list

def read_tissue_database():
    database_file = ['/home/panxy/eclipse/sponge/expression/human_tissue_experiments_filtered.tsv', 
                     '/home/panxy/eclipse/sponge/expression/human_tissue_knowledge_filtered.tsv']
    tissues = ['Adipose tissue', 'Colon' , 'Heart', 'Hypothalamus', 'Kidney', 'Liver', 'Lung', 'Ovary', 'Skeletal muscle', 'Spleen', 'Testis']
    
    tissue_express_dict = {}
    for datafile in database_file:
        fp = open(datafile, 'r')
        for line in fp:
            if 'RNA-seq' not in line:
                continue
            values = line.rstrip('\r\n').split('\t')
            gene = values[0]
            tissue = values[3]
            expval = values[5].split()[0]
            tissue_express_dict.setdefault(gene, []).append((tissue,expval))
            
        fp.close()
    
    final_express_dict = {}    
    for key in tissue_express_dict.keys():
        values = tissue_express_dict[key]
        tiss_vals = [0]*11
        for val in values:
            tissue = val[0]
            expval = float(val[1])
            index = tissues.index(tissue)
            tiss_vals[index] = expval
        final_express_dict[key] = tiss_vals
    #pdb.set_trace()
    return final_express_dict, tissues

def get_mRNA_lncRNA_expression_microarray_data(whole_data, disease_gene_list=[], ensg_ensp_map={}, mRNA = True):
    '''read microarray expression data'''
    labels = []
    mrna_num = 0
    data = []
    lncRNA_list = []
    #pdb.set_trace()
    for key, val in whole_data.iteritems():
        if mRNA:
            try:
                if val[0].startswith('A_'):
                    mrna_num = mrna_num + 1
                    if ensg_ensp_map.has_key(key):
                        ensp = ensg_ensp_map[key]
                    else:
                        #print key, val[0]
                        continue
                    data.append(val[2:])    
                    if ensp in disease_gene_list:
                        labels.append(1)
                    else:
                        labels.append(0)   
            except:
                pdb.set_trace()
        else:
            if val[0].startswith('CUST_'):
                data.append(val[2:]) 
                lncRNA_list.append(key)
                
    #print 'mRNA number', mrna_num, len(data)        
    return np.array(data), np.array(labels), lncRNA_list


       
        
def predict_for_lncRNA_using_mRNA(input_file, out_file, RNAseq = True,  data=0, ratio=5, confidence=2, use_mean = False, disease_acc = None, coexp=False, log2 = False, knn = False):
    disease_gene_dict, disease_name_map, whole_disease_gene_dict = read_DISEASE_database(confidence=confidence)
    ensg_ensp_map = get_ENSP_ENSG_map()
    #log2_flag  = log2
    if RNAseq:
        gene_type_dict,gene_name_ensg, gene_id_position = read_gencode_gene_type()
        if data == 0:
            whole_data, samples = read_human_RNAseq_expression(input_file, gene_name_ensg, log2 = log2) # for microarray expression data
        elif data ==1:
            whole_data, samples = read_evolutionary_expression_data(input_file, use_mean = use_mean, log2 = log2)
        elif data == 2:
            whole_data, samples = read_average_read_to_normalized_RPKM(input_file, use_mean = use_mean, log2 = log2)
        elif data == 3:
            whole_data, samples = read_gtex_expression(input_file, gene_type_dict)
        else:
            whole_data, samples = read_tissue_database()
        #pdb.set_trace()
    else:
        whole_data = read_normalized_series_file(input_file)   
        
    f_imp = open(out_file + '.imp', 'w')
    
    f_imp.write('\t'.join([' '] + samples))
    f_imp.write('\n')     
    fw_dis = open('overlap_num_disease', 'w')
    use_SVM = False   
    fw = open(out_file, 'w') 
    #ratio = 3
    if RNAseq:
        disease_lncRNA_data, lncRNAlabels,lncRNA_list, atmp  = get_mRNA_lncRNA_expression_RNAseq_data(whole_data, gene_type_dict=gene_type_dict, mRNA=False)
    else:
        disease_lncRNA_data, lncRNAlabels, lncRNA_list, atmp = get_mRNA_lncRNA_expression_microarray_data(whole_data, mRNA=False)
    fw.write('\t'.join([' '] + lncRNA_list))
    fw.write('\n') 
    all_other_disease_mRNA = set()
    for val in whole_disease_gene_dict.values():
        all_other_disease_mRNA = all_other_disease_mRNA | val
    
    all_pcc = []
    oboparser = obo_object()
    oboparser.read_obo_file()
    for key in disease_gene_dict:
        #disease_associated_data = []
        #labels = []
        if RNAseq:
            disease_mRNA_data, mRNAlabels, atmp, mRNA_list = get_mRNA_lncRNA_expression_RNAseq_data(whole_data, disease_gene_dict[key], ensg_ensp_map, gene_type_dict)    
        else:
            disease_mRNA_data, mRNAlabels, atmp, mRNA_list = get_mRNA_lncRNA_expression_microarray_data(whole_data, disease_gene_dict[key], ensg_ensp_map)
            
        #pdb.set_trace()    
        posi_num = np.count_nonzero(mRNAlabels)
        new_mRNA_list = [val for is_disease, val in zip(mRNAlabels, mRNA_list) if is_disease]
        
        #pare_disease = oboparser.getAncestors(key)
        child_disease = oboparser.getDescendents(key)
        #related_disease = pare_disease | child_disease  #- set(['DOID:4'])
        new_gene_set = get_disease_genes(whole_disease_gene_dict,child_disease)
        #new_gene_set = get_disease_genes(whole_disease_gene_dict, related_disease)
        
        other_mRNA_list=whole_disease_gene_dict[key]
        other_disease_mRNA = all_other_disease_mRNA - other_mRNA_list - new_gene_set
        #pdb.set_trace()
        #label_len = len(mRNAlabels)
        if posi_num < TRAIN_NUM:
            #print key, label_len
            continue
        
        
        disease = disease_name_map[key]
        if 5*posi_num > len(other_disease_mRNA):
            continue
        #if disease == 'GENETIC DISEASE':
        #    pdb.set_trace()
        #if disease.upper() == 'CANCER':
        #    continue
        fw_dis.write(key + '\t' + disease + '\t' + str(posi_num) + '\n')
        print disease, posi_num
        fw.write(key + '\t')
        f_imp.write(disease + '\t')
        if disease_acc is None:
            if coexp:
                #all_pcc = all_pcc + coexpression_hist_fig(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw)
                coexpression_based_prediction(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw, k = 1) 
            elif knn:
                coexpression_knn_based_prediction(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw, k = 15)
            else:
                cross_validataion_lncRNA_using_mRNA(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw, ensg_ensp_map, 
                    ratio = ratio, SVM = use_SVM, other_mRNA_list=whole_disease_gene_dict[key], gene_position_dict = gene_id_position, 
                    overlap_disease_mRNA_list = new_mRNA_list, negative_from_other_disease = other_disease_mRNA, f_imp = f_imp)
        else:
            cross_validataion_lncRNA_using_mRNA(disease_mRNA_data, mRNAlabels, disease_lncRNA_data, lncRNA_list, mRNA_list, fw, ensg_ensp_map, 
                                                ratio = ratio, SVM = use_SVM, weight = disease_acc[disease])
    #plot_hist_distance(all_pcc, 'PCC', 'gencode')
    fw.close()
    f_imp.close()
    fw_dis.close()
    return gene_id_position

def get_disease_genes(whole_disease_gene_dict, diseases):
    new_gene_set = set()
    for disease in diseases:
        if whole_disease_gene_dict.has_key(disease):
            new_gene_set = new_gene_set | whole_disease_gene_dict[disease]
    
    return new_gene_set
        

def construct_data_for_classifier_mRNA(input_file, out_file, RNAseq = True, data=0, ratio=5, confidence=2, use_mean = False, log2 = False):
    #pdb.set_trace()
    #print body2_map, confidence
    disease_gene_dict, disease_name_map, whole_disease_gene_dict = read_DISEASE_database(confidence=confidence)
    ensg_ensp_map = get_ENSP_ENSG_map()
    tissue =False
    if RNAseq:
        gene_type_dict,gene_name_ensg, gene_id_position = read_gencode_gene_type()
        if data == 0:
            whole_data, samples = read_human_RNAseq_expression(input_file, gene_name_ensg, log2 = log2) # for microarray expression data
        elif data ==1:
            whole_data, samples = read_evolutionary_expression_data(input_file, use_mean = use_mean, log2 = log2)
        elif data == 2:
            whole_data, samples = read_average_read_to_normalized_RPKM(input_file, use_mean = use_mean, log2 = log2)
        elif data == 3:
            whole_data, samples = read_gtex_expression(input_file, gene_type_dict)
        else:
            whole_data, samples = read_tissue_database()
            tissue = True   
    else:
        whole_data = read_normalized_series_file(input_file) # for RNAseq expression data
        
    use_SVM = False
    all_other_disease_mRNA = set()
    for val in whole_disease_gene_dict.values():
        all_other_disease_mRNA = all_other_disease_mRNA | val
    
    #f_imp = open(out_file + '.imp', 'w')
    #f_imp.write()
    #ratio = 3
    #pdb.set_trace()
    oboparser = obo_object()
    oboparser.read_obo_file()
    fw = open(out_file, 'w') 
    for key in disease_gene_dict:
        disease_associated_data = []
        labels = []
        if RNAseq:
            disease_associated_data, labels, atmp, mRNA_list = get_mRNA_lncRNA_expression_RNAseq_data(whole_data, disease_gene_dict[key], 
                                                                        ensg_ensp_map, gene_type_dict, tissue=tissue)
        else:
            disease_associated_data, labels, atmp, mRNA_list = get_mRNA_lncRNA_expression_microarray_data(whole_data, disease_gene_dict[key], 
                                                                                                          ensg_ensp_map)
        #pdb.set_trace()    
        posi_num = np.count_nonzero(labels)
        label_len = len(labels)
        #new_mRNA_list = [val for is_disease, val in zip(labels, mRNA_list) if is_disease]

        pdb.set_trace()
        '''pare_disease = oboparser.getAncestors(key)
        child_disease = oboparser.getDescendents(key)
        related_disease = pare_disease | child_disease - set(['DOID:4'])
        
        new_gene_set = get_disease_genes(whole_disease_gene_dict, related_disease)
        
        other_mRNA_list= whole_disease_gene_dict[key]
        other_disease_mRNA = all_other_disease_mRNA - other_mRNA_list - new_gene_set
        '''
        child_disease = oboparser.getDescendents(key)
        #related_disease = pare_disease | child_disease  #- set(['DOID:4'])
        new_gene_set = get_disease_genes(whole_disease_gene_dict, child_disease)
        #new_gene_set = get_disease_genes(whole_disease_gene_dict, related_disease)
        
        other_mRNA_list=whole_disease_gene_dict[key]
        other_disease_mRNA = all_other_disease_mRNA - other_mRNA_list - new_gene_set
        #pdb.set_trace()
        if posi_num < TRAIN_NUM:
            #print key, label_len
            continue
        
        if 5*posi_num > len(other_disease_mRNA):
            continue     
           
        print disease_name_map[key], posi_num
        fw.write(key + '\t')
        #f_imp.write(disease_name_map[key] + '\t')
        #cross_validataion_mRNA(disease_associated_data, labels, ratio=ratio, SVM = use_SVM) 
        #cross_validataion_multiple_mRNA(disease_associated_data, labels, mRNA_list, fw, ensg_ensp_map, ratio=ratio, 
        #                            other_mRNA_list=other_mRNA_list,negative_from_other_disease = other_disease_mRNA)
        #pdb.set_trace()
        validataion_multiple_mRNA(disease_associated_data, labels, mRNA_list, fw, ensg_ensp_map, ratio=ratio, 
                                 other_mRNA_list=other_mRNA_list,negative_from_other_disease = other_disease_mRNA)
    fw.close()
    #f_imp.close()

def plot_roc_curve(labels, probality, disease, result_file):
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 15.0
    figure = plt.figure() 
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    
    rects1 = plt.plot(fpr, tpr, label=' (AUC=%6.3f) ' %roc_auc)
    #rects1 = plt.plot(fpr, tpr, label=disease)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(disease)
    plt.legend(loc="lower right")
    
    plt.savefig('roc_figure/' + result_file.split('/')[-1] + '_' + disease + '.eps', format="eps") 
    plt.clf()
    plt.close('all')
    return roc_auc
       
#importance = [('fe2', 0.1), ('fe3', 0.4), ('fe1', 0.8)]
def plot_accuracy_bar(accuracy, disease_list, datafile):
    '''df = pd.DataFrame(importance, columns=['disease', 'accuracy'])
    #df['accuracy'] = df['accuracy'] / df['accuracy'].sum()
    
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='disease', y='accuracy', legend=False)
    plt.xlim(0, 1)
    plt.title('Accuracy for diseases')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.savefig('accuracy_fig/' + datafile.split('/')[-1] + '_imp.png')
    '''
    ind = np.arange(len(accuracy))
    '''df = pd.DataFrame(imp_list, columns=['tissue', 'score'])
    #df['accuracy'] = df['accuracy'] / df['accuracy'].sum()
    
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='tissue', y='score', legend=False)
    '''
    fig, ax = plt.subplots(figsize=(15,10))
    rects1 = plt.bar(ind, accuracy, 0.25, color='b')
    #plt.ylabel('Importance score')
    ax.set_xticks(ind)
    ax.set_xticklabels(disease_list, rotation=90, fontsize=10, style='normal', family='serif')
    plt.ylabel('Accuracy', fontsize=10)
    plt.tight_layout()
    #plt.xlim([0,5])
    #plt.xlabel('Tissue')
    plt.title('Accuracy for diseases')
    plt.savefig('accuracy_fig/' + datafile.split('/')[-1] + '_indep.eps', format="eps")
    plt.clf()
    plt.close()
    
def calculate_average_performance_mRNA_crossvlidation(result_file, plot_roc=True):
    fp = open(result_file, 'r')
    result = []
    label = []
    probability = []
    result_dis_acc = {}
    importance = []
    disease_list = []
    tmp_resu = []
    disease=''
    for line in fp:
        values = line.rstrip('\r\n').split('\t')
        if 'ROC_label' in line:
            label = [int(val) for val in values[1:]]
        elif 'ROC_probability' in line:
            probability = [float(val) for val in values[1:]] 
            if plot_roc: 
                auc = plot_roc_curve(label, probability, disease, result_file)  
                tmp_resu = tmp_resu + [auc]
                result.append(tmp_resu)
        else:
            tmp_resu = []
            disease = values[0]        
            tmp_resu = [float(val) for val in values[1:]]    
            #result.append(tmp_resu)
            result_dis_acc[disease] = tmp_resu[0]
            importance.append(tmp_resu[0])
            disease_list.append(disease)
    fp.close()
    print 'average performance for all diseases:'
    print np.mean(result, axis=0)
    
    plot_accuracy_bar(importance, disease_list, result_file)
    return result_dis_acc

def plot_bar_imp(imp_list, disease, imp_file, tissues, ylabel = 'Importance score'):
    ind = np.arange(len(imp_list))
    '''df = pd.DataFrame(imp_list, columns=['tissue', 'score'])
    #df['accuracy'] = df['accuracy'] / df['accuracy'].sum()
    
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='tissue', y='score', legend=False)
    '''
    fig, ax = plt.subplots(figsize=(12,8))
    rects1 = plt.bar(ind, imp_list, 0.25, color='b')
    #plt.ylabel('Importance score')
    ax.set_xticks(ind)
    ax.set_xticklabels(tissues, rotation=90, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(disease, fontsize=15)
    plt.tight_layout()
    #plt.xlim([0,5])
    #plt.xlabel('Tissue')
    #if disease == 'NON-SMALL CELL LUNG CARCINOMA' or disease == 'KIDNEY DISEASE':
    #    plt.show()
    plt.savefig('imp_fig/' + disease + imp_file.split('/')[-1] + '.eps', format='eps')
    plt.clf()
    plt.close()
    
def plot_tissue_importance(result_imp_file):
    print 'ploting tissue iportance'
    fp = open(result_imp_file, 'r')
    disease=''
    index = 0
    for line in fp:
        values = line.rstrip('\r\n').split('\t')
        if index == 0:
            tissues = values[1:] 
            index = index  + 1
        else:
            disease = values[0]        
            #imp_list = [(val1, float(val)) for val1, val in zip(tissues,values[1:])]
            imp_list = [float(val) for val in values[1:-1]]
            plot_bar_imp(imp_list, disease, result_imp_file, tissues)
    fp.close()
    #print 'average performance for all diseases:'
    #print np.mean(result, axis=0)
    #return result_dis_acc    
def get_disease_associated_matrix(disease_tissue_file = 'data/DiseaseTissueAssociationMatrixLageEtAl2008PNAS.tbl.txt'):
    disease_tissue_dct  = {}
    with open(disease_tissue_file, 'r') as fp:
        index = 0
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            if index == 0:
                tissues = values[1:]
                index = index + 1
            else:
                disease  = values[0]
                scores = [float(val) for val in values[1:]]
                disease_tissue_dct[disease] = scores
    
    return disease_tissue_dct, tissues

def plot_disease_tissue_score(disease, disease_tissue_dct, tissues, disease_name):
    #pdb.set_trace()
    score_list = disease_tissue_dct[disease] 
    plot_bar_imp(score_list, disease_name, disease, tissues, ylabel = 'Association score')

def benchmark_predict_disease_lncRNAs(predict_lncRNA_result_file, fw, source, expanison = False, discrete =False):
    #ncRNA_alias_identifier_map = stringrnautils.read_ncRNA_alias()
    gene_type_dict,gene_name_ensg, gene_id_position = read_gencode_gene_type()
    fp = open(predict_lncRNA_result_file, 'r')
    gene_names = []
    #gene_sign = []
    #pdb.set_trace()
    head = True
    rna_ids = []
    protein_ids = []
    scores = []
    organisms = []
    
    for line in fp:
        if head:
            gene_names = line.rstrip('\r\n').split('\t')[1:]
            head = False
            '''for val in genes[1:]:
                if ncRNA_alias_identifier_map.has_key(val.upper()):
                    gene_names.append(ncRNA_alias_identifier_map[val.upper()])
                    gene_sign.append(1)
                else:
                    gene_names.append(val.upper())
                    gene_sign.append(0)
            '''
            continue
        
        values = line.rstrip('\r\n').split('\t')
        disease = values[0].upper()
        score_vals = values[1:]
        for index in range(len(score_vals)):
            #if gene_sign[index] == 0:
            #    continue
            rna_ids.append(gene_names[index])
            protein_ids.append(disease)
            scores.append(float(score_vals[index]))
            organisms.append('9606')  
      
    fp.close() 
    fig_name = predict_lncRNA_result_file.split('/')[-1]
    new_scores  = stringrnautils.benchmark(organisms, rna_ids, protein_ids, scores, increases=True, discrete = discrete, window_size= WINDOW_SIZE, fit_name=fig_name, expanison = expanison)
    for organism_id, rna_id, protein_id, new_score in zip(organisms, rna_ids, protein_ids, new_scores):
        fw.write('%s\n'  % '\t'.join((organism_id, rna_id, protein_id, '0', 'prediction',
                                str(new_score), source, '', '')))
    

def calculate_vector_length(data, data_source, norm = 2):
    distance_list = []
    for val in data:
        distance = np.linalg.norm(val, ord=norm)
        distance_list.append(distance)
    
    plot_hist_distance(distance_list, 'vector_length', data_source, norm = norm)     
    #return distance_list

def plot_hist_distance(data_list, distance_type, data_source, genetype = 'mRNA', norm = 2):
    print 'plot histogram fig'
    fig, ax = plt.subplots()
    #ax = fig.add_subplot(111)
    #weights = np.ones_like(data_list)/len(data_list)
    plt.hist(data_list, 50000)
    ax.set_xlabel(distance_type)
    #ax.set_xlim([0, 2000])
    ax.set_xlim([0, 1])
    plt.show()
    #plt.savefig('distance_fig/' + data_source + '_' + distance_type + genetype + '.eps', type="eps")


def calculate_euclidean_distance(data, data_source):
    print 'calculating Euclidean distance'
    rows, cols = data.shape
    print rows, cols
    #data_pcc = np.zeros(rows, cols)
    euclidean_list  = []
    data = normalize(data, axis=0)
    '''for i in xrange(rows): # rows are the number of rows in the matrix. 
        for j in xrange(i, rows):
            if i == j:
                continue
            rval = euclidean_distance(datam[i,:], datam[j,:])[0]
            abs_rval = np.absolute(rval)
            euclidean_list.append(abs_rval)
    '''
    S = pairwise_distances(data, metric="euclidean")
    S = np.hstack(S)
    #pdb.set_trace()        
    plot_hist_distance(S, 'Euclidean', data_source)        
    return euclidean_list

def calculate_mRNA_lncRNA_pcc(mRNA_data, lncRNA_data, data_source):
    print 'calculating PCC distance'
    rows, cols = mRNA_data.shape
    #data_pcc = np.zeros(rows, cols)
    pcc_list  = []
    #data = normalize(data, axis=0)
    scaler = StandardScaler()
        #scaler = MinMaxScaler()
    scaler.fit(mRNA_data)
    data = scaler.transform(mRNA_data)
    lncRNA_data = scaler.transform(lncRNA_data)
    
    for vali in mRNA_data: # rows are the number of rows in the matrix. 
        pcc_list = pcc_list + [stats.pearsonr(vali,valj)[0] for valj in lncRNA_data]
        
        '''for j in xrange(i, rows):
            if i == j:
                continue
            rval = stats.pearsonr(data[i,:], data[j,:])[0]
            abs_rval = np.absolute(rval)
            pcc_list.append(abs_rval)
        '''
    #pdb.set_trace()
    #print len(pcc_list)
    plot_hist_distance(pcc_list, 'PCC', data_source, 'lncRNA')        
    return pcc_list

def plot_distance_fig(data, data_source):
    #fea_len = len(data[0])
    #print fea_len
    #get_normalized_values_by_column(data, fea_len)
    calculate_pcc(data, data_source)
    #calculate_euclidean_distance(data, data_source)
    #calculate_vector_length(data, data_source, norm = 1)

def get_lincRNA_mRNA(gencode_genes, gene_type_dict):
    genecode_mRNA = set()
    genecode_lincRNA = set()
    lncRNA_type_dicts = ['3prime_overlapping_ncrna', 'ambiguous_orf', 'antisense', 'antisense_RNA', 'lincRNA', 'ncrna_host', 'non_coding', 
            'non_stop_decay', 'processed_transcript', 'retained_intron', 'sense_intronic', 'sense_overlapping']
    for key in gencode_genes:
        if 'mir' in key.lower():
            continue
        if gene_type_dict.has_key(key):
            if gene_type_dict[key] == 'protein_coding':
                genecode_mRNA.add(key)         
            #elif gene_type_dict[key] == 'lincRNA' or gene_type_dict[key] == 'antisense' or gene_type_dict[key] == 'pseudogene' or gene_type_dict[key] == 'sense_overlapping':
            #elif gene_type_dict[key] == 'lincRNA' or gene_type_dict[key] == 'antisense' or gene_type_dict[key] == 'sense_overlapping'  or gene_type_dict[key] == 'processed_transcript':
            elif gene_type_dict[key] in lncRNA_type_dicts:
                genecode_lincRNA.add(key)
        else:
            continue
        
        #genecode_lincRNA.add(key)
             
    return genecode_mRNA, genecode_lincRNA

def get_expression_based_gene(data, gene_list):
    extracted_data = []
    for gene in gene_list:
        extracted_data.append(data[gene])
        
    return np.array(extracted_data)
        

def get_mRNA_expression():
    file_names = ['data/gencodev7/gene.matrix.csv', 'data/GSE43520/Human_Normalized_RPKM_NonSS_LncRNA_ProteinCoding_MainDataset_NonStrandSpecific.txt', 
                  'data/GSE30352/Human_Ensembl57_TopHat_UniqueReads.txt'] 
    gene_type_dict,gene_name_ensg, gene_id_position = read_gencode_gene_type()
    gencode_data, samples = read_human_RNAseq_expression(file_names[0], gene_name_ensg) # for microarray expression data
    gse43520_data, samples = read_evolutionary_expression_data(file_names[1])
    gse30352_data, samples = read_average_read_to_normalized_RPKM(file_names[2])
    
    gencode_genes = gencode_data.keys()
    gse43520_genes = gse43520_data.keys()         
    gse30352_genes = gse30352_data.keys()
    #gene_type_dict = read_gencode_gene_type()
    #ncRNA_alias_identifier_map = stringrnautils.read_ncRNA_alias()
    #ensg_ensp_map = get_ENSP_ENSG_map()
    
    #genecode_mRNA, genecode_lincRNA = get_lincRNA_mRNA(gencode_genes, gene_type_dict, ncRNA_alias_identifier_map, ensg_ensp_map)
    genecode_mRNA, genecode_lincRNA = get_lincRNA_mRNA(gencode_genes, gene_type_dict)
    gse43520_mRNA, gse43520_lincRNA = get_lincRNA_mRNA(gse43520_genes, gene_type_dict)
    gse30352_mRNA, gse30352_lincRNA = get_lincRNA_mRNA(gse30352_genes, gene_type_dict)
    
    genecode_mRNA_expression = get_expression_based_gene(gencode_data, genecode_mRNA)
    gse43520_mRNA_expression = get_expression_based_gene(gse43520_data, gse43520_mRNA)
    gse30352_mRNA_expression = get_expression_based_gene(gse30352_data, gse30352_mRNA)

    genecode_lincRNA_expression = get_expression_based_gene(gencode_data, genecode_lincRNA)
    gse43520_lincRNA_expression = get_expression_based_gene(gse43520_data, gse43520_lincRNA)
    gse30352_lincRNA_expression = get_expression_based_gene(gse30352_data, gse30352_lincRNA)
      
    print 'print mRNA distance hist'
    plot_distance_fig(genecode_mRNA_expression, 'E-MTAB-513')
    plot_distance_fig(gse43520_mRNA_expression, 'GSE43520')
    plot_distance_fig(gse30352_mRNA_expression, 'GSE30352')
    '''
    print 'print mRNA and lncRNA hist'
    calculate_mRNA_lncRNA_pcc(genecode_mRNA_expression, genecode_lincRNA_expression, 'E-MTAB-513')
    calculate_mRNA_lncRNA_pcc(gse43520_mRNA_expression, gse43520_lincRNA_expression, 'GSE43520')
    calculate_mRNA_lncRNA_pcc(gse30352_mRNA_expression, gse30352_lincRNA_expression, 'GSE30352')    
    '''
    
    
def get_genes_overlap():
    file_names = ['data/gencodev7/gene.matrix.csv', 'data/GSE43520/Human_Normalized_RPKM_NonSS_LncRNA_ProteinCoding_MainDataset_NonStrandSpecific.txt', 
                  'data/GSE30352/Human_Ensembl57_TopHat_UniqueReads.txt', 'data/gtex/expression-SMTSD.tsv.gz'] 
    gene_type_dict,gene_name_ensg, gene_id_position = read_gencode_gene_type()
    gencode_data, samples = read_human_RNAseq_expression(file_names[0], gene_name_ensg, log2 = True) # for microarray expression data
    gse43520_data, samples = read_evolutionary_expression_data(file_names[1], log2 = True)
    gse30352_data, samples = read_average_read_to_normalized_RPKM(file_names[2], log2 = True)
    gtex_data, samples = read_gtex_expression(file_names[3], gene_type_dict)
    
    gencode_genes = gencode_data.keys()
    gse43520_genes = gse43520_data.keys()         
    gse30352_genes = gse30352_data.keys()
    gtex_genes = gtex_data.keys()
    #gene_type_dict = read_GRCh37_gene_type()
    #ncRNA_alias_identifier_map = stringrnautils.read_ncRNA_alias()
    #ensg_ensp_map = get_ENSP_ENSG_map()
    
    #genecode_mRNA, genecode_lincRNA = get_lincRNA_mRNA(gencode_genes, gene_type_dict, ncRNA_alias_identifier_map, ensg_ensp_map)
    genecode_mRNA, genecode_lincRNA = get_lincRNA_mRNA(gencode_genes, gene_type_dict)
    gse43520_mRNA, gse43520_lincRNA = get_lincRNA_mRNA(gse43520_genes, gene_type_dict)
    gse30352_mRNA, gse30352_lincRNA = get_lincRNA_mRNA(gse30352_genes, gene_type_dict)            
    gtex_mRNA, gtex_lincRNA = get_lincRNA_mRNA(gtex_genes, gene_type_dict) 
        
        
    return genecode_mRNA, genecode_lincRNA, gse43520_mRNA, gse43520_lincRNA,gse30352_mRNA, gse30352_lincRNA, gtex_mRNA, gtex_lincRNA    

def get_overlap_set(gencode, gse43520, gse30352, gtex):
    data = {}
    '''
    data[('E-MTAB-513', 'GSE43520')] = len(gencode & gse43520)
    data[('E-MTAB-513', 'GSE30352')] = len(gencode & gse30352)
    data[('E-MTAB-513', 'GTEx')] = len(gencode & gtex)
    data[('GSE43520', 'GSE30352')] = len(gse30352 & gse43520)
    data[('GSE43520', 'GTEx')] = len(gtex & gse43520)
    data[('GSE30352', 'GTEx')] = len(gtex & gse30352)
    data[('E-MTAB-513', 'GSE43520', 'GSE30352')] = len(gencode & gse43520 & gse30352)
    data[('E-MTAB-513', 'GSE43520', 'GTEx')] = len(gencode & gse43520 & gtex)
    data[('GSE43520', 'GSE30352', 'GTEx')] = len(gse30352 & gse43520 & gtex)
    data[('E-MTAB-513', 'GSE43520', 'GSE30352', 'GTEx')]= len(gencode & gse30352 & gse43520 & gtex)
    data[('E-MTAB-513',)] = len(gencode - gse30352 - gse43520 - gtex)
    data[('GSE43520',)] = len(gse43520 - gse30352 - gencode - gtex)
    data[('GSE30352',)] = len(gse30352 - gencode - gse43520 - gtex)
    data[('GTEx',)] = len(gtex - gse30352 - gse43520 - gencode)
    '''
    data[('D1', 'D2')] = len(gencode & gse43520)
    data[('D1', 'D3')] = len(gencode & gse30352)
    data[('D1', 'D4')] = len(gencode & gtex)
    data[('D2', 'D3')] = len(gse30352 & gse43520)
    data[('D2', 'D4')] = len(gtex & gse43520)
    data[('D3', 'D4')] = len(gtex & gse30352)
    data[('D1', 'D2', 'D3')] = len(gencode & gse43520 & gse30352)
    data[('D1', 'D2', 'D4')] = len(gencode & gse43520 & gtex)
    data[('D2', 'D3', 'D4')] = len(gse30352 & gse43520 & gtex)
    data[('D1', 'D2', 'D3', 'D4')]= len(gencode & gse30352 & gse43520 & gtex)
    data[('D1',)] = len(gencode - gse30352 - gse43520 - gtex)
    data[('D2',)] = len(gse43520 - gse30352 - gencode - gtex)
    data[('D3',)] = len(gse30352 - gencode - gse43520 - gtex)
    data[('D4',)] = len(gtex - gse30352 - gse43520 - gencode)
    
    
    return data

def plot_bar_overlap_fig():
    
    from random import random
    genecode_mRNA, genecode_lincRNA, gse43520_mRNA, gse43520_lincRNA,gse30352_mRNA, gse30352_lincRNA, gtex_mRNA, gtex_lincRNA = get_genes_overlap()
    cats1 = ['E-MTAB-513', 'GSE43520', 'GSE30352', 'GTEx']
    cats = ['D1', 'D2', 'D3', 'D4']
    #data = {('cat1',): 523, ('cat2',): 231, ('cat3',): 102, ('cat4',): 72, ('cat1','cat2'): 710,('cat1','cat3'): 891,('cat1','cat3','cat4') : 621}

    data = get_overlap_set(genecode_mRNA, gse43520_mRNA, gse30352_mRNA, gtex_mRNA)
    #data = get_overlap_set(genecode_lincRNA, gse43520_lincRNA, gse30352_lincRNA, gtex_lincRNA)
    colors = dict([(k,(random(),random(),random())) for k in data.keys()])
    print colors
    for i, cat in enumerate(sorted(cats)):
        y = 0
        for key, val in data.items():
            if cat in key:
                plt.bar(i, val, bottom=y, color=colors[key])
                plt.text(i,y,' '.join(key))
                y += val
    plt.xticks(np.arange(len(cats))+0.4, cats1 )
    #plt.legend( [val2 +': '  + val1 for val1, val2 in zip(cats1, cats)], loc='upper center', bbox_to_anchor=(0.8, 1.1))
    plt.title('Protein coding gene')
    plt.show()
    plt.clf()
   
def plot_venn_figure():
    print 'plotting venn fig'
    genecode_mRNA, genecode_lincRNA, gse43520_mRNA, gse43520_lincRNA,gse30352_mRNA, gse30352_lincRNA, gtex_mRNA, gtex_lincRNA = get_genes_overlap()
    #fig = plt.figure() 
    #ax1=fig.add_subplot(1,2,1)
    #ax2=fig.add_subplot(1,1,1)
    set_labels = ('E-MTAB-513', 'GSE43520', 'GSE30352', 'GTEx')
    #pdb.set_trace()
    v1 = venn.venn([genecode_mRNA, gse43520_mRNA, gse30352_mRNA, gtex_mRNA], set_labels, title_name = 'PCG', figsize=(7,7))
    '''ax2.set_title("protein coding", fontsize=20)
    v1.get_patch_by_id('100').set_color('blue')
    v1.get_patch_by_id('010').set_color('green')
    v1.get_patch_by_id('001').set_color('gray')
    v1.get_patch_by_id('001').set_color('red')
    '''
    #for text in v1.set_labels:
    #    text.set_fontsize(20)
    v2 = venn.venn([genecode_lincRNA, gse43520_lincRNA, gse30352_lincRNA, gtex_lincRNA], set_labels, title_name = 'lncRNA', figsize=(7,7))
    #v2.get_patch_by_id('100').set_color('blue')
    #v2.get_patch_by_id('010').set_color('green')
    #v2.get_patch_by_id('001').set_color('gray')
    #v2.get_patch_by_id('001').set_color('red')
    #for text in v2.set_labels:
    #    text.set_fontsize(20)
    
    #plt.title("lncRNA", fontsize=20)
    
    #plt.tight_layout()
    #plt.show() 

def plot_figure(data, tissue_names, gene, dataname):
    width = 0.5
    fig, ax = plt.subplots()
    len_tiss = len(tissue_names)
    rects1 = ax.bar(range(len_tiss), data, width, color='r')
    ind = np.arange(len_tiss) 
    # add some
    ax.set_ylabel('expression value', fontsize=15, rotation=90)
    if gene == 'ENSP00000321106':
        gene = 'TAC1'
    #if gene == 'ENSP00000215637':
    #    gene = 'MADCAM1'
    #pl.xticks(np.arange(len(labels)), labels)
    plt.title(gene +' in ' + dataname)
    
    ax.set_xticks(ind)
    ax.set_xticklabels(tissue_names, fontsize=15, rotation=60)
    #ax.set_xlabel('')
    #ax.legend( rects1[0], 'features score' )
    plt.tight_layout()
    
    if gene == 'TAC1':
        plt.show()
        pdb.set_trace()
    #if gene == 'MADCAM1':
    #    plt.show()
    #    pdb.set_trace()
    plt.savefig('tissue_fig/' + dataname + '_' + gene + '.eps', format="eps")
    plt.clf()
    plt.close()      

def plot_expression_fig():
    #tissue_names =["adipose","adrenal","blood","brain","breast","colon","heart","kidney","liver","lung","lymph","ovary","prostate","skeletal_muscle","testes","thyroid"]
    tissue_names = ["thyroid",    "testis"    ,"ovary"    ,"leukocyte"    ,"skeletal-muscle"    ,"prostate"    ,"lymph-node"    ,"lung"    ,"adipose"    ,"adrenal"    ,"brain"    ,"breast"   , "colon",    "kidney"    ,"heart"    ,"liver"]
    #TAC1:ENSG00000006128
    TAC1 = np.array([0.385593,    0.0632795,    0.00333995,    0,    0,    0.164683,    0,    0.0248445,    0.0231999,    0,    0.608698,    0.0699199,    0.65125,    0,    0.135134,    0])
    #TAC1 = np.array([0.0864357,0.0913751,0,6.70834,0.546626,8.19027,0.789331,0.354887,0.0585287,0.1864,0.666308,0.106234,1.23798,0,0.876098,3.97795])
    #TAC1 = TAC1/sum(TAC1)
    #MADCAM1:ENSG00000099866
    MADCAM1 = np.array([0.0609162,    0.0862219,    0.0243802,    0.0276235,    0.0172159,    0.0265382,    0.0591759,    0.0303729,    0.0242002,    0.0612292,    0.175013,    0.068069,    0.109662,    0.0343804,    0.0270036,    0.0279304])
    #MADCAM1 =np.array([0.133161,0.293953,0.247603,0.699063,0.186646,1.18634,0.0152818,0.0594867,0.100087,0.212992,0.481881,0.0660167,0.278259,0,0.364933,0.365753])
    #MADCAM1 = MADCAM1/sum(MADCAM1)
    width = 0.5
    fig, ax = plt.subplots()
    len_tiss = len(tissue_names)
    rects1 = ax.bar(range(len_tiss), MADCAM1, width, color='r')
    ind = np.arange(len_tiss) 
    # add some
    #ax.set_ylabel('Tissue specificity', fontsize=15, rotation=90)
    ax.set_ylabel('Expression value', fontsize=15, rotation=90)
    plt.title('MADCAM1' +' in ' + 'E-MTAB-513')
    #plt.ylim([0, 0.351])
    ax.set_xticks(ind)
    ax.set_xticklabels(tissue_names, fontsize=15, rotation=60)
    plt.tight_layout()
    plt.show()

def plot_gene_express_in_tissue():
    file_names = ['data/gencodev7/gene.matrix.csv', 'data/GSE43520/Human_Normalized_RPKM_NonSS_LncRNA_ProteinCoding_MainDataset_NonStrandSpecific.txt', 
                  'data/GSE30352/Human_Ensembl57_TopHat_UniqueReads.txt'] 
    disease_gene_dict, disease_name_map, whole_disease_gene_dict = read_DISEASE_database(confidence=2)
    ensp_ensg_map = get_ensg_ensp_map()
    gene_type_dict,gene_name_ensg, gene_id_position = read_gencode_gene_type()
    gencode_data, gencode_samples = read_human_RNAseq_expression(file_names[0], gene_name_ensg) # for microarray expression data
    gse43520_data, gse43520_samples = read_evolutionary_expression_data(file_names[1])
    gse30352_data, gse30352_samples = read_average_read_to_normalized_RPKM(file_names[2])
    
    
    disease_gene_list = disease_gene_dict['DOID:0050589'] #'Inflammatory bowel disease'
    #pdb.set_trace()
    for key in disease_gene_list:
        if ensp_ensg_map.has_key(key):
            ensg = ensp_ensg_map[key]
        else:
            continue
        
        if gencode_data.has_key(ensg):
            gencode_exp = gencode_data[ensg]
            plot_figure(gencode_exp, gencode_samples, key, 'E-MTAB-513')
                    
        if gse30352_data.has_key(ensg):
            gse30352_exp = gse30352_data[ensg]
            plot_figure(gse30352_exp, gse30352_samples, key, 'GSE30352')
        

    
        if gse43520_data.has_key(ensg):
            gse43520_exp = gse43520_data[ensg]
            plot_figure(gse43520_exp, gse43520_samples, key, 'GSE43520')

def add_more_lncRNAs():
    ncRNA_alias_identifier_map = {}
    ncRNA_alias_identifier_map['LSINCT5'] = 'ENSG00000281560'
    ncRNA_alias_identifier_map['DLG2AS'] = 'ENSG00000274006'
    ncRNA_alias_identifier_map['BX118339'] = 'ENSG00000280989'
    ncRNA_alias_identifier_map['HELLPAR'] = 'ENSG00000281344'
    ncRNA_alias_identifier_map['KUCG1'] = 'ENSG00000280752'
    ncRNA_alias_identifier_map['MIR100HG'] = 'ENSG00000255090'    
    ncRNA_alias_identifier_map['MKRN3-AS1'] = 'ENSG00000260978'  
    ncRNA_alias_identifier_map['NPTN-IT1'] = 'ENSG00000281183'     
    ncRNA_alias_identifier_map['PANDAR'] = 'ENSG00000281450'
    ncRNA_alias_identifier_map['SNHG4'] = 'ENSG00000281398'    
    ncRNA_alias_identifier_map['SPRY4-IT1'] = 'ENSG00000281881'                  
    ncRNA_alias_identifier_map['YIYA'] = 'ENSG00000281664'
    ncRNA_alias_identifier_map['BLACAT1'] = 'ENSG00000281406' 
    ncRNA_alias_identifier_map['7SL'] = 'ENSG00000276168'
    ncRNA_alias_identifier_map['ANRIL'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['NCRAN'] = 'ENSG00000163597'
    ncRNA_alias_identifier_map['AIR'] = 'ENSG00000268257'
    ncRNA_alias_identifier_map['51A'] = 'ENSG00000268516'
    ncRNA_alias_identifier_map['ASFMR1'] = 'ENSG00000268066'
    ncRNA_alias_identifier_map['CDKN2B-AS10'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['CDKN2B-AS11'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['CDKN2B-AS12'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['CDKN2B-AS13'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['CDKN2B-AS2'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['CDKN2B-AS3'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['CDKN2B-AS4'] = 'ENSG00000240498' 
    ncRNA_alias_identifier_map['CDKN2B-AS5'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['CDKN2B-AS6'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['CDKN2B-AS7'] = 'ENSG00000240498'  
    ncRNA_alias_identifier_map['CDKN2B-AS8'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['CDKN2B-AS9'] = 'ENSG00000240498'
    ncRNA_alias_identifier_map['DBE-T'] = 'ENSG00000277162'
    ncRNA_alias_identifier_map['FMR4'] = 'ENSG00000268066'
    ncRNA_alias_identifier_map['GDNFOS'] = 'ENSG00000248587'
    ncRNA_alias_identifier_map['HTTAS'] = 'ENSG00000278778'
    ncRNA_alias_identifier_map['PCAT-1'] = 'ENSG00000253438'
    ncRNA_alias_identifier_map['PSORS1C3'] = 'ENSG00000204528'
    ncRNA_alias_identifier_map['PTENPG1'] ='ENSG00000237984'
    ncRNA_alias_identifier_map['SCAANT1'] = 'ENSG00000280620'
    ncRNA_alias_identifier_map['ANTI-NOS2A'] = 'ENSG00000266647'
    ncRNA_alias_identifier_map['LNCRNA-LET'] = 'ENSG00000281183'
    ncRNA_alias_identifier_map['PTCSC'] = 'ENSG00000204528'
    ncRNA_alias_identifier_map['CCAT2'] = 'ENSG00000280997'
    ncRNA_alias_identifier_map['CTBP1-AS'] = 'ENSG00000280927'
    ncRNA_alias_identifier_map['ENST00000513542'] = 'ENSG00000250902'
    ncRNA_alias_identifier_map['UCH1LAS'] = 'ENSG00000251173'
    ncRNA_alias_identifier_map['THRIL'] = 'ENSG00000280634'
    ncRNA_alias_identifier_map['SCHLAP1'] = 'ENSG00000281131'
    ncRNA_alias_identifier_map['NEAT-1'] ='ENSG00000245532'
    ncRNA_alias_identifier_map['GHET1'] ='ENSG00000281189'
    ncRNA_alias_identifier_map['HLA-AS1'] = 'ENSG00000257551'
    
    return ncRNA_alias_identifier_map

def transfer_name_ensg_for_lncRNAdisease():
    
    ncRNA_alias_identifier_map = {}
    fp1 = gzip.open('/home/panxy/eclipse/string-rna/id_dictionaries/ncRNAorthfile.tsv.gz')  
    for line in fp1:
        values = line.rstrip('\r\n').split('\t')
        ncRNA_alias_identifier_map[values[1].upper()] = values[0]
    fp1.close()  
    
    new_dict = add_more_lncRNAs()
    ncRNA_alias_identifier_map.update(new_dict)
    
    gene_type_dict,gene_name_ensg, gene_id_position = read_gencode_gene_type()
    
    #disease_gene_dict, disease_name_map = read_DISEASE_database(confidence=confidence)
    #disease_list = [val.upper() for val in disease_name_map.values()]
    fw = open('data/disease/data_disease_new.txt', 'w')
    organism_id = '9606'
    for line in open('data/disease/data_disease.txt'):
        values = line.rstrip('\r\n').split('\t')
        
        disease = values[2].upper()
        #if disease.upper() not in disease_list:
        #    print disease
        
        gene = values[1].upper()
        if 'ENSG' in gene:
            tmp_name = gene.split('.')[0]
            #fw.write(line[:-1] + '\t' + tmp_name +'\n')
            fw.write('%s\n'  % '\t'.join((organism_id, tmp_name, disease, '0', 'database',
                        '0.900', 'LncRNADisease', '', '')))
            continue
        find = True
        if gene_name_ensg.has_key(gene):
            #fw.write(line[:-1] + '\t' + gene_name_ensg[gene] +'\n')
            fw.write('%s\n'  % '\t'.join((organism_id, gene_name_ensg[gene], disease, '0', 'database',
                    '0.900', 'LncRNADisease', '', '')))
        elif ncRNA_alias_identifier_map.has_key(gene):
            fw.write('%s\n'  % '\t'.join((organism_id, ncRNA_alias_identifier_map[gene], disease, '0', 'database',
                    '0.900', 'LncRNADisease', '', '')))
            #fw.write(line[:-1] + '\t' + ncRNA_alias_identifier_map[gene] +'\n')
        else:
            gene_alias = values[10].split(';')
            for val_name in gene_alias:
                if gene_name_ensg.has_key(val_name):
                    #fw.write(line[:-1] + '\t' + gene_name_ensg[val_name] +'\n')
                    fw.write('%s\n'  % '\t'.join((organism_id, gene_name_ensg[val_name], disease, '0', 'database',
                            '0.900', 'LncRNADisease', '', '')))
                    break
                elif ncRNA_alias_identifier_map.has_key(val_name):
                    #fw.write(line[:-1] + '\t' + ncRNA_alias_identifier_map[val_name] +'\n')
                    fw.write('%s\n'  % '\t'.join((organism_id, ncRNA_alias_identifier_map[val_name], disease, '0', 'database',
                            '0.900', 'LncRNADisease', '', '')))
                    break
                else:
                    find =False
        if not find:    
            print gene

    
    fw.close()

def get_disease_in_database():
    lncRNAdisease_set = set()
    for line in open('data/disease/data_disease_new.txt'):
        values = line.rstrip('\r\n').split('\t')
        lncRNAdisease_set.add(values[2].upper())

    disease_gene_dict, disease_name_map, something = read_DISEASE_database(confidence=confidence)
    DISEASE_list = [val.upper() for val in disease_name_map.values()]    
    DISEASE_set = set(DISEASE_list)
    return lncRNAdisease_set, DISEASE_set, disease_gene_dict

def compare_disease_overlap():
    lncRNAdisease_set, DISEASE_set, disease_gene_dict = get_disease_in_database()
    #for dis in lncRNAdisease_set:
    #    if dis not in DISEASE_set:
    #        print dis
    
    fig = plt.figure() 
    #ax1=fig.add_subplot(1,1,1)
    #ax2=fig.add_subplot(1,2,2)
    #set_labels = ('LncRNADisease', 'DISEASES')
    
    #venn2([lncRNAdisease_set, DISEASE_set], set_labels = set_labels, ax=ax1)
    #plt.title('Disease overlap')
    
    gene_counts = {}
    for dis_key in disease_gene_dict.keys():
        genes = disease_gene_dict[dis_key]
        gene_counts[dis_key] = len(genes)
    
    ax2=fig.add_subplot(1,1,1)
    
    plt.bar(range(len(gene_counts)),gene_counts.values()) 
    plt.xlabel('Disease index')
    plt.ylabel('# of genes')
    plt.title('Distribution of genes in disease')
    
    plt.show()     
    
    
def read_master_files(master_file):
    disease_lncRNA_dict = {}
    master_disease_set = set()
    master_lncRNA_set = set()
    
    print 'read file %s' %master_file 
    fp = open(master_file, 'r')
    head = True
    for line in fp:
        if head:
            gene_names = line.rstrip('\r\n').split('\t')[1:]
            master_lncRNA_set = set(gene_names)
            head = False
            continue
    
        values = line.rstrip('\r\n').split('\t')
        disease = values[0].upper()
        master_disease_set.add(disease)
        score_vals = values[1:]
        for index in range(len(score_vals)):
            gene = gene_names[index]
            key = (disease, gene)
            
            disease_lncRNA_dict[key] = float(score_vals[index])
    fp.close()
    '''for line in open(master_file):
        values = line.rstrip('\r\n').split('\t')
        gene = values[1]
        disease = values[2]
        score = float(values[5])
        disease_lncRNA_dict[(disease, gene)] = score
        master_disease_set.add(disease)
        master_lncRNA_set.add(gene)
    '''    
    return disease_lncRNA_dict, master_disease_set, master_lncRNA_set

'''def plot_lncRNA_plot(lncRNAdisease_list, disease_set, ):
    for key in lncRNAdisease_list:
    if key[0] not in shared_disease:
        continue
'''

def select_association_for_roc(lncRNAdisease_list, shared_disease, master_lncRNA_set, dis_lncRNA):
    roc_label_list = []
    roc_score_list = []
    for val in lncRNAdisease_list:
        if val[0] not in shared_disease or val[1] not in master_lncRNA_set:
            continue
        
        #pdb.set_trace()
        score = dis_lncRNA[val]
        roc_label_list.append(1)
        roc_score_list.append(score)
        selected_list = set()
        for tmp_ind in range(100):
            choice_key = random.choice(list(master_lncRNA_set))
            #print choice_key
            selecte_key = (val[0], choice_key)
            if selecte_key not in lncRNAdisease_list and selecte_key not in selected_list:
                roc_label_list.append(0)
                roc_score_list.append(dis_lncRNA[selecte_key])
                selected_list.add(selecte_key)
                break
    return roc_label_list, roc_score_list

def map_to_doid_disease():
    disease_doid = read_DOID_BTO_map()
    disease_set = set()
    lncRNAdisease_list = []
    fw = open('data/disease/data_disease_doid.txt', 'w')
    for line in open('data/disease/data_disease_new.txt'):
        values = line.rstrip('\r\n').split('\t')
        gene = values[1]
        disease = values[2].upper()
        if disease in disease_doid:
            new_line = line.replace(disease, disease_doid[disease])
            fw.write(new_line)
        else:
            fw.write(line)
    fw.close()
        #lncRNAdisease_dict.setdefault(c, set()).add(gene)
        #lncRNAdisease_list.append((disease, gene))
        #disease_set.add(disease)
        
def read_disease_files(file_name):
    disease_genes = {}
    for line in open(file_name, 'r'):
        values = line.rstrip('\r\n').split('\t')
        gene = values[1]
        disease = values[2]
        disease_genes.setdefault(disease, set([])).add(gene)
    return disease_genes

def performance_accuracy_for_random(expanison = False):
    #lncRNAdisease_set, DISEASE_set, disease_gene_dict = get_disease_in_database()
    #shared_disease = lncRNAdisease_set & DISEASE_set
    datasource =['E-MTAB-513', 'GSE43520', 'GSE30352', 'GTEx']
    #resultfiles = ['result/E-MTAB-513_lncRNA_knn.tsv', 'result/GSE43520_lncRNA_knn.tsv', 'result/GSE30352_lncRNA_knn.tsv', 'result/GTEx_lncRNA_knn.tsv']
    #resultfiles = ['result/E-MTAB-513_lncRNA_coexp_3nn.tsv', 'result/GSE43520_lncRNA_coexp_3nn.tsv', 'result/GSE30352_lncRNA_coexp_3nn.tsv', 
    #               'result/GTEx_lncRNA_coexp_3nn.tsv']
    resultfiles = ['result/E-MTAB-513_lncRNA.tsv', 'result/GSE43520_lncRNA.tsv', 'result/GSE30352_lncRNA.tsv', 'result/GTEx_lncRNA.tsv']
    disease_set = set()
    lncRNAdisease_list = set([])
    disease_genes = read_disease_files('data/disease/data_disease_doid.txt')
    oboparser = obo_object()
    oboparser.read_obo_file()
    #for line in open('data/disease/data_disease_new.txt'):
    for line in open('data/disease/data_disease_doid.txt', 'r'):
        values = line.rstrip('\r\n').split('\t')
        gene = values[1]
        disease = values[2].upper()
        #lncRNAdisease_dict.setdefault(c, set()).add(gene)
        lncRNAdisease_list.add((disease, gene))
        disease_set.add(disease)
        # exapnd association from its child disease
        if expanison:
            if 'DOID' in disease:
                child_disease = oboparser.getDescendents(disease)
                for dis in child_disease:
                    if disease_genes.has_key(dis):
                        genes = disease_genes[dis]
                        for id1_exp in genes:
                            lncRNAdisease_list.add((disease, id1_exp))
                            
                parent_disease = oboparser.getDescendents(disease)
                own_genes = disease_genes[disease]
                for dis in parent_disease:
                    for id1_exp in own_genes:
                        lncRNAdisease_list.add((dis, id1_exp))
    
    gencode_dis_lncRNA, gencode_master_disease_set, gencode_master_lncRNA_set = read_master_files(resultfiles[0])
    GSE43520_dis_lncRNA, GSE43520_master_disease_set, GSE43520_master_lncRNA_set = read_master_files(resultfiles[1])        
    GSE30352_dis_lncRNA, GSE30352_master_disease_set, GSE30352_master_lncRNA_set = read_master_files(resultfiles[2])    
    GTEx_dis_lncRNA, GTEx_master_disease_set, GTEx_master_lncRNA_set = read_master_files(resultfiles[3]) 
    
    gencode_shared_disease =  gencode_master_disease_set & disease_set   
    GSE43520_shared_disease =  GSE43520_master_disease_set & disease_set  
    GSE30352_shared_disease =  GSE30352_master_disease_set & disease_set
    GTEx_shared_disease =  GTEx_master_disease_set & disease_set
    #pdb.set_trace()

    
    genecode_label_roc_list, genecode_score_roc_list = select_association_for_roc(lncRNAdisease_list, gencode_shared_disease, gencode_master_lncRNA_set, gencode_dis_lncRNA)
    GSE43520_label_roc_list, GSE43520_score_roc_list = select_association_for_roc(lncRNAdisease_list, GSE43520_shared_disease, GSE43520_master_lncRNA_set, GSE43520_dis_lncRNA)
    GSE30352_label_roc_list, GSE30352_score_roc_list = select_association_for_roc(lncRNAdisease_list, GSE30352_shared_disease, GSE30352_master_lncRNA_set, GSE30352_dis_lncRNA)
    GTEx_label_roc_list, GTEx_score_roc_list = select_association_for_roc(lncRNAdisease_list, GTEx_shared_disease, GTEx_master_lncRNA_set, GTEx_dis_lncRNA)
    #pdb.set_trace()
    plt.close("all")
    Figure = plt.figure() 
    #pdb.set_trace()
    plot_roc_curve_lncRNA(genecode_label_roc_list, genecode_score_roc_list, datasource[0])
    plot_roc_curve_lncRNA(GSE43520_label_roc_list, GSE43520_score_roc_list, datasource[1])
    plot_roc_curve_lncRNA(GSE30352_label_roc_list, GSE30352_score_roc_list, datasource[2])
    plot_roc_curve_lncRNA(GTEx_label_roc_list, GTEx_score_roc_list, datasource[3])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    #plt.savefig(save_fig_dir + selected + '_' + class_type + '.png') 
    plt.show()    

def performance_accuracy_for_random_multi(expanison = False):
    #lncRNAdisease_set, DISEASE_set, disease_gene_dict = get_disease_in_database()
    #shared_disease = lncRNAdisease_set & DISEASE_set
    datasource =['E-MTAB-513', 'GSE43520', 'GSE30352', 'GTEx']
    #resultfiles = ['result/E-MTAB-513_lncRNA_knn.tsv', 'result/GSE43520_lncRNA_knn.tsv', 'result/GSE30352_lncRNA_knn.tsv', 'result/GTEx_lncRNA_knn.tsv']
    resultfiles = ['result/E-MTAB-513_lncRNA_coexp_1nn.tsv', 'result/GSE43520_lncRNA_coexp_1nn.tsv', 'result/GSE30352_lncRNA_coexp_1nn.tsv', 
                   'result/GTEx_lncRNA_coexp_1nn.tsv']
    #resultfiles = ['result/E-MTAB-513_lncRNA.tsv', 'result/GSE43520_lncRNA.tsv', 'result/GSE30352_lncRNA.tsv', 'result/GTEx_lncRNA.tsv']
    disease_set = set()
    lncRNAdisease_list = set([])
    disease_genes = read_disease_files('data/disease/data_disease_doid.txt')
    oboparser = obo_object()
    oboparser.read_obo_file()
    #for line in open('data/disease/data_disease_new.txt'):
    for line in open('data/disease/data_disease_doid.txt', 'r'):
        values = line.rstrip('\r\n').split('\t')
        gene = values[1]
        disease = values[2].upper()
        #lncRNAdisease_dict.setdefault(c, set()).add(gene)
        lncRNAdisease_list.add((disease, gene))
        disease_set.add(disease)
        # exapnd association from its child disease
        if expanison:
            if 'DOID' in disease:
                child_disease = oboparser.getDescendents(disease)
                for dis in child_disease:
                    if disease_genes.has_key(dis):
                        genes = disease_genes[dis]
                        for id1_exp in genes:
                            lncRNAdisease_list.add((disease, id1_exp))
                            
                parent_disease = oboparser.getAncestors(disease)
                #pdb.set_trace()
                own_genes = disease_genes[disease]
                for dis in parent_disease:
                    for id1_exp in own_genes:
                        lncRNAdisease_list.add((dis, id1_exp))
    
    gencode_dis_lncRNA, gencode_master_disease_set, gencode_master_lncRNA_set = read_master_files(resultfiles[0])
    GSE43520_dis_lncRNA, GSE43520_master_disease_set, GSE43520_master_lncRNA_set = read_master_files(resultfiles[1])        
    GSE30352_dis_lncRNA, GSE30352_master_disease_set, GSE30352_master_lncRNA_set = read_master_files(resultfiles[2])    
    GTEx_dis_lncRNA, GTEx_master_disease_set, GTEx_master_lncRNA_set = read_master_files(resultfiles[3]) 
    
    gencode_shared_disease =  gencode_master_disease_set & disease_set   
    GSE43520_shared_disease =  GSE43520_master_disease_set & disease_set  
    GSE30352_shared_disease =  GSE30352_master_disease_set & disease_set
    GTEx_shared_disease =  GTEx_master_disease_set & disease_set
    #pdb.set_trace()
    genecode_label_roc_list, genecode_score_roc_list =[],[]
    GSE43520_label_roc_list, GSE43520_score_roc_list =[], []
    GSE30352_label_roc_list, GSE30352_score_roc_list = [], []
    GTEx_label_roc_list, GTEx_score_roc_list = [], []
    for time in range(10):
        genecode_label_roc_list1, genecode_score_roc_list1 = select_association_for_roc(lncRNAdisease_list, gencode_shared_disease, gencode_master_lncRNA_set, gencode_dis_lncRNA)
        print 'gencode', len(genecode_label_roc_list1)
        genecode_label_roc_list = genecode_label_roc_list + genecode_label_roc_list1
        genecode_score_roc_list = genecode_score_roc_list + genecode_score_roc_list1
        GSE43520_label_roc_list1, GSE43520_score_roc_list1 = select_association_for_roc(lncRNAdisease_list, GSE43520_shared_disease, GSE43520_master_lncRNA_set, GSE43520_dis_lncRNA)
        print 'GSE43520', len(GSE43520_label_roc_list1)
        GSE43520_label_roc_list = GSE43520_label_roc_list + GSE43520_label_roc_list1
        GSE43520_score_roc_list = GSE43520_score_roc_list + GSE43520_score_roc_list1        
        GSE30352_label_roc_list1, GSE30352_score_roc_list1 = select_association_for_roc(lncRNAdisease_list, GSE30352_shared_disease, GSE30352_master_lncRNA_set, GSE30352_dis_lncRNA)
        print 'GSE30352', len(GSE30352_label_roc_list1)
        GSE30352_label_roc_list = GSE30352_label_roc_list + GSE30352_label_roc_list1
        GSE30352_score_roc_list = GSE30352_score_roc_list + GSE30352_score_roc_list1         
        GTEx_label_roc_list1, GTEx_score_roc_list1 = select_association_for_roc(lncRNAdisease_list, GTEx_shared_disease, GTEx_master_lncRNA_set, GTEx_dis_lncRNA)
        print 'GTEx', len(GTEx_label_roc_list1)
        GTEx_label_roc_list = GTEx_label_roc_list + GTEx_label_roc_list1
        GTEx_score_roc_list = GTEx_score_roc_list + GTEx_score_roc_list1 
    #pdb.set_trace()
    plt.close("all")
    Figure = plt.figure() 
    #pdb.set_trace()
    plot_roc_curve_lncRNA(genecode_label_roc_list, genecode_score_roc_list, datasource[0])
    plot_roc_curve_lncRNA(GSE43520_label_roc_list, GSE43520_score_roc_list, datasource[1])
    plot_roc_curve_lncRNA(GSE30352_label_roc_list, GSE30352_score_roc_list, datasource[2])
    plot_roc_curve_lncRNA(GTEx_label_roc_list, GTEx_score_roc_list, datasource[3])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CNC')
    plt.legend(loc="lower right")
    #plt.savefig(save_fig_dir + selected + '_' + class_type + '.png') 
    plt.show()  

def performance_accuracy_for_random_multi_cancer_noncancer(expanison = False):
    #lncRNAdisease_set, DISEASE_set, disease_gene_dict = get_disease_in_database()
    #shared_disease = lncRNAdisease_set & DISEASE_set
    datasource =['E-MTAB-513', 'GSE43520', 'GSE30352', 'GTEx']
    #resultfiles = ['result/E-MTAB-513_lncRNA_knn.tsv', 'result/GSE43520_lncRNA_knn.tsv', 'result/GSE30352_lncRNA_knn.tsv', 'result/GTEx_lncRNA_knn.tsv']
    #resultfiles = ['result/E-MTAB-513_lncRNA_coexp_5nn.tsv', 'result/GSE43520_lncRNA_coexp_5nn.tsv', 'result/GSE30352_lncRNA_coexp_5nn.tsv', 
    #               'result/GTEx_lncRNA_coexp_5nn.tsv']
    resultfiles = ['result/E-MTAB-513_lncRNA.tsv', 'result/GSE43520_lncRNA.tsv', 'result/GSE30352_lncRNA.tsv', 'result/GTEx_lncRNA.tsv']
    disease_set = set()
    lncRNAdisease_list = set([])
    disease_genes = read_disease_files('data/disease/data_disease_doid.txt')
    oboparser = obo_object()
    oboparser.read_obo_file()
    cancer_dis = set()
    noncancer_dis = set()
    all_disease = set()
    #for line in open('data/disease/data_disease_new.txt'):
    for line in open('data/disease/data_disease_doid.txt', 'r'):
        values = line.rstrip('\r\n').split('\t')
        gene = values[1]
        disease = values[2].upper()
        #lncRNAdisease_dict.setdefault(c, set()).add(gene)
        lncRNAdisease_list.add((disease, gene))
        disease_set.add(disease)
        all_disease.add(disease)
        # exapnd association from its child disease
        if expanison:
            if 'DOID' in disease:
                child_disease = oboparser.getDescendents(disease)
                for dis in child_disease:
                    if disease_genes.has_key(dis):
                        genes = disease_genes[dis]
                        for id1_exp in genes:
                            lncRNAdisease_list.add((disease, id1_exp))
                            
                parent_disease = oboparser.getAncestors(disease)
                #pdb.set_trace()
                own_genes = disease_genes[disease]
                for dis in parent_disease:
                    all_disease.add(dis)
                    for id1_exp in own_genes:
                        lncRNAdisease_list.add((dis, id1_exp))
    
    gencode_dis_lncRNA, gencode_master_disease_set, gencode_master_lncRNA_set = read_master_files(resultfiles[0])
    GSE43520_dis_lncRNA, GSE43520_master_disease_set, GSE43520_master_lncRNA_set = read_master_files(resultfiles[1])        
    GSE30352_dis_lncRNA, GSE30352_master_disease_set, GSE30352_master_lncRNA_set = read_master_files(resultfiles[2])    
    GTEx_dis_lncRNA, GTEx_master_disease_set, GTEx_master_lncRNA_set = read_master_files(resultfiles[3]) 
    
    for dis_tmp in gencode_master_disease_set:    
        parent_disease = oboparser.getAncestor_all(dis_tmp)
        if 'DOID:162' in parent_disease:
            cancer_dis.add(dis_tmp)
        else:
            noncancer_dis.add(dis_tmp)           
            
    print len(cancer_dis), len(noncancer_dis)
    
    gencode_shared_disease =  gencode_master_disease_set & noncancer_dis   
    GSE43520_shared_disease =  gencode_master_disease_set & noncancer_dis  
    GSE30352_shared_disease =  gencode_master_disease_set & noncancer_dis
    GTEx_shared_disease =  gencode_master_disease_set & noncancer_dis

    print len(gencode_shared_disease),len(GTEx_shared_disease)
    #pdb.set_trace()
    genecode_label_roc_list, genecode_score_roc_list =[],[]
    GSE43520_label_roc_list, GSE43520_score_roc_list =[], []
    GSE30352_label_roc_list, GSE30352_score_roc_list = [], []
    GTEx_label_roc_list, GTEx_score_roc_list = [], []
    for time in range(10):
        genecode_label_roc_list1, genecode_score_roc_list1 = select_association_for_roc(lncRNAdisease_list, gencode_shared_disease, gencode_master_lncRNA_set, gencode_dis_lncRNA)
        print 'gencode', len(genecode_label_roc_list1)
        genecode_label_roc_list = genecode_label_roc_list + genecode_label_roc_list1
        genecode_score_roc_list = genecode_score_roc_list + genecode_score_roc_list1
        GSE43520_label_roc_list1, GSE43520_score_roc_list1 = select_association_for_roc(lncRNAdisease_list, GSE43520_shared_disease, GSE43520_master_lncRNA_set, GSE43520_dis_lncRNA)
        print 'GSE43520', len(GSE43520_label_roc_list1)
        GSE43520_label_roc_list = GSE43520_label_roc_list + GSE43520_label_roc_list1
        GSE43520_score_roc_list = GSE43520_score_roc_list + GSE43520_score_roc_list1        
        GSE30352_label_roc_list1, GSE30352_score_roc_list1 = select_association_for_roc(lncRNAdisease_list, GSE30352_shared_disease, GSE30352_master_lncRNA_set, GSE30352_dis_lncRNA)
        print 'GSE30352', len(GSE30352_label_roc_list1)
        GSE30352_label_roc_list = GSE30352_label_roc_list + GSE30352_label_roc_list1
        GSE30352_score_roc_list = GSE30352_score_roc_list + GSE30352_score_roc_list1         
        GTEx_label_roc_list1, GTEx_score_roc_list1 = select_association_for_roc(lncRNAdisease_list, GTEx_shared_disease, GTEx_master_lncRNA_set, GTEx_dis_lncRNA)
        print 'GTEx', len(GTEx_label_roc_list1)
        GTEx_label_roc_list = GTEx_label_roc_list + GTEx_label_roc_list1
        GTEx_score_roc_list = GTEx_score_roc_list + GTEx_score_roc_list1 
    #pdb.set_trace()
    plt.close("all")
    Figure = plt.figure() 
    #pdb.set_trace()
    plot_roc_curve_lncRNA(genecode_label_roc_list, genecode_score_roc_list, datasource[0])
    plot_roc_curve_lncRNA(GSE43520_label_roc_list, GSE43520_score_roc_list, datasource[1])
    plot_roc_curve_lncRNA(GSE30352_label_roc_list, GSE30352_score_roc_list, datasource[2])
    plot_roc_curve_lncRNA(GTEx_label_roc_list, GTEx_score_roc_list, datasource[3])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('DislncRF')
    plt.title('Non-cancer diseases')
    #plt.title('CNC (K=5)')
    plt.legend(loc="lower right")
    #plt.savefig(save_fig_dir + selected + '_' + class_type + '.png') 
    plt.show()


def performance_accuracy_for_random_old():
    #lncRNAdisease_set, DISEASE_set, disease_gene_dict = get_disease_in_database()
    #shared_disease = lncRNAdisease_set & DISEASE_set
    datasource =['E-MTAB-513', 'GSE43520', 'GSE30352', 'GTEx']
    #resultfiles = ['result/E-MTAB-513_lncRNA_knn.tsv', 'result/GSE43520_lncRNA_knn.tsv', 'result/GSE30352_lncRNA_knn.tsv', 'result/GTEx_lncRNA_knn.tsv']
    #resultfiles = ['result/E-MTAB-513_lncRNA_coexp_3nn.tsv', 'result/GSE43520_lncRNA_coexp_3nn.tsv', 'result/GSE30352_lncRNA_coexp_3nn.tsv', 
    #               'result/GTEx_lncRNA_coexp_3nn.tsv']
    resultfiles = ['result/E-MTAB-513_lncRNA.tsv', 'result/GSE43520_lncRNA.tsv', 'result/GSE30352_lncRNA.tsv', 'result/GTEx_lncRNA.tsv']
    disease_set = set()
    lncRNAdisease_list = []
    #for line in open('data/disease/data_disease_new.txt'):
    for line in open('data/disease/data_disease_doid.txt', 'r'):
        values = line.rstrip('\r\n').split('\t')
        gene = values[1]
        disease = values[2].upper()
        #lncRNAdisease_dict.setdefault(c, set()).add(gene)
        lncRNAdisease_list.append((disease, gene))
        disease_set.add(disease)
    
    gencode_dis_lncRNA, gencode_master_disease_set, gencode_master_lncRNA_set = read_master_files(resultfiles[0])
    GSE43520_dis_lncRNA, GSE43520_master_disease_set, GSE43520_master_lncRNA_set = read_master_files(resultfiles[1])        
    GSE30352_dis_lncRNA, GSE30352_master_disease_set, GSE30352_master_lncRNA_set = read_master_files(resultfiles[2])    
    
    gencode_shared_disease =  gencode_master_disease_set & disease_set   
    GSE43520_shared_disease =  GSE43520_master_disease_set & disease_set  
    GSE30352_shared_disease =  GSE30352_master_disease_set & disease_set
    
    genecode_label_roc_list, genecode_score_roc_list = select_association_for_roc(lncRNAdisease_list, gencode_shared_disease, gencode_master_lncRNA_set, gencode_dis_lncRNA)
    GSE43520_label_roc_list, GSE43520_score_roc_list = select_association_for_roc(lncRNAdisease_list, GSE43520_shared_disease, GSE43520_master_lncRNA_set, GSE43520_dis_lncRNA)
    GSE30352_label_roc_list, GSE30352_score_roc_list = select_association_for_roc(lncRNAdisease_list, GSE30352_shared_disease, GSE30352_master_lncRNA_set, GSE30352_dis_lncRNA)
    #pdb.set_trace()
    plt.clf()
    plt.cla()
    Figure = plt.figure() 
    #pdb.set_trace()
    plot_roc_curve_lncRNA(genecode_label_roc_list, genecode_score_roc_list, datasource[0])
    plot_roc_curve_lncRNA(GSE43520_label_roc_list, GSE43520_score_roc_list, datasource[1])
    plot_roc_curve_lncRNA(GSE30352_label_roc_list, GSE30352_score_roc_list, datasource[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    #plt.savefig(save_fig_dir + selected + '_' + class_type + '.png') 
    plt.show()    
    
def plot_roc_curve_lncRNA(labels, probality, legend_text):
    #fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    
    rects1 = plt.plot(fpr, tpr, label=legend_text +' (AUC=%6.3f) ' %roc_auc)

def read_final_master(master_file = 'master_files/predictions.tsv'):
    disease_lncRNA_dict = {}
    with open(master_file, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            gene = values[1]
            disease = values[2]
            score = float(values[-2])
            
    
def get_overlap_predcited_real(real_disease_file = 'new_verfied_lncRNA_disease/breast_lung', predicted_master_file = 'result/E-MTAB-513_lncRNA.tsv' ):
    gencode_dis_lncRNA, gencode_master_disease_set, gencode_master_lncRNA_set = read_master_files(predicted_master_file)
    gene_type_dict,gene_name_ensg, gene_id_position = read_gencode_gene_type()
    
    predicted_disease_lncRNA = {}
    for key, val in gencode_dis_lncRNA.iteritems():
        dis, gene = key
        predicted_disease_lncRNA.setdefault(dis.upper(), []).append((val, gene))
    
    experiment_disease_lncRNA = {}
    with open(real_disease_file, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            
            dis_local = values[0].upper()
            try:
                gene_local = values[1] 
                experiment_disease_lncRNA.setdefault(dis_local, set()).add(gene_local) 
            except:
                print line
            
    disease = 'breast cancer'
    lncRNA_snp = get_lcnRNA_disease_snp_assoc(gene_id_position, gencode_master_lncRNA_set, disease)
    pdb.set_trace()
    pred_sort_dis_lncRNA = predicted_disease_lncRNA[disease.upper()]
    pred_sort_dis_lncRNA.sort(reverse =True) 
    #pdb.set_trace()
    top_200_lncRNA = [val[1] for val in pred_sort_dis_lncRNA[:200]]
    #pdb.set_trace()
    experm_disease_lncRNA = experiment_disease_lncRNA[disease.upper()]
    for val in experm_disease_lncRNA:
        if val in top_200_lncRNA:
            print val, top_200_lncRNA.index(val)
        else:
            print 'not in top 200 associated lncRNA'

def read_cancer_lncRNA_database(database_file = 'data/disease/lncRNA_cancer_association.txt'):
    cancer_lncRNA_assoc = {}
    with open(database_file, 'r') as fp:
        head = True
        for line in fp:
            if head:
                head = False
                continue
            values = line.rstrip('\r\n').split('\t')
            ensmb_id = values[4]
            if ensmb_id == 'N/A':
                continue
            cancer = values[7]
            cancer_lncRNA_assoc.setdefault(cancer, set()).add(ensmb_id)
    fw = open('data/disease/Lnc2Cancer.txt', 'w')
    for key, val in cancer_lncRNA_assoc.iteritems():
        for ensem_id in val:
            fw.write('%s\n'  % '\t'.join(('9606', ensem_id, key, '0', 'database',
                    '0.900', 'Lnc2Cancer', '', '')))
    fw.close()        
    return cancer_lncRNA_assoc

def get_counts_key(data):
    key_count = Counter(data)
    data_len = len(key_count)
    
    #for val, ind in key_count.iteritems():
    counts = key_count.values() 
    keys = key_count.keys()
    
    return counts, keys
    
def plot_mean_variance(menMeans,womenMeans, num_tissue = 16):
    #from matplotlib.ticker import AutoMinorLocator
    #x_label = ['unannotated', 'antisense', 'circRNA', 'pseudogene', 'protein coding', 'lincRNA', 'processed_transcript']
    #ind = np.arange(len(x_label))  # the x locations for the groups
    width = 0.35 
    
    ind = np.arange(len(menMeans))
    count_man, ind_man = get_counts_key(menMeans)
    count_women, ind_women = get_counts_key(womenMeans)
    #pdb.set_trace()
    fig, ax = plt.subplots(figsize=(8,8))

    rects1 = ax.bar(ind_man, count_man, width, color='r')
    #rects1 = plt.hist(menMeans, np.arange(num_tissue), color=['crimson'], label = 'lncRNA')
    ind_women = [val + width for val in ind_women] 
    rects2 = ax.bar(ind_women, count_women, width, color='b')
    #rects2 = plt.hist(womenMeans, np.arange(num_tissue), color=['b'], label='PCG')
    #plt.hist([menMeans,womenMeans], num_tissue, histtype='bar', color=['b', 'crimson'], label=['lncRNA', 'PCG'])
    # add some
    plt.ylabel('Count', fontsize=20)
    plt.xlabel('Number of tissues', fontsize=20)
    #ax.set_title('Variation of binding energy between sponge and shuffled.')
    #ax.set_xticks(ind)
    #ind1 = [str(val) for val in ind]
    #ax.set_xticklabels(ind )
    #plt.xticks(ind, ind1)
    #minor_locator = AutoMinorLocator(1)
    #ax.xaxis.set_minor_locator(minor_locator)
    ax.legend( (rects1[0], rects2[0]), ('lncRNA', 'PCG'), loc='upper center')
    '''
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.01*height, '%0.1f'%height,
                    ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    ax.set_ylim([0,100])
    '''
    plt.tight_layout()
    
    #ax.set_ylim([-25, 0])
    #ax.set_ylim(ax.get_ylim()[::-1])
    plt.show() 

def plot_expression_distribution(input_file, RNAseq = True,  data=0, ratio=5, confidence=2, use_mean = False, disease_acc = None, log2_flag  = True):
    print expression_file
    ensg_ensp_map = get_ENSP_ENSG_map()
    if RNAseq:
        gene_type_dict,gene_name_ensg, gene_id_position = read_gencode_gene_type()
        if data == 0:
            whole_data, samples = read_human_RNAseq_expression(input_file, gene_name_ensg, log2 = log2_flag) # for microarray expression data
        elif data ==1:
            whole_data, samples = read_evolutionary_expression_data(input_file, use_mean = use_mean, log2 = log2_flag)
        elif data == 2:
            whole_data, samples = read_average_read_to_normalized_RPKM(input_file, use_mean = use_mean, log2 = log2_flag)
        else:
            whole_data, samples = read_tissue_database()
        #pdb.set_trace()
    else:
        whole_data = read_normalized_series_file(input_file)  
        
    disease_lncRNA_data, lncRNAlabels,lncRNA_list, atmp  = get_mRNA_lncRNA_expression_RNAseq_data(whole_data, gene_type_dict=gene_type_dict, 
                                                                                                  mRNA=False)
    
    disease_mRNA_data, mRNAlabels, atmp, mRNA_list = get_mRNA_lncRNA_expression_RNAseq_data(whole_data, ensg_ensp_map=ensg_ensp_map, 
                                                                                            gene_type_dict=gene_type_dict) 
    #pdb.set_trace()
    #plot mean distribution
    #plot_gaussian_distribution(disease_lncRNA_data.mean(axis=0), disease_mRNA_data.mean(axis=0))
    cutoff = 0.5
    #disease_lncRNA_data = preprocess_data_tissue(disease_lncRNA_data)
    #disease_mRNA_data = preprocess_data_tissue(disease_mRNA_data)
    #cutoff = 0.25
    
    #normalize between tissues
    '''disease_mRNA_data, scaler = preprocess_data(disease_mRNA_data.transpose())
    disease_mRNA_data = disease_mRNA_data.transpose()
    disease_lncRNA_data, scaler = preprocess_data(disease_lncRNA_data.transpose())
    disease_lncRNA_data = disease_lncRNA_data.transpose()
    plot_gaussian_distribution(disease_lncRNA_data.mean(axis=0), disease_mRNA_data.mean(axis=0))
    '''
    
    num_tissue = len(disease_lncRNA_data[0])
    ind_cutoff = disease_mRNA_data < cutoff
    big_cutoff = disease_mRNA_data >= cutoff
    disease_mRNA_data[ind_cutoff] = 0
    disease_mRNA_data[big_cutoff] = 1
    mRNA_count_gene = disease_mRNA_data.sum(axis = 1)
    
    ind_cutoff = disease_lncRNA_data < cutoff
    big_cutoff = disease_lncRNA_data >= cutoff
    disease_lncRNA_data[ind_cutoff] = 0
    disease_lncRNA_data[big_cutoff] = 1
    lncRNA_count_gene = disease_lncRNA_data.sum(axis = 1)
    #pdb.set_trace()
    plot_mean_variance(lncRNA_count_gene, mRNA_count_gene, num_tissue =num_tissue)
    #check the distribution in tissues

def read_lnccancer():
    enst_ensg = map_enst_to_ensg()
    doid_bto_map = read_DOID_BTO_map()
    fw = open('data/disease/Lnc2Cancer_doid.txt', 'w')
    with open('data/disease/Lnc2Cancer.txt', 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            gene = values[1]
            disease = values[2].upper()
            if not doid_bto_map.has_key(disease):
                continue
            doid = doid_bto_map[disease]
            if 'ENST' in gene:
                if enst_ensg.has_key(gene):
                    ensg = enst_ensg[gene]
                    fw.write('9606\t' + ensg + '\t' + doid + '\t0\tdatabase\t0.900\tLnc2Cancer\n')
            else:
                fw.write('9606\t' + gene + '\t' + doid + '\t0\tdatabase\t0.900\tLnc2Cancer\n')
                    
    
    fw.close()

def read_cancer_name():
    enst_ensg = map_enst_to_ensg()
    doid_bto_map = read_DOID_BTO_map()
    fw = open('data/disease/lnc2c_doid','w')
    with open('data/disease/lnc2cancer', 'r') as fp:
        for line in fp:   
            values = line.rstrip().split('\t') 
            gene = values[3]
            if 'N/A' in gene:
                continue    
            disease = values[6].upper()
            if not doid_bto_map.has_key(disease):
                doid = disease
            else:
                doid = doid_bto_map[disease]
            if 'ENST' in gene:
                if enst_ensg.has_key(gene):
                    ensg = enst_ensg[gene]
                    fw.write('9606\t' + ensg + '\t' + doid + '\t0\tdatabase\t0.900\tLnc2Cancer\n')
            else:
                fw.write('9606\t' + gene + '\t' + doid + '\t0\tdatabase\t0.900\tLnc2Cancer\n')            
                
    
    fw.close()
def merge_two_lncRNA_disease():
    lncran_dis_gene = set()
    fw = open('data/disease/data_disease_doid_new.txt', 'w')
    with open('data/disease/data_disease_doid_more.txt', 'r') as fp:
        for line in fp:
            if 'DOID:' not in line:
                continue
            values = line.rstrip().split('\t')
            tval = (values[1], values[2])
            if tval in lncran_dis_gene:
                continue
            else:
                lncran_dis_gene.add(tval)
                fw.write(line)
            
    with open('data/disease/lnc2c_doid', 'r') as fp:
        for line in fp:
            if 'DOID:' not in line:
                continue
            values = line.rstrip().split('\t')
            tval = (values[1], values[2])
            if tval in lncran_dis_gene:
                continue
            else:
                lncran_dis_gene.add(tval)
                fw.write(line)            
    
    fw.close()
    
def read_DisGeNET_database():
    doid_bto_map = read_DOID_BTO_map()
    ensg_ensp = get_ENSP_ENSG_map()
    gene_type_dict, gene_name_ensg, gene_id_position = read_gencode_gene_type()
    disease_gene = {}
    fw  = open('data/disease/DisGeNET_new.txt', 'w')
    head = True
    with gzip.open('data/disease/curated_gene_disease_associations.txt.gz', 'r') as fp:
        for line in fp:
            if head:
                head = False
                continue
            values = line.rstrip('\r\n').split('\t')
            gene = values[3]
            if not gene_name_ensg.has_key(gene):
                continue
                
            disease = values[5].upper()

            '''if not doid_bto_map.has_key(disease):
                if ',' in disease:
                    dis_tmp = disease.split(',')
                    for dis in dis_tmp:
                        if doid_bto_map.has_key(dis):
                            disease = dis
                            break
                else:
                    continue
            '''
            if not doid_bto_map.has_key(disease):
                continue
            new_gene = gene_name_ensg[gene]
            if gene_type_dict[new_gene] != 'protein_coding':
                continue
            if not ensg_ensp.has_key(new_gene):
                continue
            new_gene = ensg_ensp[new_gene]
            new_disease =  doid_bto_map[disease]        
            
            fw.write(new_gene + '\t' + new_disease + '\t' + disease + '\t' + values[2] + '\n')
            
    
    fw.close()

def remove_unreal_lncrna():
    lncRNA_type_dicts = ['3prime_overlapping_ncrna', 'ambiguous_orf', 'antisense', 'antisense_RNA', 'lincRNA', 'ncrna_host', 'non_coding', 
        'non_stop_decay', 'processed_transcript', 'retained_intron', 'sense_intronic', 'sense_overlapping']
    gene_type_dict, gene_name_ensg, gene_id_position = read_gencode_gene_type()
    fw = open('data/disease/data_disease_doid_latest.txt', 'w')
    with open('data/disease/data_disease_doid_new.txt', 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            ensg = values[1]
            if ensg not in gene_type_dict:
                continue
            if gene_type_dict[ensg] not in lncRNA_type_dicts:
                continue
            fw.write(line)    
    fw.close()      
    
def get_BTO_tissue_map(map_file = 'data/tissue_map'):
    tissue_map = {}
    with open(map_file, 'r') as fp:
        for line in fp:
            tissue, bto = line.rstrip().split(',')
            tissue_map[tissue.upper()] = bto
            
    return tissue_map

def read_DOID_BTO_map(map_file = 'data/DOID_BTO_mapping.tsv'):
    doid_bto_map = {}
    with open(map_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            identifier = values[1]
            doid_bto_map[values[2].upper()] = identifier
    
    return doid_bto_map

def read_textmining_result(disease_tissue, tissue_file ='data/disease/tissue_disease/diff_textmining_knowledge_sorted.tsv'):
    #textming_tissue = {}
    with open(tissue_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            tissue = values[1]
            disease = values[3]
            score = float(values[0])
            disease_tissue.setdefault(disease, []).append((tissue, score))
            
    #return textming_tissue
    
# tissue map: http://www.berkeleybop.org/ontologies/bto.obo
def read_tissue_disease_association(disease_tissue, disease_tissue_file = 'data/disease/tissue_disease/knowledge_tissue_disease.tsv'):
    #disease_tissue = {}
    with open(disease_tissue_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            score = float(values[-1])
            tissue = values[2]
            disease = values[5]
            disease_tissue.setdefault(disease, []).append((tissue, score))
            
    #return disease_tissue

def plot_tissue_imp_real_rf(imp_list, textming_list, disease, tissues, ylabel = 'Importance score', dataset = 'E-MTAB-513'):
    ind = np.arange(len(imp_list))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8,6))
    rects1 = ax.bar(ind, imp_list, width, color='b')
    rects2 = ax.bar(ind+width, textming_list, width, color='r')
    #plt.ylabel('Importance score')
    ax.legend( (rects1[0], rects2[0]), ('Random forest', 'Text mining and knowledge'), loc=1)
    ax.set_xticks(ind)
    ax.set_xticklabels(tissues, rotation=90, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(disease, fontsize=15)
    plt.tight_layout()
    #plt.xlim([0,5])
    #plt.xlabel('Tissue')
    #if disease == 'PROSTATE CANCER' or disease == "PARKINSON'S DISEASE":
    #    plt.show()
    #else:
    if disease == 'NON-SMALL CELL LUNG CARCINOMA' or disease == 'KIDNEY DISEASE':
        plt.show()
    #plt.savefig('imp_text_rf/' + dataset + '_' + disease + '.eps', format='eps')
    plt.clf()
    plt.close()


def calculate_cc_tissue_imp(result_imp_file = 'result/E-MTAB-513_lncRNA.tsv.imp', dataset = 'E-MTAB-513'):
    BTO_tissue_map = get_BTO_tissue_map()
    disease_tissue = {}
    read_textmining_result(disease_tissue)
    read_tissue_disease_association(disease_tissue)
    
    #disease_tissue = {}
    #disease_tissue.update(disease_tissue1)
    #disease_tissue.update(disease_tissue2)
    #pdb.set_trace()
    doid_bto_map = read_DOID_BTO_map()
    fp = open(result_imp_file, 'r')
    disease_dict={}
    index = 0
    for line in fp:
        values = line.rstrip('\r\n').split('\t')
        if index == 0:
            tissues = values[1:]
            index = index  + 1
        else:
            disease = values[0]        
            #imp_list = [(val1, float(val)) for val1, val in zip(tissues,values[1:])]
            imp_list = [float(val) for val in values[1:-1]]
            disease_dict[disease.upper()] = imp_list
            
    fp.close()
    
    #pdb.set_trace()
    BTO_tissues = [BTO_tissue_map[val.upper()] for val in tissues]
    cc_score = []
    for val1, val in disease_dict.iteritems():
        rf_know_imp = []
        if not doid_bto_map.has_key(val1):
            #print val1
            continue
        #print val1
        key = doid_bto_map[val1]
        #pdb.set_trace()
        if not disease_tissue.has_key(key):
            continue
        knowledge_tissues = disease_tissue[key]
        tissue_plot = []
        for ind, imp_score in enumerate(val):
            #pdb.set_trace()
            tissue = BTO_tissues[ind]
            know_score = get_knowledge_tissue_score(tissue, knowledge_tissues)
            if know_score == -1:
                continue
            rf_know_imp.append((imp_score, know_score))
            tissue_plot.append(tissues[ind])
            
        if not len(rf_know_imp):
            continue
        
        #print key
        #pdb.set_trace()    
        rf_list,know_list = map(list,zip(*rf_know_imp))
        sum_know = sum(know_list)
        sum_rf = sum(rf_list)
        know_list = [val/sum_know for val in know_list]
        rf_list = [val/sum_rf for val in rf_list]

        plot_tissue_imp_real_rf(rf_list, know_list, val1, tissue_plot, dataset = dataset)
        pcc, pval = stats.pearsonr(rf_list, know_list)
        if isnan( pcc):
            continue
        cc_score.append(abs(pcc))
        #pdb.set_trace()
        #print pcc
    print np.mean(cc_score)
#def calulate_cc_lists(a,b):
            
        
def get_knowledge_tissue_score(tissue, score_list):
    #pdb.set_trace()
    for val in score_list:
        if val[0] == tissue:
            return val[1]
        
    return -1

def map_enst_to_ensg():
    gene_enst_ensg = {}
    fp = gzip.open('data/dict/Homo_sapiens.GRCh37.70.gtf.gz')
    for line in fp:
        values = line.rstrip('\r\n').split('\t')
        gene_ann = values[-1]
        split_gene_ann = gene_ann.split(';')
        gene_type_info = split_gene_ann[4]
        gene_type = extract_gene_type_name(gene_type_info)
        if gene_type == 'protein_coding':
            continue
        gene_id_info = split_gene_ann[0]
        gene_id = extract_gene_type_name(gene_id_info)
        
        tran_id_info = split_gene_ann[1]
        tran_id = extract_gene_type_name(tran_id_info)
        gene_enst_ensg[tran_id] = gene_id  
    
    return gene_enst_ensg      
              
def read_predict_lncrna(predict_file = 'result/E-MTAB-513_lncRNA.tsv'):
    index = 0
    disease_lncrna = {}
    with open(predict_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            if index == 0:
                genes = values[1:]
            else:
                disease = values[0].upper()
                scores = values[1:]
                gene_scores = []
                for val1, val2 in zip(genes, scores):
                    gene_scores.append((float(val2), val1))
                disease_lncrna[disease] = gene_scores
            index = index + 1

    #gene_scores.sort(reverse=True)
    return disease_lncrna

def read_sign_expressed_IBD(gene_enst_ensg):
    lnc2IBD = set()
    with open('data/disease/IBD_lncRNA', 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            gene = values[1].split('.')[0]
            fold = abs(float(values[4]))
            pvalue = float(values[-1])
            if fold < 2 or pvalue > 0.01:
                continue
            if not gene_enst_ensg.has_key(gene):
                continue
            lnc2IBD.add(gene_enst_ensg[gene])

    return lnc2IBD  

def read_sign_expressed_IBD_all(gene_enst_ensg):
    lnc2IBD = set()
    with open('data/disease/IBD_lncRNA', 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            gene = values[1].split('.')[0]
            fold = abs(float(values[4]))
            pvalue = float(values[-1])
            #if fold < 2 or pvalue > 0.01:
            #    continue
            if not gene_enst_ensg.has_key(gene):
                continue
            lnc2IBD.add(gene_enst_ensg[gene])

    return lnc2IBD 
      
def case_study_IBD(predict_file):

    gene_enst_ensg = map_enst_to_ensg()
    #pdb.set_trace()
    #lnc2IBD = read_sign_expressed_IBD(gene_enst_ensg)
    lnc2IBD = read_ibd_diff_loci_lncRNA() #read_IBD_loci_lncRNA(gene_enst_ensg)
    print len(lnc2IBD)   
    #pdb.set_trace()     
    pred_scores = read_predict_lncrna(predict_file)# = 'result/E-MTAB-513_lncRNA.tsv')
    #gene_scores = pred_scores['INFLAMMATORY BOWEL DISEASE']
    gene_scores = pred_scores['DOID:0050589']
    gene_scores.sort(reverse=True)
    
    all_lncrna = [val[1] for val in gene_scores]
    overlap_lnc = len(lnc2IBD & set(all_lncrna))
    print 'overlap lcnrna', overlap_lnc
    print 'total # of lcnRNA', len(gene_scores)
    
    top_200_lncRNA = [val[1] for val in gene_scores[:100]]
    print top_200_lncRNA[:10]
    for val in lnc2IBD:
        if val in top_200_lncRNA:
            print val, top_200_lncRNA.index(val) + 1
        #else:
        #    print 'not in top 200 associated lncRNA'   
    
    #plot_circ_figs(gene_scores)
def get_100_lncRNAs(predict_file = 'E-MTAB-513_lncRNA.tsv'):
    gencode_pred_scores = read_predict_lncrna(predict_file)# = 'result/E-MTAB-513_lncRNA.tsv')
    #gene_scores = pred_scores['INFLAMMATORY BOWEL DISEASE']
    gencode_gene_scores = gencode_pred_scores['DOID:0050589']
    gencode_gene_scores.sort(reverse=True)
    
    gencode_lncrna = [val[1] for val in gencode_gene_scores[:100]]
    
    return set(gencode_lncrna)

def case_study_IBD_venn():

    gene_enst_ensg = map_enst_to_ensg()
    #pdb.set_trace()
    lnc2IBD = read_sign_expressed_IBD(gene_enst_ensg)
    #lnc2IBD = read_ibd_diff_loci_lncRNA() #read_IBD_loci_lncRNA(gene_enst_ensg)
    pred_scores = read_predict_lncrna('result/E-MTAB-513_lncRNA.tsv')# = 'result/E-MTAB-513_lncRNA.tsv')
    #gene_scores = pred_scores['INFLAMMATORY BOWEL DISEASE']
    gene_scores = pred_scores['DOID:0050589']
    gene_scores.sort(reverse=True)
    
    all_lncrna = [val[1] for val in gene_scores]
    lnc2IBD = lnc2IBD & set(all_lncrna)
    print len(lnc2IBD)   
    #for rna in lnc2IBD:
    #    print rna
    print 'gencode'    
    gencode_pred_lncRNAs = get_100_lncRNAs(predict_file = 'result/E-MTAB-513_lncRNA.tsv')# = 'result/E-MTAB-513_lncRNA.tsv')
    #pdb.set_trace() 
    #all_lncrna = [val for val in gencode_pred_lncRNAs]
    #for rna in gencode_pred_lncRNAs:
    #    print rna
    #gene_scores = pred_scores['INFLAMMATORY BOWEL DISEASE']
    gse43520_pred_lncRNAs = get_100_lncRNAs(predict_file = 'result/GSE43520_lncRNA.tsv')
    print 'gse43520'
    #for rna in gse43520_pred_lncRNAs:
    #    print rna    
    gtex_pred_lncRNAs = get_100_lncRNAs(predict_file = 'result/GTEx_lncRNA.tsv')
    print 'gtex'
    #for rna in gtex_pred_lncRNAs:
    #    print rna
    #overlap_lnc = len(lnc2IBD & set(all_lncrna))
    #print 'overlap lcnrna', overlap_lnc
    #print 'total # of lcnRNA', len(gene_scores)
    #pdb.set_trace() 
    set_labels = ('123 lncRNAs', 'E-MTAB-513', 'GSE43520', 'GTEx')
    #pdb.set_trace()
    v1 = venn.venn([lnc2IBD, gencode_pred_lncRNAs, gse43520_pred_lncRNAs, gtex_pred_lncRNAs], set_labels, figsize=(8,8))
    
    
def get_IBD_roc(predict_file, diff_lnc, lnc2IBD):

    #pdb.set_trace()     
    pred_scores = read_predict_lncrna(predict_file)# = 'result/E-MTAB-513_lncRNA.tsv')
    #gene_scores = pred_scores['INFLAMMATORY BOWEL DISEASE']
    gene_scores = pred_scores['DOID:0050589']
    score_dict = {}
    nega_rna = []
    for val1, val2 in gene_scores:
        score_dict[val2] = val1
        if val2 not in diff_lnc:
            nega_rna.append(val2)
        
    roc_score = []
    for val in lnc2IBD:
        if score_dict.has_key(val):
            roc_score.append(score_dict[val])
    #pdb.set_trace()
    num_len = len(roc_score)
    labels = [1] * num_len + [0] * num_len
    random.shuffle(nega_rna)
    for val in nega_rna[:num_len]:
        roc_score.append(score_dict[val])
    
    return labels, roc_score
    #plot_roc_curve_lncRNA(labels, roc_score, 'IBD')
    
def plot_IBD_roc():
    gene_enst_ensg = map_enst_to_ensg()
    #pdb.set_trace()
    diff_lnc = read_sign_expressed_IBD_all(gene_enst_ensg)
    #lnc2IBD = read_sign_expressed_IBD(gene_enst_ensg)
    lnc2IBD = read_ibd_diff_loci_lncRNA() #read_IBD_loci_lncRNA(gene_enst_ensg)
    print len(lnc2IBD)   
    datasource =['E-MTAB-513', 'GSE43520', 'GSE30352', 'GTEx']
    #resultfiles = ['result/E-MTAB-513_lncRNA_knn.tsv', 'result/GSE43520_lncRNA_knn.tsv', 'result/GSE30352_lncRNA_knn.tsv', 'result/GTEx_lncRNA_knn.tsv']
    #resultfiles = ['result/E-MTAB-513_lncRNA_coexp_3nn.tsv', 'result/GSE43520_lncRNA_coexp_3nn.tsv', 'result/GSE30352_lncRNA_coexp_3nn.tsv', 
    #               'result/GTEx_lncRNA_coexp_3nn.tsv']
    resultfiles = ['result/E-MTAB-513_lncRNA.tsv', 'result/GSE43520_lncRNA.tsv', 'result/GSE30352_lncRNA.tsv', 'result/GTEx_lncRNA.tsv']
    genecode_label_roc_list, genecode_score_roc_list = get_IBD_roc(resultfiles[0], diff_lnc, lnc2IBD)
    GSE43520_label_roc_list, GSE43520_score_roc_list = get_IBD_roc(resultfiles[1], diff_lnc, lnc2IBD)
    GSE30352_label_roc_list, GSE30352_score_roc_list = get_IBD_roc(resultfiles[2], diff_lnc, lnc2IBD)
    GTEx_label_roc_list, GTEx_score_roc_list = get_IBD_roc(resultfiles[3], diff_lnc, lnc2IBD)
    plt.close("all")
    Figure = plt.figure() 
    #pdb.set_trace()
    plot_roc_curve_lncRNA(genecode_label_roc_list, genecode_score_roc_list, datasource[0])
    plot_roc_curve_lncRNA(GSE43520_label_roc_list, GSE43520_score_roc_list, datasource[1])
    plot_roc_curve_lncRNA(GSE30352_label_roc_list, GSE30352_score_roc_list, datasource[2])
    plot_roc_curve_lncRNA(GTEx_label_roc_list, GTEx_score_roc_list, datasource[3])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    #plt.savefig(save_fig_dir + selected + '_' + class_type + '.png') 
    plt.show()  
    
def read_IBD_loci_lncRNA(gene_enst_ensg, ibd_file = 'data/disease/1040_lncRNAs_within_IBD_loci'):
    #gene_enst_ensg = map_enst_to_ensg()
    print ibd_file
    loci_lncRNA = set()
    with open(ibd_file, 'r') as fp:
        for line in fp:
            values = line.split()
            lncRNA = values[0]
            if gene_enst_ensg.has_key(lncRNA):
                loci_lncRNA.add(gene_enst_ensg[lncRNA])
    
    return loci_lncRNA
            
def read_ibd_diff_loci_lncRNA(ibd_file = 'data/disease/IBD_diff_loci_lncRNA'):
    diff_loci_lncRNA = set()
    with open(ibd_file, 'r') as fp:
        for line in fp:
            values = line.split()
            diff_loci_lncRNA.add(values[0])
    
    return diff_loci_lncRNA    

def read_predict_benchmark_master(master_file = 'master_files/E-MTAB-513.tsv'):
    disease_lncrna = {}
    with open(master_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            disease = values[2]
            gene = values[1]
            score = float(values[-2])
            disease_lncrna.setdefault(disease, []).append((score, gene))
        
    return disease_lncrna

def read_upreulated_lncrna(data = 'data/disease/Up_regulated_lncRNA_in_PC.txt'):
    regulated_lncrna = []
    with open(data, 'r') as fp:
        for line in fp:
            if line[0] == '#' or 'FDR' in line:
                continue
            values = line.rstrip().split()
            regulated_lncrna.append(values[0])
    
    return regulated_lncrna

def spec_case_study(predict_file = 'result/GTEx_lncRNA.tsv'):
    dis_doid = 'DOID:1909'
    gene_id = 'ENSG00000250961'
    master_scores = read_predict_lncrna(predict_file)# = 'result/E-MTAB-513_lncRNA.tsv')
    #master_scores = read_predict_benchmark_master()
    gene_scores = master_scores[dis_doid]
    gene_scores.sort(reverse=True)
    pdb.set_trace()
    
def case_study(predict_file):
    dis_target = 'PROSTATE CANCER'
    dis_doid = 'DOID:10283'
    #dis_doid = 'DOID:1909'
    master_scores = read_predict_lncrna(predict_file)# = 'result/E-MTAB-513_lncRNA.tsv')
    #master_scores = read_predict_benchmark_master()
    gene_scores = master_scores[dis_doid]
    gene_scores.sort(reverse=True)
    
    lnc2cancer = {}
    with open('data/disease/data_disease_doid.txt', 'r') as fp:
        for line in fp:
            values = line.rstrip().split('\t')
            disease = values[2].upper()
            gene = values[1]
            lnc2cancer.setdefault(disease, set([])).add(gene)

    disease_genes = lnc2cancer[dis_doid]
    all_lncrna = [val[1] for val in gene_scores]
    overlap_lnc = len(disease_genes & set(all_lncrna))
    print 'overlap lcnrna', overlap_lnc
    print 'total # of lcnRNA', len(gene_scores)
    top_200_lncRNA = [val[1] for val in gene_scores[:100]]
    print 'disease', len(disease_genes)
    #upregulated_lncrna = read_upreulated_lncrna()
    #downregulated_lncrna = read_upreulated_lncrna(data = 'data/disease/Down_regulated_lncRNA_in_PC.txt' )
    print top_200_lncRNA[:10]   
    for val in disease_genes:
        if val in top_200_lncRNA:
            print val, top_200_lncRNA.index(val) + 1
        #else:
        #    print 'not in top 200 associated lncRNA' 
       
    #plot_circ_figs(gene_scores)
#polar bar
def plot_circ_figs(scores):
    print 'plot polar bar'
    N = 20
    gene_scores = scores[:N]
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    colors = 10 * np.random.rand(N)
    radii = [val[0] for val in gene_scores]
    labels = [val[1] for val in gene_scores]
    #pdb.set_trace()
    width = 2*np.pi / N #* np.random.rand(N)
    
    ax = plt.subplot(111, projection='polar')
    bars = ax.bar(theta, radii, width=width, bottom=0.0)
    
    # Use custom colors and opacity
    for r, bar in zip(colors, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.5)
        
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_visible(False)
        
    def autolabel(rects):
        # attach some text labels
        index = 0
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            if index == 3:
                tex = ax.text(rect.get_x()+rect.get_width()/2., 1.1*height, label, fontsize=10,
                        ha='center', va='baseline', rotation=60)
            elif index == 2:
                tex = ax.text(rect.get_x()+rect.get_width()/2., 1.1*height, label, fontsize=10,
                        ha='center', va='baseline', rotation=30) 
            elif index == 6:
                tex = ax.text(rect.get_x()+rect.get_width()/2., 1.1*height, label, fontsize=10,
                        ha='left', va='baseline', rotation=110)  
            elif index <=2:
                tex = ax.text(rect.get_x()+rect.get_width()/2., 1.1*height, label, fontsize=10,
                        ha='center', va='center')                
            elif index >2 and index <=12:
                tex = ax.text(rect.get_x()+rect.get_width()/2., 1.1*height, label, fontsize=10,
                        ha='center', va='top')
            elif index == 12 or index == 13:
                tex = ax.text(rect.get_x()+rect.get_width()/2., 1.1*height, label, fontsize=10,
                        ha='right', va='bottom')
            elif index == 16 or index == 17:
                tex = ax.text(rect.get_x()+rect.get_width()/2., 1.1*height, label, fontsize=10,
                        ha='left', va='bottom')
            else:
                tex = ax.text(rect.get_x()+rect.get_width()/2., 1.1*height, label, fontsize=10,
                        ha='center', va='bottom')
            if index >3 and index < 6:
                tex.set_rotation('vertical')
            if index >13 and index < 16:
                tex.set_rotation('vertical')
            
            index = index + 1
    #for tick in ax.get_xticklabels():
    #    tick.set_rotation(45)
    autolabel(bars)
    plt.tight_layout()
    plt.show()    


def do_some_stas(input_file, RNAseq = True, data=0, ratio=5, confidence=3, use_mean = False, log2 = False):
    #pdb.set_trace()
    #print body2_map, confidence
    disease_gene_dict, disease_name_map, whole_disease_gene_dict = read_DISEASE_database(confidence=confidence)
    ensg_ensp_map = get_ENSP_ENSG_map()
    tissue =False
    if RNAseq:
        gene_type_dict,gene_name_ensg, gene_id_position = read_gencode_gene_type()
        if data == 0:
            whole_data, samples = read_human_RNAseq_expression(input_file, gene_name_ensg, log2 = log2) # for microarray expression data
        elif data ==1:
            whole_data, samples = read_evolutionary_expression_data(input_file, use_mean = use_mean, log2 = log2)
        elif data == 2:
            whole_data, samples = read_average_read_to_normalized_RPKM(input_file, use_mean = use_mean, log2 = log2)
        elif data == 3:
            whole_data, samples = read_gtex_expression(input_file, gene_type_dict)
        else:
            whole_data, samples = read_tissue_database()
            tissue = True   
    else:
        whole_data = read_normalized_series_file(input_file)    
    num_dis = 0    
    for key in disease_gene_dict:
        disease_associated_data = []
        labels = []
        if RNAseq:
            disease_associated_data, labels, atmp, mRNA_list = get_mRNA_lncRNA_expression_RNAseq_data(whole_data, disease_gene_dict[key], 
                                                                        ensg_ensp_map, gene_type_dict, tissue=tissue)
        else:
            disease_associated_data, labels, atmp, mRNA_list = get_mRNA_lncRNA_expression_microarray_data(whole_data, disease_gene_dict[key], 
                                                                                                          ensg_ensp_map)
        #pdb.set_trace()    
        posi_num = np.count_nonzero(labels)
        label_len = len(labels)
        if posi_num < 100:
            continue
        #if 5*posi_num > len(other_disease_mRNA):
        #    continue   
        num_dis = num_dis + 1
    
    print num_dis


def test_obo_parser():
    oboparser = obo_object()
    oboparser.read_obo_file()
    resu=oboparser.getAncestors('DOID:3459')
    print resu
                
if __name__ == '__main__':
    #read_DISEASE_database()
    """ 
        python -file=classify_gencode.py -file=data/GSE43520/Human_Normalized_RPKM_NonSS_LncRNA_ProteinCoding_MainDataset_NonStrandSpecific.txt
         -outfile=result/result_name -ratio=5 -conf=2 -type=lncRNA -data=1
        python classify_gencode.py -file=data/gencodev7/gene.matrix.csv -outfile=result/result_name -ratio=5 -conf=2 -type=lncRNA -data=0
        python classify_gencode.py -file=data/GSE30352/Human_Ensembl57_TopHat_UniqueReads.txt -outfile=result/test -ratio=5 -conf=2 -type=lncRNA -data=2
        
        mircroarry: python -file= classify_gencode.py data/GSE34894/GSE34894_series_matrix.txt.gz 
    """
    #class_type = "mRNA"
    #expression_file = args.file
    
    ratio = args.ratio
    class_type = args.type
    #class_type = sys.argv[3] if len(sys.argv) > 3 else "mRNA"
    confidence = args.conf
    #data = args.data
    #outfile = args.outfile #E-MTAB-62
    datasource ={0:'E-MTAB-513', 1:'GSE43520', 2:'GSE30352', 3:'GTEx'}
    filenames = {0:'data/gencodev7/genes.fpkm_table', 1:'data/GSE43520/genes.fpkm_table', 
                 2:'data/GSE30352/genes.fpkm_table', 3:'data/gtex/GTEx_Analysis_v6p_RNA-seq_RNA-SeQCv1.1.8_gene_median_rpkm.gct.gz'}
    #print expression_file, ratio, class_type, body2_map, confidence
    #express_dict = convert_average_read_to_normalized_RPKM(expression_file)
    #pdb.set_trace()
    #expression_file = sys.argv[1]
    random.seed(100)
    log2 = True
    expanison = True
    for data in [0, 1, 2, 3]:
    #for data in [3]:
        expression_file = filenames[data]
        if class_type == "mRNA":
            print '#validation on mRNA'
            outfile = 'result/' + datasource[data] + '_mRNA.tsv'
            construct_data_for_classifier_mRNA(expression_file, outfile, data=data, ratio=ratio, confidence=confidence, use_mean = False, log2 = log2) 
            print 'calculating average performance and plot ROC curve'
            result_dis_acc = calculate_average_performance_mRNA_crossvlidation(outfile)
            
        elif class_type == "lncRNA":
            print '#predict disease associated lncRNA using mRNA'
            outfile = 'result/' + datasource[data] + '_lncRNA.tsv'
            predict_for_lncRNA_using_mRNA(expression_file, outfile, data=data, ratio=ratio, confidence=confidence, use_mean = False, log2 = log2)
            plot_tissue_importance(outfile + '.imp')
            calculate_cc_tissue_imp(outfile + '.imp', dataset = datasource[data])
            print '#benchmarking predicted disease associated lncRNA using lncRNADisease database'
            benfile = 'master_files/' + datasource[data] + '.tsv'
            fw = open(benfile, 'w')
            benchmark_predict_disease_lncRNAs(outfile, fw, datasource[data], expanison = expanison)
            case_study(outfile)
            case_study_IBD(outfile)
            fw.close()
        elif class_type == 'coexpress':
            print '#predict disease associated lncRNA using coexpression'
            outfile = 'result/' + datasource[data] + '_lncRNA_coexp_1nn.tsv'
            predict_for_lncRNA_using_mRNA(expression_file, outfile, data=data, ratio=ratio, confidence=confidence, use_mean = False, coexp=True, log2 = log2)
            #plot_tissue_importance(outfile + '.imp')
            #print '#benchmarking predicted disease associated lncRNA using lncRNADisease database'
            #benfile = 'master_files/' + datasource[data] + '_coexp_3nn.tsv'
            #fw = open(benfile, 'w')
            #benchmark_predict_disease_lncRNAs(outfile, fw, datasource[data], expanison = expanison)      
            #fw.close()       
        elif class_type == 'knn':
            print '#predict disease associated lncRNA using knn coexpression'
            outfile = 'result/' + datasource[data] + '_lncRNA_knn.tsv' 
            #predict_for_lncRNA_using_mRNA(expression_file, outfile, data=data, ratio=ratio, confidence=confidence, use_mean = False, coexp=False, 
            #                              knn = True, log2 = log2)
            #plot_tissue_importance(outfile + '.imp')
            #print '#benchmarking predicted disease associated lncRNA using lncRNADisease database'
            benfile = 'master_files/' + datasource[data] + '_knn.tsv'
            fw = open(benfile, 'w')
            benchmark_predict_disease_lncRNAs(outfile, fw, datasource[data], expanison = expanison, discrete =True)      
            fw.close()             
        elif class_type == "benchmark":
            outfile = 'result/' + datasource[data] + '_lncRNA_knn.tsv'
            #predict_for_lncRNA_using_mRNA(expression_file, outfile, data=data, ratio=ratio, confidence=confidence, use_mean = True, log2 = log2)
            #plot_tissue_importance(outfile + '.imp')
            #calculate_cc_tissue_imp()
            print '#benchmarking predicted disease associated lncRNA using Golden-standard database'
            benfile = 'master_files/' + datasource[data] + '.tsv'
            fw = open(benfile, 'w')
            benchmark_predict_disease_lncRNAs(outfile, fw, datasource[data], expanison = expanison, discrete =True)
    
            fw.close()                
        elif class_type == "plot":
            print 'expression distribution fig'
            #do_some_stas(expression_file)
            #read_cancer_name()
            #remove_unreal_lncrna()
            #read_evolutionary_expression_data()
            #read_average_read_to_normalized_RPKM()
            #outfile = 'result/' + datasource[data] + '_mRNA.tsv'
            #print outfile
            #calculate_average_performance_mRNA_crossvlidation(outfile)
            #read_lnccancer()
            #merge_two_lncRNA_disease()
            #plot_expression_distribution(expression_file, data=data)
            #print 'plot venn figure'
            #plot_venn_figure()
            #plot_bar_overlap_fig()
            #break
            #compare_disease_overlap()
            #gene_type_dict, gene_name_ensg, gene_id_position = read_gencode_gene_type()
            #data, samples = read_gtex_expression()
            #pdb.set_trace()
            #spec_case_study()
            #read_DisGeNET_database()
            #print 'plot expression figure'
            #ENSG00000253364:RP11-731F5.2
            #plot_gene_express_in_tissue()
            #plot_tissue_importance('result/E-MTAB-513_lncRNA.tsv.imp')
            #print 'map lncRNAdisease database'
            #transfer_name_ensg_for_lncRNAdisease()
            #read_evolutionary_expression_data(filenames[1])
            #read_average_read_to_normalized_RPKM(filenames[2])
            #print 'lncRNA classification performance'
            #performance_accuracy_for_random(expanison = False)
            #performance_accuracy_for_random_multi_cancer_noncancer(expanison = True)
            #performance_accuracy_for_random_multi(expanison = True)
            #map_to_doid_disease()
            #outfile = 'result/' + datasource[data] + '_lncRNA.tsv'
            #plot_IBD_roc()
            #case_study(outfile)
            #plot_circ_figs()
            #compare_disease_overlap()
            case_study_IBD_venn()
            #case_study_IBD(outfile)
            #outfile = 'result/' + datasource[data] + '_lncRNA.tsv'
            #calculate_cc_tissue_imp(outfile + '.imp', dataset = datasource[data])
            #calculate_cc_tissue_imp(outfile)
            #plot_tissue_importance(outfile)
            #test_obo_parser()
            #print 'plot dis hist'
            ##get_mRNA_expression()
            #plot_expression_fig()

            break
        
    #print CUTOFF, TRAIN_NUM, SCALEUP, SCALEDOWN
    #performance_accuracy_for_random(expanison = False)
    #elif class_type == "combine": 
        #
    #    print 'combining different data sources'
        #pdb.set_trace()
    #    stringrnautils.combine_masterfiles(("E-MTAB-513.tsv", "GSE43520.tsv", "GSE30352.tsv", "GTEx.tsv"),
    #                                   'predictions.tsv', window_size=25, negative_evidence=False, rebenchmark_everything=True)
    print 'top candidates'
    #get_overlap_predcited_real()
    #read_cancer_lncRNA_database()
    #dis_dict = {}
    #get_all_snp_disease_assoc()
    #print 'disease tissue score'
    #disease_tissue_dct, tissues = get_disease_associated_matrix()
    "prostate cancer: 176807"
    #plot_disease_tissue_score('176807', disease_tissue_dct, tissues, "Prostate cancer")        
