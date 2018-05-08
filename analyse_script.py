
# coding: utf-8

# In[1]:


import os
import glob
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


# In[50]:


flatten = lambda l: [item for sublist in l for item in sublist]


def return_dict_data(folder):
    data_dict = dict()
    i = 0
    for name in os.listdir(folder):
        if name.endswith('.png') or name.endswith('.jpg') and not name.startswith('PRC') and not name.startswith('ROC'):
            filepath = os.path.join(folder, name)
            img = load_img(filepath, grayscale=True)
            img_array = img_to_array(img)
            img_list = img_array[:,:,0].tolist()
            img_flat_list = flatten(img_list)
        
            data_dict['image%s' %i] = img_flat_list
            i= i+1
    return data_dict
    
    
def normalize_to_df(dictionary):
    return pd.DataFrame.from_dict(dictionary)/255.0


def compute_f1_score(df_mask, df_result, path, threshold=0.5):
    i=0
    f1_scores = dict()
    for name in df_mask.columns:
        y_true = df_mask[name].values.astype(int)
        scores = df_result[name].values
        y_pred = (scores>threshold).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        f1_scores[str(i)] = f1
        i=i+1
    
    return pd.DataFrame(data=[f1_scores], index=[path])

def compute_rand_index_adjusted(df_mask, df_result, path):
    i=0
    r_index_adj = dict()
    for name in df_mask.columns:
        y_true = df_mask[name].values.astype(int)
        ypred = df_result[name].values
        ria = adjusted_rand_score(y_true, ypred)
        r_index_adj[str(i)] = ria
        i=i+1
    return pd.DataFrame(data=[r_index_adj], index=[path])

def compute_precision_recal(df_mask, df_result, path):
    i=0
    precision_df = pd.DataFrame()
    recall_df = pd.DataFrame()
    for name in df_result.columns:
        y = df_mask[name].values.astype(int)
        scores = df_result[name].values
        precision, recall, thresholds = precision_recall_curve(y, scores, pos_label=1)
        precision_df = pd.concat([precision_df,pd.DataFrame({str(i):precision.tolist()})], axis=1)
        recall_df = pd.concat([recall_df,pd.DataFrame({str(i):recall.tolist()})],axis=1) 
        i=i+1
    precision_df.to_csv(path_or_buf=path+'\\precision.csv')
    recall_df.to_csv(path_or_buf=path+'\\recall.csv')
    return precision_df, recall_df

def compute_ROC(df_mask, df_result, path):
    i=0
    fpr_df = pd.DataFrame()
    tpr_df = pd.DataFrame()
    
    for name in df_result.columns:
        y = df_mask[name].values.astype(int)
        scores = df_result[name].values
        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        fpr_df = pd.concat([fpr_df, pd.DataFrame({str(i):fpr.tolist()})], axis=1)
        tpr_df = pd.concat([tpr_df, pd.DataFrame({str(i):tpr.tolist()})], axis=1) 
        i=i+1
    fpr_df.to_csv(path_or_buf=path+'\\FPR.csv')
    tpr_df.to_csv(path_or_buf=path+'\\TPR.csv')
    return fpr_df, tpr_df


def plot_curve_PRC(precision_df, recal_df, path):
    
    plt.figure(1, figsize=(16,8))
    
    for name in precision_df.columns:
        plt.plot(recal_df[name], precision_df[name], linewidth=6)

    plt.plot([0, 1], [1, 0], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc='best')
    plt.legend(['Image 0', 'Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'], loc='best')
    plt.rc('font', size=30)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    
    plt.savefig(path+'\\PRC.png')
    plt.close()
    return

def plot_curve_ROC(fpr_df, tpr_df, path):
    
    plt.figure(2, figsize=(16,8))
    
    for name in fpr_df.columns:
        plt.plot(fpr_df[name], tpr_df[name], linewidth=6)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('1 - Specificity (False positive rate)')
    plt.ylabel('Sensitivity (True positive rate)')
    plt.title('Receiver operating characteristic')
    plt.legend(['Image 0', 'Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'], loc='best')
    plt.rc('font', size=30)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.savefig(path+'\\ROC.png')
    plt.close()
    return

def print_score(df_score, path):
    mean_f1 = df_score.values.mean()
    std2_f1 = 2*df_score.values.std()

    mean_f1 = ("%.2f" % mean_f1)
    std2_f1 = ("%.2f" % std2_f1)
    print(path + ' : ' + str(mean_f1) + ' +/- ' + str(std2_f1))
    
def compute_model_confussion_matrix(df_mask, df_result, path):
    tn_result = 0
    fp_result = 0
    fn_result = 0
    tp_result = 0
    
    for name in df_result.columns:
        array = confusion_matrix(mask_df[name].values.round().astype(int),
                                 prediction_df[name].values.round().astype(int))
        tn_result = tn_result + array[0,0]
        fp_result = fp_result + array[0,1]
        fn_result = fn_result + array[1,0]
        tp_result = tp_result + array[1,1]
         
    results = {'back_back':tn_result, 'back_sample': fp_result, 'sample_back': fn_result, 'sample_sample':tp_result}
    df = pd.DataFrame(data=[results])
    df.to_csv(path_or_buf=path+'\\Confusion_matrix.csv')
    return df

def plot_normalized_confusin_matrix(results, path):
    reshaped = results.values.reshape((2, 2))/1572864
    
    plt.figure(1, figsize=(8, 12))
    plt.imshow(reshaped, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    tick_marks = 2
    plt.xticks([1, 0], ('sample', 'background'))
    plt.yticks([1, 0], ('sample', 'background'))

    fmt = '.2f'
    thresh = reshaped.max() / 2.
    for i, j in itertools.product(range(reshaped.shape[0]), range(reshaped.shape[1])):
        plt.text(j, i, format(reshaped[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if reshaped[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig(path+'\\Confusion_matrix.png')
    plt.close()


# In[51]:


# directory paths

list_result_dir = glob.glob('*unet*')
original_mask_dir = glob.glob('OriginalMasks')


# In[ ]:


def compute_ROC_total(df_mask, df_result):
    i=0
    fpr_df = pd.DataFrame()
    tpr_df = pd.DataFrame()
    
    mask_append = pd.DataFrame()
    result_append = pd.DataFrame()
    for name in df_result.columns:
        mask_append = pd.concat([mask_append, df_mask[name]], ignore_index=True)
        result_append = pd.concat([result_append, df_result[name]], ignore_index=True)
        fpr, tpr, thresholds = metrics.roc_curve(mask_append.values.astype(int), result_append.values, pos_label=1)
    return fpr, tpr
    
def compute_PRC_total(df_mask, df_result):
    i=0
    fpr_df = pd.DataFrame()
    tpr_df = pd.DataFrame()
    
    mask_append = pd.DataFrame()
    result_append = pd.DataFrame()
    for name in df_result.columns:
        mask_append = pd.concat([mask_append, df_mask[name].astype(int)], ignore_index=True)
        result_append = pd.concat([result_append, df_result[name]], ignore_index=True)
        precision, recall, thresholds = precision_recall_curve(mask_append, result_append, pos_label=1)
    return precision, recall



# In[ ]:


# best out of best ROC curves development

# directory paths

list_result_dir = ['RZD1000unet0', 'RZD5000unet7','RDZ10000ITER10unet4','RDZ10000ITER15unet3', 'RDZ10000ITER20unet4']
labels = ['RDZ1000unet0', 'RDZ5000unet7','RDZ10000ITER10unet4','RDZ10000ITER15unet3', 'RDZ10000ITER20unet4']
mask_dict = return_dict_data('OriginalMasks')
mask_df = normalize_to_df(mask_dict)
i=1
Ria_all = pd.DataFrame()
F1_all = pd.DataFrame()

plt.figure(1, figsize=(16,8))


for path in list_result_dir:
    prediction_dict = return_dict_data(path)
    prediction_df = normalize_to_df(prediction_dict)
    
    fpr, tpr = compute_ROC_total(mask_df, prediction_df)
    plt.plot(fpr, tpr, linewidth=6)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('1 - Specificity (False positive rate)')
plt.ylabel('Sensitivity (True positive rate)')
plt.title('Receiver operating characteristic')
plt.legend(labels, loc='best')
plt.rc('font', size=30)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.savefig('C:\\Users\\smocko\\Desktop\\FinalResults\\ROC_compare.png')
#plt.show()
plt.close()

    


# In[ ]:


# best out of best PRC curves development

# directory paths

list_result_dir = ['RZD1000unet0', 'RZD5000unet7','RDZ10000ITER10unet4','RDZ10000ITER15unet3', 'RDZ10000ITER20unet4']
labels = ['RDZ1000unet0', 'RDZ5000unet7','RDZ10000ITER10unet4','RDZ10000ITER15unet3', 'RDZ10000ITER20unet4']
mask_dict = return_dict_data('OriginalMasks')
mask_df = normalize_to_df(mask_dict)
i=1
Ria_all = pd.DataFrame()
F1_all = pd.DataFrame()

plt.figure(1, figsize=(16,8))

for path in list_result_dir:
    prediction_dict = return_dict_data(path)
    prediction_df = normalize_to_df(prediction_dict)
    
    precision, recall = compute_PRC_total(mask_df, prediction_df)
    plt.plot(recall, precision, linewidth=6)

plt.plot([0, 1], [1, 0], 'k--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall')
plt.legend(loc='best')
plt.legend(labels, loc='best')
plt.rc('font', size=30)
plt.xlim((0, 1))
plt.ylim((0, 1))

plt.savefig('C:\\Users\\smocko\\Desktop\\FinalResults\\PRC_compare.png')
#plt.show()
plt.close()


# In[17]:


# directory paths

list_result_dir = glob.glob('*unet*')
original_mask_dir = glob.glob('OriginalMasks')


# In[52]:


mask_dict = return_dict_data('OriginalMasks')
mask_df = normalize_to_df(mask_dict)
i=1
Ria_all = pd.DataFrame()
F1_all = pd.DataFrame()


for path in list_result_dir:
    prediction_dict = return_dict_data(path)
    prediction_df = normalize_to_df(prediction_dict)
    
    RIA_df = compute_rand_index_adjusted(mask_df, prediction_df, path)
    RIA_df.to_csv(path_or_buf=path+'\\RIA_score.csv')
    
    Ria_all = Ria_all.append(RIA_df)
        
    f1_score_df = compute_f1_score(mask_df, prediction_df,path, threshold=0.8)
    f1_score_df.to_csv(path_or_buf=path+'\\f1_score.csv')
    
    F1_all = F1_all.append(f1_score_df)
    
    confussion_matrix = compute_model_confussion_matrix(mask_df, prediction_df, path)
    plot_normalized_confusin_matrix(confussion_matrix, path)
    
    precision_df, recall_df = compute_precision_recal(mask_df, prediction_df, path)
    plot_curve_PRC(precision_df, recall_df, path)
    
    fpr_df, tpr_df = compute_ROC(mask_df, prediction_df, path)
    plot_curve_ROC(fpr_df, tpr_df, path)
    
    del prediction_df
    
    print(str(round(i*2.37))+' %', end = '\r')
    i=i+1


# In[ ]:


#def plot_curve_ROC(fpr_df, tpr_df, path):
#    
#    plt.figure(1, figsize=(16,8))
#    
#    for name in fpr_df.columns:
#        plt.plot(fpr_df[name], tpr_df[name], linewidth=6)
#
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.xlabel('1 - Specificity (False positive rate)')
#    plt.ylabel('Sensitivity (True positive rate)')
#    plt.title('Receiver operating characteristic')
#    plt.legend(['Image 0', 'Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'], loc='best')
#    plt.rc('font', size=30)
#    plt.xlim((0, 1))
#    plt.ylim((0, 1))
#    plt.savefig(path+'\\ROC.jpg')
#    plt.close()
#    return


# In[ ]:


#plt.figure(1, figsize=(16,8))
    
#for name in fpr_df.columns:
#    plt.plot(fpr_df[name], tpr_df[name], linewidth=6)

#plt.plot([0, 1], [0, 1], 'k--')
#plt.xlabel('1 - Specificity (False positive rate)')
#plt.ylabel('Sensitivity (True positive rate)')
#plt.title('Receiver operating characteristic')
#plt.legend(['Image 0', 'Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'], loc='best')
#plt.rc('font', size=30)
#plt.xlim((0, 1))
#plt.ylim((0, 1))
#plt.savefig(path+'\\ROC.jpg')
#plt.show()
#plt.close()


# In[ ]:


unet10ki10 = Ria_all[0:10].values.tolist()
unet10ki15 = Ria_all[10:15].values.tolist()
unet10ki20 = Ria_all[15:20].values.tolist()
unet1k = Ria_all[21:31].values.tolist()
unet5k = Ria_all[31:41].values.tolist()

plt.figure(1, figsize=(16,8))

plt.ylabel('ARI')
plt.title('Boxplots Adjusted Rand Index')
labels = ['RDZ1K', 'RDZ5K', 'RDZ10KI10', 'RDZ10KI15', 'RDZ10KI20']
plt.rc('font', size=30)
plt.ylim((0, 1))
plt.boxplot([unet1k, unet5k, unet10ki10, unet10ki15, unet10ki20], labels=labels)
plt.savefig('C:\\Users\\smocko\\Desktop\\FinalResults\\RI_boxplots.png')
plt.close()


# In[ ]:


unet10ki10 = F1_all[0:10].values.tolist()
unet10ki15 = F1_all[10:15].values.tolist()
unet10ki20 = F1_all[15:20].values.tolist()
unet1k = F1_all[21:31].values.tolist()
unet5k = F1_all[31:41].values.tolist()

plt.figure(1, figsize=(16,8))

plt.ylabel('DSC')
plt.title('Boxplots Dice Similarity coefficient')
labels = ['RDZ1K', 'RDZ5K', 'RDZ10KI10', 'RDZ10KI15', 'RDZ10KI20']
plt.rc('font', size=30)
plt.ylim((0, 1))
plt.boxplot([unet1k, unet5k, unet10ki10, unet10ki15, unet10ki20], labels=labels)
plt.savefig('C:\\Users\\smocko\\Desktop\\FinalResults\\Dice_boxplots.png')
plt.close()


# In[ ]:


# Edge based detection - canny detector
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import sobel
import imageio

files = ['2-9.png', '3-1.png', '5.png',
         '9-1.png', '19.png', '21-2.png']

for file in files:
    # load image
    filepath = os.path.join(file)
    img = load_img(filepath, grayscale=True)
    img_array = img_to_array(img)
    a = img_array[:,:,0]
    
    # edge detection
    edges = canny(a)
    fill_edges = ndi.binary_fill_holes(edges)
    edges_cleaned = morphology.remove_small_objects(fill_edges, 21).astype('uint8')
    edges_cleaned[edges_cleaned == 1] = 255
    filename = 'C:\\Users\\smocko\\Desktop\\FinalResults\\Canny'+file
    imageio.imwrite(filename, edges_cleaned)
    
    # region detection
    elevation_map = sobel(edges)
    markers = np.zeros_like(a)
    markers[a < 30] = 1
    markers[a > 150] = 2
    segmentation = morphology.watershed(elevation_map, markers).astype('uint8')
    segmentation[segmentation == 1] = 255
    filename = 'C:\\Users\\smocko\\Desktop\\FinalResults\\Sobel'+file
    imageio.imwrite(filename, segmentation)
    

