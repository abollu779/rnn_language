import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
import time
import csv
import os
import nibabel
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage.filters import gaussian_filter
import scipy.io as sio

import sys
sys.path.insert(0,'/home/abollu/cneuromod/brain_prediction_pipeline')

from brain_language_ridge_tools import cross_val_ridge, corr
import time as tm
from scipy import signal

subject_runs = dict(F = [4,5,6,7],
        G = [3,4,5,6],
        H = [3,4,9,10],
        I = [7,8,9,10],
        J = [7,8,9,10],
        K = [7,8,9,10],
        L = [7,8,9,10],
        M = [7,8,9,10],
        N= [7,8,9,10])

surfaces = dict( F = 'fMRI_story_F',
        G = 'fMRI_story_G',
        H = 'fMRI_story_H',
        I = 'fMRI_story_I',
        J = 'fMRI_story_J',
        K = 'fMRI_story_K',
        L = 'fMRI_story_L',
        M = 'fMRI_story_M',
        N = 'fMRI_story_N')

transforms = dict( F = 'F_ars_auto2',
        G = 'G_ars_auto2',
        H = 'H_ars_auto2',
        I = 'I_ars_auto2',
        J = 'J_ars_auto2',
        K = 'K_ars_auto2',
        L = 'L_ars_auto2',
        M = 'M_ars_auto2',
        N = 'N_ars_auto2')

with open('/home/mktoneva/HP/brain_prediction_pipeline/utils/locations.txt', 'r') as f:
    locs = csv.reader(f,delimiter=',')
    loc306 = np.array([[float(w1[0].split(' ')[1]),float(w1[0].split(' ')[2])] for w1 in locs ])
loc102 = loc306[::3]
dists = euclidean_distances(loc102, loc306)
neighbors = np.argsort(dists,axis = 1)
neighbors = neighbors[:,:27]
sensor_groups = np.zeros((102,306))
for i in range(102):
    sensor_groups[i,neighbors[i]] = 1
    
def load_transpose_zscore(file): 
    dat = nibabel.load(file).get_data()
    dat = dat.T
    return zscore(dat,axis = 0)

def smooth_run_not_masked(data,smooth_factor):
    smoothed_data = np.zeros_like(data)
    for i,d in enumerate(data):
        smoothed_data[i] = gaussian_filter(data[i], sigma=smooth_factor, order=0, output=None,
                 mode='reflect', cval=0.0, truncate=4.0)
    return smoothed_data

def distributed_speaker_per_TR(data, oldtime, newtime):
    """Assigns speaker that said most words during each TR as the speaker label
    for that TR.
    """
    newdata = []
    oldtime_running_idx = 0
    num_multispeaker_TRs, num_singlespeaker_TRs, num_nospeech_TRs = 0, 0, 0
    for TR_end_time in newtime:
        TR_embedding = np.zeros(data.shape[1]) # data.shape[1]: number of speakers/characters
        while oldtime_running_idx < len(oldtime) and oldtime[oldtime_running_idx] < TR_end_time:
            TR_embedding[np.argmax(data[oldtime_running_idx])] += 1.
            oldtime_running_idx += 1
        
        num_speakers = len(np.nonzero(TR_embedding)[0])
        if num_speakers > 1:
            num_multispeaker_TRs += 1
        elif num_speakers == 1:
            num_singlespeaker_TRs += 1
        else:
            num_nospeech_TRs += 1
            
        TR_embedding /= np.sum(TR_embedding)
        newdata.append(TR_embedding)
    print("TR Data: Total={}, NoSpeech={}, SingleSpeaker={}, MultiSpeaker={}".format(len(newtime), num_nospeech_TRs, num_singlespeaker_TRs, num_multispeaker_TRs))
    return np.vstack(newdata)
    
import numexpr as ne
# from Alex Huth / Gallant Lab
def lanczosinterp2D(data, oldtime, newtime, window=3):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i).  A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.  If [causal], only past time points will be used
    to computed the present value, and future time points will be ignored.
    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the sinc interpolation filter.
    [window] lobes of the sinc function will be used.  [window] should be an integer.
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime))
    print ("Doing sinc interpolation with cutoff={} and {} lobes.".format(cutoff, window))
    ## Construct new signal ##
    # newdata = np.zeros((len(newtime), data.shape[1]))
    # for ndi in range(len(newtime)):
    #         for di in range(len(oldtime)):
    #             newdata[ndi,:] += sincfun(cutoff, newtime[ndi]-oldtime[di], window, causal) * data[di,:]
    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)
    ## Construct new signal by multiplying the sinc matrix by the data ##
    newdata = np.dot(sincmat, data)
    return newdata

def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = t * cutoff
    pi = np.pi
    #val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val = ne.evaluate("window * sin(pi*t) * sin(pi*t/window) / (pi**2 * t**2)")
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0
    return val

def load_and_process(file,start_trim = 20, end_trim = 15, do_detrend=True, smoothing_factor = 1,
                     do_zscore = True):
    dat = nibabel.load(file).get_data()
    # very important to transpose otherwise data and brain surface don't match
    dat = dat.T 
    #trimming
    if end_trim>0:
        dat = dat[start_trim:-end_trim]
    else: # to avoid empty error when end_trim = 0
        dat = dat[start_trim:]
    # detrending
    if do_detrend:
        dat = signal.detrend(np.nan_to_num(dat),axis =0)
    # smoothing
    if smoothing_factor>0:
        # need to zscore before smoothing
        dat = np.nan_to_num(zscore(dat))
        dat = smooth_run_not_masked(dat, smoothing_factor)
    # zscore
    if do_zscore:
        dat = np.nan_to_num(zscore(dat))
    return dat

def zscore_word_data(data):
    # zscores time over each time window, and returns a 2D data structure.
    # to zscore over all time windows, and not by time window, use function above
    n_words = data.shape[0]
    data = np.reshape(data,[n_words,-1])
    data = np.nan_to_num(zscore(data))
    return data

def delay_one(mat, d):
        # delays a matrix by a delay d. Positive d ==> row t has row t-d
    new_mat = np.zeros_like(mat)
    if d>0:
        new_mat[d:] = mat[:-d]
    elif d<0:
        new_mat[:d] = mat[-d:]
    else:
        new_mat = mat
    return new_mat

def delay_mat(mat, delays):
        # delays a matrix by a set of delays d.
        # a row t in the returned matrix has the concatenated:
        # row(t-delays[0],t-delays[1]...t-delays[last] )
    new_mat = np.concatenate([delay_one(mat, d) for d in delays],axis = -1)
    return new_mat

# train/test is the full NLP feature
# train/test_pca is the NLP feature reduced to 10 dimensions via PCA that has been fit on the training data
# feat_dir is the directory where the NLP features are stored
# train_indicator is an array of 0s and 1s indicating whether the word at this index is in the training set
def get_nlp_features_fixed_length(layer, seq_len, feat_type, feat_dir, train_indicator, SKIP_WORDS=20, END_WORDS=5176):
  
    if layer == -1 and feat_type == 'bert':
        all_layers_train = []
        all_layers_test = []
        for layer2 in range(13):
            loaded = np.load(feat_dir + feat_type + '_length_'+str(seq_len)+ '_layer_' + str(layer2) + '.npy')
            train = loaded[SKIP_WORDS:END_WORDS,:][train_indicator]
            test = loaded[SKIP_WORDS:END_WORDS,:][~train_indicator]
        
            pca = PCA(n_components=10, svd_solver='full')
            pca.fit(train)
            train_pca = pca.transform(train)
            test_pca = pca.transform(test)
    
            all_layers_train.append(train_pca)
            all_layers_test.append(test_pca)

        return train_pca,test_pca, np.hstack(all_layers_train), np.hstack(all_layers_test)

  
    loaded = np.load(feat_dir + feat_type + '_length_'+str(seq_len)+ '_layer_' + str(layer) + '.npy')
    if feat_type == 'elmo':
        train = loaded[SKIP_WORDS:END_WORDS,:][:,:512][train_indicator]   # only forward LSTM
        test = loaded[SKIP_WORDS:END_WORDS,:][:,:512][~train_indicator]   # only forward LSTM
    elif feat_type == 'bert':
        train = loaded[SKIP_WORDS:END_WORDS,:][train_indicator]
        test = loaded[SKIP_WORDS:END_WORDS,:][~train_indicator]
    else:
        print('Unrecognized NLP feature type {}. Available options elmo, bert'.format(feat_type))
    
    pca = PCA(n_components=10, svd_solver='full')
    pca.fit(train)
    train_pca = pca.transform(train)
    test_pca = pca.transform(test)

    return train, test, train_pca, test_pca 


# train_indicator is an array of 0s and 1s indicating whether the word at this index is in the training set
def load_features_to_regress_out(feat_name_list, train_indicator, feat_type='', feat_dir='',SKIP_WORDS=20, END_WORDS=5176):
        
    if len(feat_name_list) == 0:
        return [],[]
        
    regress_features_train = []
    regress_features_test = []
    
    print(feat_name_list)
    for feat_name in feat_name_list:
        
        feat_name_split = feat_name.split('-')
        
        if 'prev' in feat_name_split or 'back' in feat_name_split:
            delay = int(feat_name_split[1])
        elif 'next' in feat_name_split or 'fwd' in feat_name_split:
            delay = -int(feat_name_split[1])
        else:
            delay = 0
        
        if delay == 0:
            print('using delay of {}'.format(delay))
            feature_train, feature_test = load_features(feat_name_split, delay, train_indicator, feat_type, feat_dir)

            regress_features_train.append(feature_train)
            regress_features_test.append(feature_test)
        else:
            while abs(delay) > 0:
                print('using delay of {}'.format(delay))
        
                feature_train, feature_test = load_features(feat_name_split, delay, train_indicator, feat_type, feat_dir)

                regress_features_train.append(feature_train)
                regress_features_test.append(feature_test)  
                
                delay = delay - np.sign(delay)
                
                if 'back' in feat_name_split or 'fwd' in feat_name_split:   # only want the features at a particular delayed position, not all feats up to that point
                        break
                                        
    return np.hstack(regress_features_train), np.hstack(regress_features_test)


def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n/n_folds))
    for i in range(0,n_folds -1):
        ind[i*n_items:(i+1)*n_items] = i
    ind[(n_folds-1)*n_items:] = (n_folds-1)
    return ind    

def run_class_time_CV_crossval_ridge(data, predict_feat_dict, 
                   regress_feat_names_list=[], SKIP_WORDS = 5, END_WORDS = 5176,
                   delays = [0], detrend = True, do_correct = [], n_folds = 4, splay = [],
                   do_acc = True, frequency= 0, downsampled=1, seed=0):

    # name = subject name
    # features = NLP features for all words
    # SKIP_WORDS = how many words to skip from the beginning in case the features are not good there
    # END_WORDS = how many words to skip from the end in case the features are not good there
    # method = ridge method: plain, svd, kernel_ridge, kernel_ridge_svd, ridge_sk
    # THIS IS CURRENTLY OVERRIDEN BECAUSE THE CODE JUST CALLS ridge WITH A FIXED LAMBDA
    # lambdas = lambdas to try
    # delays = look at current word + which other words? 0 = current, -1 previous, +1 next.
    #          most common is [-2,-1,0,1,2]
    # detrend = remove mean of last 5 words? MAYBE THIS SHOULD BE FALSE WHEN LOOKING AT CONTEXT
    # do_correct = not used now
    # n_folds = number of CV folds
    # splay = only do the analysis on the words in this array
    # do_acc = run single subject classification

    # detrend
    predict_feat_type = predict_feat_dict['feat_type']
    nlp_feat_type = predict_feat_dict['nlp_feat_type']
    feat_dir = predict_feat_dict['nlp_feat_dir']
    layer = predict_feat_dict['layer']
    seq_len = predict_feat_dict['seq_len']


    n_words = data.shape[0]
    if detrend:
        running_mean = np.vstack([np.mean(np.mean(data[i-5:i,:,:],2),0) for i in range(5,n_words)])
        data[5:] = np.stack([(data[5:,:,i]-running_mean).T for i in range(data.shape[2])]).T
      
    n_words = data.shape[0]
    n_time = data.shape[2]
    n_sensor = data.shape[1]

    ind = CV_ind(n_words, n_folds=n_folds)

    corrs = np.zeros((n_folds, n_time))
    acc = np.zeros((n_folds, n_time))
    preds_d = np.zeros((data.shape[0],data.shape[1]*data.shape[2]))


    all_test_data = []

    for ind_num in range(n_folds):
        start_time = time.time()

        train_ind = ind!=ind_num
        test_ind = ind==ind_num
        
        if predict_feat_type == 'elmo' or predict_feat_type == 'bert':
                train_features,test_features,_,_ = get_nlp_features_fixed_length(layer, seq_len, nlp_feat_type, feat_dir, train_ind)
        else:
                train_features,test_features = load_features_to_regress_out([predict_feat_type], train_ind, nlp_feat_type, feat_dir)
       
        #regress_train_features, regress_test_features = load_features_to_regress_out(regress_feat_names_list, train_ind, nlp_feat_type, feat_dir) 
    

        # split data
        train_data = data[train_ind]
        test_data = data[test_ind]

        # normalize data
        train_data = zscore_word_data(train_data)
        test_data = zscore_word_data(test_data)

        all_test_data.append(test_data)

        train_features = np.nan_to_num(zscore(train_features)) 
        test_features = np.nan_to_num(zscore(test_features)) 
        
        # if regressing out features, do it now
        if len(regress_feat_names_list) > 0:

            regress_train_features, regress_test_features = load_features_to_regress_out(regress_feat_names_list, train_ind, nlp_feat_type, feat_dir)
            
            #regress_weights, _ = cross_val_ridge(regress_train_features,train_features, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)
            preds_test,preds_train, _ = cross_val_ridge(regress_train_features,train_features,regress_test_features, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)  
        
            #preds_train = np.dot(regress_train_features, regress_weights)
            #preds_test = np.dot(regress_test_features, regress_weights)
            
            
            train_features = np.reshape(train_features-preds_train, train_features.shape)
            test_features = np.reshape(test_features-preds_test, test_features.shape)
            print('done regressing out')
            #del regress_weights
        
        #weights = cross_val_ridge(train_features,train_data,n_splits = 10,
        #                              lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)[0]
        
        preds,_,_ = cross_val_ridge(train_features,train_data,test_features,n_splits = 10,lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)
        
        #preds =  np.dot(test_features, weights)
        corrs[ind_num,:] = corr(preds,test_data).reshape(n_sensor,n_time).mean(0)
        preds_d[test_ind] = preds
        n_pred = preds.shape[0]
        #del  weights

        print('CV fold ' + str(ind_num) + ' ' + str(time.time()-start_time))


    return corrs, preds_d, np.vstack(all_test_data)

  
def run_class_time_CV_fmri_crossval_ridge_neuromod(data, features, regress_features=[], method = 'kernel_ridge', 
                                          lambdas = np.array([0.1,1,10,100,1000]),
                                          detrend = False, n_folds = 12, skip=5, output_fold_weights_only=False):
        
        
    n_words = data.shape[0]
    n_voxels = data.shape[1]

    ind = CV_ind(n_words, n_folds=n_folds)
    
    if output_fold_weights_only:
        fold_weights = {}
    else:
        corrs = np.zeros((n_folds, n_voxels))
        all_test_data = []
        all_preds = []
        all_predictfeat_data = []
        all_regressfeat2predictfeat_preds = []
        all_regressfeat_data = []
        all_predictfeat2regressfeat_preds = []

    for ind_num in range(n_folds):
        train_ind = ind!=ind_num
        test_ind = ind==ind_num
        
        # split data
        train_data = data[train_ind]
        test_data = data[test_ind]
        
        train_features = features[train_ind]
        test_features = features[test_ind]

        # skip TRs between train and test data
        if ind_num == 0: # just remove from front end
            train_data = train_data[skip:,:]
            train_features = train_features[skip:,:]
        elif ind_num == n_folds-1: # just remove from back end
            train_data = train_data[:-skip,:]
            train_features = train_features[:-skip,:]
        else:
            test_data = test_data[skip:-skip,:]
            test_features = test_features[skip:-skip,:]

        # normalize data
        train_data = np.nan_to_num(zscore(np.nan_to_num(train_data)))
        test_data = np.nan_to_num(zscore(np.nan_to_num(test_data)))
        
        train_features = np.nan_to_num(zscore(train_features))
        test_features = np.nan_to_num(zscore(test_features))

        all_test_data.append(test_data)
        
        if len(regress_features) > 0:
            regress_train_features = regress_features[train_ind]
            regress_test_features = regress_features[test_ind]

            if ind_num == 0: # just remove from front end
                regress_train_features = regress_train_features[skip:,:]
            elif ind_num == n_folds-1: # just remove from back end
                regress_train_features = regress_train_features[:-skip,:]
            else:
                regress_test_features = regress_test_features[skip:-skip,:]

            regress_weights, _ = cross_val_ridge(regress_train_features,train_features, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)
            preds_train = np.dot(regress_train_features, regress_weights)
            preds_test = np.dot(regress_test_features, regress_weights)

            # Example: Regressing out speaker information from ELMo embeddings. 
            # We try to predict ELMo embedding using speaker identity. Therefore, it makes sense that:
            # test_data: test_features (ELMo embeddings)
            # predictions: preds_test
            all_predictfeat_data.append(test_features)
            all_regressfeat2predictfeat_preds.append(preds_test)

            # According to above example, this would try to predict speaker identity using ELMo embedding.
            backwards_regress_weights, _ = cross_val_ridge(train_features, regress_train_features, n_splits=10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)
            backwards_preds_test = np.dot(test_features, backwards_regress_weights)
            all_regressfeat_data.append(regress_test_features)
            all_predictfeat2regressfeat_preds.append(backwards_preds_test)

            train_features = np.reshape(train_features-preds_train, train_features.shape)
            test_features = np.reshape(test_features-preds_test, test_features.shape)
            print('done regressing out')
            del regress_weights
    

        start_time = tm.time()
        weights, _ = cross_val_ridge(train_features,train_data, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'plain',do_plot = False)
        if output_fold_weights_only:
            fold_weights[ind_num] = weights
        else:
            preds = np.dot(test_features, weights)
            corrs[ind_num,:] = corr(preds,test_data)
            all_preds.append(preds)
            del weights
        print('fold {} completed, took {} seconds'.format(ind_num, tm.time()-start_time))
    
    if output_fold_weights_only:
        output = {'fold_weights_t':fold_weights}
    else:
        if len(regress_features)>0:
            all_predictfeat_data = np.vstack(all_predictfeat_data)
            all_regressfeat2predictfeat_preds = np.vstack(all_regressfeat2predictfeat_preds)
            all_regressfeat_data = np.vstack(all_regressfeat_data)
            all_predictfeat2regressfeat_preds = np.vstack(all_predictfeat2regressfeat_preds)
        output = {'corrs_t': corrs, 'preds_t': np.vstack(all_preds), 'test_t':np.vstack(all_test_data), 'regressfeat2predictfeat_preds_t':all_regressfeat2predictfeat_preds, 'predictfeat_test_t':all_predictfeat_data, 'predictfeat2regressfeat_preds_t':all_predictfeat2regressfeat_preds, 'regressfeat_test_t':all_regressfeat_data}
    return output


def run_class_time_CV_fmri_crossval_ridge(data, predict_feat_dict,
                                          regress_feat_names_list = [],method = 'kernel_ridge', 
                                          lambdas = np.array([0.1,1,10,100,1000]),
                                          detrend = False, n_folds = 4, skip=5):
    
    predict_feat_type = predict_feat_dict['feat_type']
    nlp_feat_type = predict_feat_dict['nlp_feat_type']
    feat_dir = predict_feat_dict['nlp_feat_dir']
    layer = predict_feat_dict['layer']
    seq_len = predict_feat_dict['seq_len']
        
        
    n_words = data.shape[0]
    n_voxels = data.shape[1]
        
    print(n_words)

    ind = CV_ind(n_words, n_folds=n_folds)

    corrs = np.zeros((n_folds, n_voxels))
    acc = np.zeros((n_folds, n_voxels))
    acc_std = np.zeros((n_folds, n_voxels))
    preds_d = np.zeros((data.shape[0], data.shape[1]))

    all_test_data = []
    
    
    for ind_num in range(n_folds):
        train_ind = ind!=ind_num
        test_ind = ind==ind_num
        
        word_CV_ind = TR_to_word_CV_ind(train_ind)
        if nlp_feat_type == 'brain': 
                word_CV_ind = train_ind
        
        if predict_feat_type == 'elmo' or predict_feat_type == 'bert':
                tmp_train_features,tmp_test_features,_,_ = get_nlp_features_fixed_length(layer, seq_len, nlp_feat_type, feat_dir, word_CV_ind)
        else:
                tmp_train_features,tmp_test_features = load_features_to_regress_out(predict_feat_type.split('+'), word_CV_ind, nlp_feat_type, feat_dir)
        
        if nlp_feat_type != 'brain': 
            train_features,test_features = prepare_fmri_features(tmp_train_features, tmp_test_features, word_CV_ind, train_ind)
        else: # no need to concatenate multiple TRs for brain to brain predictions
            train_features = tmp_train_features
            test_features = tmp_test_features
        
        if len(regress_feat_names_list) > 0:
            tmp_regress_train_features, tmp_regress_test_features = load_features_to_regress_out(regress_feat_names_list, word_CV_ind, nlp_feat_type, feat_dir) 
            
            if nlp_feat_type != 'brain':     
                regress_train_features,regress_test_features = prepare_fmri_features(tmp_regress_train_features, tmp_regress_test_features, word_CV_ind, train_ind)
            else:
                regress_train_features = tmp_regress_train_features
                regress_test_features = tmp_regress_test_features
                
        # split data
        train_data = data[train_ind]
        test_data = data[test_ind]

        # skip TRs between train and test data
        if ind_num == 0: # just remove from front end
            train_data = train_data[skip:,:]
            train_features = train_features[skip:,:]
        elif ind_num == n_folds-1: # just remove from back end
            train_data = train_data[:-skip,:]
            train_features = train_features[:-skip,:]
        else:
            test_data = test_data[skip:-skip,:]
            test_features = test_features[skip:-skip,:]

        # normalize data
        train_data = np.nan_to_num(zscore(np.nan_to_num(train_data)))
        test_data = np.nan_to_num(zscore(np.nan_to_num(test_data)))
        all_test_data.append(test_data)
        
        train_features = np.nan_to_num(zscore(train_features))
        test_features = np.nan_to_num(zscore(test_features)) 
        
        # if regressing out features, do it now
        if len(regress_feat_names_list) > 0:
            
            # skip TRs between train and test data
            if ind_num == 0: # just remove from front end
                regress_train_features = regress_train_features[skip:,:]
            elif ind_num == n_folds-1: # just remove from back end
                regress_train_features = regress_train_features[:-skip,:]
            else:
                regress_train_features = regress_train_features[skip:-skip,:]
            
            preds_test,preds_train, _ = cross_val_ridge(regress_train_features,train_features,regress_test_features, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)
            #regress_weights, _ = cross_val_ridge(regress_train_features,train_features, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)
            #preds_train = np.dot(regress_train_features, regress_weights)
            #preds_test = np.dot(regress_test_features, regress_weights)
            
            
            train_features = np.reshape(train_features-preds_train, train_features.shape)
            test_features = np.reshape(test_features-preds_test, test_features.shape)
            print('done regressing out')
            #del regress_weights
        
        start_time = tm.time()

        preds,_,_ = cross_val_ridge(train_features,train_data,test_features, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)

        #preds =  np.dot(test_features, weights)
        corrs[ind_num,:] = corr(preds,test_data)
        preds_d[test_ind] = preds
            

        print('fold {} completed, took {} seconds'.format(ind_num, tm.time()-start_time))
        #del weights


    return corrs, acc, acc_std, preds_d, np.vstack(all_test_data)



def binary_classify(Ypred, Y, n_class=20, nSample = 1000, pair_samples = []):
    # Ypred, Y predicted and real data with dims: [nsensors , nwords , ndims per word ]
    # n_class = how many words to classify at once
    # nSample = how many words to classify
    acc = np.zeros((nSample,Y.shape[-1]))
    test_word_inds = []
    if len(pair_samples)>0:
        Ypred2 = Ypred[:,pair_samples>=0]
        Y2 = Y[:,pair_samples>=0]
        pair_samples2 = pair_samples[pair_samples>=0]
    else:
        Ypred2 = Ypred
        Y2 = Y
        pair_samples2 = pair_samples
    n = Y2.shape[1]
    for iS in range(nSample):
        idx_real = np.random.choice(n, n_class)
        sample_real = Y2[:,idx_real]
        sample_pred_correct = Ypred2[:,idx_real]
        if len(pair_samples2) == 0:
            idx_wrong = np.random.choice(n, n_class)
        else:
            idx_wrong = sample_same_but_different(idx_real,pair_samples2)
        sample_pred_incorrect = Ypred2[:,idx_wrong]
        dist_correct = np.sum(np.sum((sample_real - sample_pred_correct)**2,0),0)
        dist_incorrect = np.sum(np.sum((sample_real - sample_pred_incorrect)**2,0),0)

        acc[iS] = (dist_correct < dist_incorrect)*1.0  + (dist_correct == dist_incorrect)*0.5

        test_word_inds.append(idx_real)
    return acc.mean(0), acc.std(0), acc, np.array(test_word_inds)

def binary_classify_neighborhoods(Ypred, Y, n_class=20, nSample = 1000,pair_samples = [],neighborhoods=[]):
    # Ypred, Y predicted and real data with dims: [nsubjects X nsensors , nwords , ndims per word ]
    # n_class = how many words to classify at once
    # nSample = how many words to classify

    voxels = Y.shape[-1]
    neighborhoods = np.asarray(neighborhoods, dtype=int)

    import time as tm

    acc = np.full([nSample, Y.shape[-1]], np.nan)
    acc2 = np.full([nSample, Y.shape[-1]], np.nan)
    test_word_inds = []

    if len(pair_samples)>0:
        Ypred2 = Ypred[pair_samples>=0]
        Y2 = Y[pair_samples>=0]
        pair_samples2 = pair_samples[pair_samples>=0]
    else:
        Ypred2 = Ypred
        Y2 = Y
        pair_samples2 = pair_samples
    n = Y2.shape[0]
    start_time = tm.time()
    for idx in range(nSample):
        
        idx_real = np.random.choice(n, n_class)

        sample_real = Y2[idx_real]
        sample_pred_correct = Ypred2[idx_real]

        if len(pair_samples2) == 0:
            idx_wrong = np.random.choice(n, n_class)
        else:
            idx_wrong = sample_same_but_different(idx_real,pair_samples2)
        sample_pred_incorrect = Ypred2[idx_wrong]

        #print(sample_pred_incorrect.shape)

        # compute distances within neighborhood
        dist_correct = np.sum((sample_real - sample_pred_correct)**2,0)
        dist_incorrect = np.sum((sample_real - sample_pred_incorrect)**2,0)

        neighborhood_dist_correct = np.array([np.sum(dist_correct[neighborhoods[v,neighborhoods[v,:]>-1]]) for v in range(voxels)])
        neighborhood_dist_incorrect = np.array([np.sum(dist_incorrect[neighborhoods[v,neighborhoods[v,:]>-1]]) for v in range(voxels)])


        acc[idx,:] = (neighborhood_dist_correct < neighborhood_dist_incorrect)*1.0 + (neighborhood_dist_correct == neighborhood_dist_incorrect)*0.5
        

        test_word_inds.append(idx_real)
    print(idx, tm.time()-start_time)
    return np.nanmean(acc,0), np.nanstd(acc,0), acc, np.array(test_word_inds)


def classify_predictions(preds_t, test_t, n_sensor, n_time, sensor_groups, pair_samples=[], n_class=20):

    acc_sub_time = np.zeros(n_time)
    acc_std_sub_time = np.zeros(n_time)
    
    n = preds_t.shape[0]
    np.random.seed(100)
    acc_sub_time, acc_std_sub_time, _,_ = binary_classify(preds_t.reshape([n,n_sensor,n_time]).transpose([1,0,2]),
                                                                        test_t.reshape([n,n_sensor,n_time]).transpose([1,0,2]),
                                                                        pair_samples = pair_samples, n_class=n_class)



    if len(sensor_groups)>0:
        n_sensor_group = sensor_groups.shape[0]
        acc_sub_time_sg = np.zeros((n_sensor_group, n_time))
        acc_std_sub_time_sg = np.zeros((n_sensor_group, n_time))

        for ig in range(n_sensor_group):
            tmp_test_t = test_t.reshape([n,n_sensor,n_time]).transpose([1,0,2])[sensor_groups[ig]==1]
            tmp_pred =  preds_t.reshape([n,n_sensor,n_time]).transpose([1,0,2])[sensor_groups[ig]==1]
            np.random.seed(100)
            acc_sub_time_sg[ig], acc_std_sub_time_sg[ig],_,_ = binary_classify(tmp_pred,tmp_test_t,
                                                                                            pair_samples = pair_samples, n_class=20)
    else:
        acc_sub_time_sg = []
        acc_std_sub_time_sg = []

    return acc_sub_time, acc_std_sub_time, acc_sub_time_sg , acc_std_sub_time_sg

