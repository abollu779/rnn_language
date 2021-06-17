###########
# IMPORTS #
###########
import argparse
import numpy as np

from utils.utils import delay_mat, delay_one, zscore_word_data, collect_fold_weights_class_time_CV_fmri_crossval_ridge_neuromod
from utils.utils import subject_runs, surfaces, transforms, load_transpose_zscore, smooth_run_not_masked
from utils.utils import sensor_groups, classify_predictions, binary_classify_neighborhoods, CV_ind
from utils.utils import load_and_process, lanczosinterp2D
from utils.ridge_tools import corr, R2

from sklearn.decomposition import PCA
from scipy.stats import zscore
import scipy.io as sio

import time as tm
from scipy import signal
import pickle as pk
import nibabel
from scipy import signal
import cortex

import os
from os import path

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

####################
# GLOBAL VARIABLES #
####################
SKIP_WORDS = 20
END_WORDS = 5176
SMOOTHING = 0
START_TRIM_NUM = 20
END_TRIM_NUM = 15

figures_sessions_run2 = {
    'sub-01': ['ses-011', 'ses-012'],
    'sub-02': ['ses-010', 'ses-011', 'ses-012'],
    'sub-03': ['ses-012', 'ses-013'],
    'sub-04': ['ses-010', 'ses-011', 'ses-012'],
    'sub-05': ['ses-010'],
    'sub-06': ['ses-006', 'ses-007', 'ses-008'],
}

figures_sessions_run1 = {
    'sub-01': ['ses-007', 'ses-008', 'ses-009'],
    'sub-02': ['ses-002', 'ses-003'],
    'sub-03': ['ses-007', 'ses-009', 'ses-010'],
    'sub-04': ['ses-004', 'ses-005', 'ses-006'],
    'sub-05': ['ses-002','ses-003'],
    'sub-06': ['ses-002', 'ses-003'],
}
 
repetitions = {1:figures_sessions_run1, 2:figures_sessions_run2}

#############################
# COLLECT DATA AND FEATURES #
#############################
def save_data(args):
    if args.regress_feat_types != '0':
        regress_feat_names_list = np.sort(args.regress_feat_types.split('+'))
    else:
        regress_feat_names_list = []

    # Load fMRI data
    func_files = []
    for subj_run in range(1,13):
        if subj_run < 10:
            run = '0' + str(subj_run)
        else:
            run = subj_run

        for session in repetitions[args.repetition]['sub-{}'.format(args.subject)]:
            datadir = '/share/volume0/cneuromod_data/movie10/derivatives/fmriprep-20.1.0/fmriprep/sub-{}/{}/func/'.format(args.subject, session)

            fname = '{}sub-{}_{}_task-figures{}_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(datadir, args.subject, session, run, args.repetition)
            if path.exists(fname):
                func_files.append(fname)
                print("Run {}, Session {}".format(subj_run, session))
    
    print(len(func_files))
    data = []
    for i, f in enumerate(func_files):
        curr_data = load_and_process(f, start_trim=START_TRIM_NUM, end_trim=END_TRIM_NUM,
                                do_detrend=True, smoothing_factor=SMOOTHING, do_zscore=True)
        data.append(curr_data)
        print(curr_data.shape)
    
    import time
    start_vstack = time.time()
    data = np.vstack(data)
    end_vstack = time.time()
    print("Time taken for vstack = {}s".format(end_vstack - start_vstack))
    print(data.shape)
    # data = np.vstack([load_and_process(file, start_trim = START_TRIM_NUM, end_trim = END_TRIM_NUM, 
    #                             do_detrend=True, smoothing_factor = SMOOTHING,
    #                             do_zscore = True) 
    #             for file in func_files])
    # print(data.shape)

    mask = cortex.db.get_mask('MNI','atlas_152_sub1','thin')
    # mask data keeping only cortical voxels
    data = data[:,mask]
    data = np.nan_to_num(data)
    print(data.shape)

    ###########
    # HELPERS #
    ###########
    def _get_TR_features(subject, nlp_feat_type, layer=-1, sequence_length=-1):
        TR_duration = 1.49
        TR_aligned_features = []

        for subj_run in range(1,13):
            if subj_run < 10:
                run = '0' + str(subj_run)
            else:
                run = subj_run
        
            for session in repetitions[args.repetition]['sub-{}'.format(subject)]:
                datadir = '/share/volume0/cneuromod_data/movie10/derivatives/fmriprep-20.1.0/fmriprep/sub-{}/{}/func/'.format(subject, session)

                fname = '{}sub-{}_{}_task-figures{}_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(datadir, subject, session, run, args.repetition)
                if path.exists(fname):
                    print("Run {}, Session {}".format(subj_run, session))
                
                    dat = nibabel.load(fname).get_data()
                    # very important to transpose otherwise data and brain surface don't match
                    dat = dat.T 
        
                    break

            num_TRs = dat.shape[0]
            
            if nlp_feat_type == 'elmo':
                if sequence_length == -1:
                    loaded = np.load('/share/volume0/cneuromod_data/nlp_features/movie10/figures/elmo/hidden_figs_{}_elmo_sentence_layer_{}.npy'.format(subj_run, layer),allow_pickle=True)
                else:
                    loaded = np.load('/share/volume0/cneuromod_data/nlp_features/movie10/figures/elmo/hidden_figs_{}_elmo_length_{}_layer_{}.npy'.format(subj_run, sequence_length, layer),allow_pickle=True)
                aligned_features = lanczosinterp2D(loaded.item()['features'], loaded.item()['time'], np.arange(0,num_TRs)*TR_duration, window=3)[:,:512]
            elif nlp_feat_type == 'speaker':
                loaded = np.load('/share/volume0/cneuromod_data/nlp_features/movie10/figures/character_feats/hidden_figures_seg{}.npy'.format(run),allow_pickle=True)
                aligned_features = lanczosinterp2D(loaded.item()['char_feats'], loaded.item()['timestamps'], np.arange(0,num_TRs)*TR_duration, window=3)
            elif nlp_feat_type == 'binary_speech':
                loaded = np.load('/share/volume0/cneuromod_data/nlp_features/movie10/figures/character_feats/hidden_figures_seg{}.npy'.format(run),allow_pickle=True)
                aligned_features = lanczosinterp2D(np.ones(loaded.item()['char_feats'].shape), loaded.item()['timestamps'], np.arange(0,num_TRs)*TR_duration, window=3)
            TR_aligned_features.append(aligned_features[START_TRIM_NUM:-END_TRIM_NUM])

            print('done loading features for run {}'.format(subj_run))

        stacked_features = np.vstack(TR_aligned_features)
        if stacked_features.shape[1] > 20:
            pca = PCA(n_components=10, svd_solver='full')
            pca.fit(stacked_features)
            stacked_features = pca.transform(stacked_features)
        delayed_features = delay_mat(stacked_features, np.arange(1,7))

        return delayed_features
    ###########
    delayed_predict_features_elmo = _get_TR_features(args.subject, 'elmo', args.layer, args.sequence_length)
    delayed_predict_features_speaker = _get_TR_features(args.subject, 'speaker', args.layer, args.sequence_length)
    return data, [delayed_predict_features_elmo, delayed_predict_features_speaker]

####################
# STACKING ROUTINE #
####################
def stacking_CV_fmri(data, features, idx2features, args, n_folds=12):
    # INPUTS: data (n_time, n_voxels); features (list of (n_time, n_dim))
    
    n_time, n_voxels = data.shape
    n_features = len(features)
    
    ind = CV_ind(n_time, n_folds=n_folds)
    
    # Create storage arrays for r2s
    r2s = np.zeros((n_features, n_voxels))
    r2s_train_folds = np.zeros((n_folds, n_features, n_voxels))
    r2s_weighted = np.zeros((n_features, n_voxels))
    stacked_train_r2s_fold = np.zeros((n_folds, n_voxels))
    # Create storage arrays for corrs
    corrs = np.zeros((n_features, n_voxels))
    corrs_train_folds = np.zeros((n_folds, n_features, n_voxels))
    corrs_weighted = np.zeros((n_features, n_voxels))
    stacked_train_corrs_fold = np.zeros((n_folds, n_voxels))
    # Create storage arrays for predictions
    preds_test = np.zeros((n_features, n_time, n_voxels))
    weighted_pred = np.zeros((n_features, n_time, n_voxels))
    stacked_pred = np.zeros((n_time, n_voxels))
    # Create storage array for weights
    S_average = np.zeros((n_voxels, n_features))

    # Load stored weights
    loaded_weights = dict()
    for FEATURE in range(n_features):
        feat_name = idx2features[FEATURE]
        if feat_name == 'elmo':
            weights_path = '/share/volume0/abollu/neuromod/results/elmo/fold_weights_{}_rep_2_with_elmo_layer_1_len_25.npy'.format(args.subject)
        elif feat_name == 'speaker':
            weights_path = '/share/volume0/abollu/neuromod/results/speaker/fold_weights_{}_rep_2_with_speaker.npy'.format(args.subject)
        else:
            print("Unrecognized feature encountered: {} (not speaker, elmo)".format(feat_name))
        loaded_weights[feat_name] = np.load(weights_path, allow_pickle=True).item()['fold_weights_t']
    print("Stored encoding model weights loaded successfully!")

    for ind_num in range(n_folds):
        train_ind = ind!=ind_num
        test_ind = ind==ind_num
        
        # split data
        train_features = [F[train_ind] for F in features]
        train_data = data[train_ind]
        test_features = [F[test_ind] for F in features]
        test_data = data[test_ind]
        
        # normalize data
        train_features = [np.nan_to_num(zscore(F)) for F in train_features]
        train_data = np.nan_to_num(zscore(train_data))
        test_features = [np.nan_to_num(zscore(F)) for F in test_features]
        test_data = np.nan_to_num(zscore(test_data))
        
        err = dict()
        preds_train = dict()
        
        for FEATURE in range(n_features):
            feat_name = idx2features[FEATURE]
            weights = loaded_weights[feat_name][ind_num]
            preds_train[FEATURE] = np.dot(train_features[FEATURE], weights)
            err[FEATURE] = train_data - preds_train[FEATURE]
            preds_test[FEATURE, test_ind] = np.dot(test_features[FEATURE], weights)
            r2s_train_folds[ind_num, FEATURE, :] = R2(preds_train[FEATURE], train_data)
            corrs_train_folds[ind_num, FEATURE, :] = corr(preds_train[FEATURE], train_data)
            
        # Calculate error matrix for stacking
        P = np.zeros((n_voxels,n_features,n_features))
        for i in range(n_features):
            for j in range(n_features):
                P[:,i,j] = np.mean(err[i]*err[j],0)
                
        q = matrix(np.zeros((n_features)))
        G = matrix(-np.eye(n_features,n_features))
        h = matrix(np.zeros(n_features))
        A = matrix(np.ones((1,n_features)))
        b = matrix(np.ones(1))

        S = np.zeros((n_voxels,n_features))
        
        stacked_pred_train = np.zeros_like(train_data)
        
        for i in range(n_voxels):
            PP = matrix(P[i])
            # solve stacking weights for current voxel
            S[i,:] = np.array(solvers.qp(PP, q, G, h, A, b)['x']).reshape(n_features,)
            # combine predictions from individual feature spaces for current voxel
            z = np.array([preds_test[feature_j, test_ind, i] for feature_j in range(n_features)])
            # multiply predictions by feature weights from this solver
            stacked_pred[test_ind, i] = np.dot(S[i,:], z)
            if i==0:
                print("Z shape during solving for voxel 0: {}".format(z.shape))
            
            # do the same with training predictions
            z = np.array([preds_train[feature_j][:,i] for feature_j in range(n_features)])
            stacked_pred_train[:,i] = np.dot(S[i,:], z)
        
        S_average += S
        
        stacked_train_r2s_fold[ind_num,:] = R2(stacked_pred_train, train_data)
        stacked_train_corrs_fold[ind_num,:] = corr(stacked_pred_train, train_data)
        
        for FEATURE in range(n_features):
            # weight the predictions according to S:
            # weighted single feature space predictions, computed over a fold
            weighted_pred[FEATURE, test_ind] = preds_test[FEATURE, test_ind] * S[:, FEATURE]
        
    # compute overall
    for FEATURE in range(n_features):
        r2s[FEATURE, :] = R2(preds_test[FEATURE], data)
        r2s_weighted[FEATURE, :] = R2(weighted_pred[FEATURE], data)
        corrs[FEATURE, :] = corr(preds_test[FEATURE], data)
        corrs_weighted[FEATURE, :] = corr(weighted_pred[FEATURE], data)
    stacked_r2s = R2(stacked_pred, data)
    stacked_corrs = corr(stacked_pred, data)
    
    r2s_train = r2s_train_folds.mean(0)
    stacked_train_r2s = stacked_train_r2s_fold.mean(0)
    corrs_train = corrs_train_folds.mean(0)
    stacked_train_corrs = stacked_train_corrs_fold.mean(0)
    S_average = S_average/n_folds
    
    all_r2s_stored = {'r2s':r2s, 'stacked_r2s':stacked_r2s, 'r2s_weighted':r2s_weighted, 'r2s_train':r2s_train, 'stacked_train_r2s':stacked_train_r2s}
    all_corrs_stored = {'corrs':corrs, 'stacked_corrs':stacked_corrs, 'corrs_weighted':corrs_weighted, 'corrs_train':corrs_train, 'stacked_train_corrs':stacked_train_corrs}
    
    return all_r2s_stored, all_corrs_stored, S_average 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--layer", type=int, required=False)
    parser.add_argument("--sequence_length", type=int, required=False)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--regress_feat_types", default=0)

    args = parser.parse_args()
    print(args)

    idx2features = {0:'elmo', 1:'speaker'}
    data, features = save_data(args)
    all_r2s_stored, all_corrs_stored, S_average = stacking_CV_fmri(data, features, idx2features, args)
    out_dir = '/share/volume0/abollu/neuromod/results/stacking/'
    fname = 'stacking_results_{}_rep_{}_with_speaker_elmo.npy'.format(args.subject, args.repetition)
    print('saving: {}'.format(out_dir+fname))
    np.save(out_dir+fname, {'all_r2s_stored':all_r2s_stored,'all_corrs_stored':all_corrs_stored,'S_average':S_average})