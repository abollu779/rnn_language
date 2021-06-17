###########
# IMPORTS #
###########
import argparse
import numpy as np

from utils.utils import delay_mat, delay_one, zscore_word_data, run_class_time_CV_fmri_crossval_ridge_neuromod, distributed_speaker_per_TR
from utils.utils import subject_runs, surfaces, transforms, load_transpose_zscore, smooth_run_not_masked
from utils.utils import sensor_groups, classify_predictions, binary_classify_neighborhoods, CV_ind
from utils.utils import lanczosinterp2D

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

################
# MAIN ROUTINE #
################
def save_brain_activity_preds(args):
    if args.regress_feat_types != '0':
        regress_feat_names_list = np.sort(args.regress_feat_types.split('+'))
    else:
        regress_feat_names_list = []

    preprocessed_data_path = '/share/volume0/abollu/neuromod/data/preprocessed_brain_data/sub{}_rep{}.npy'.format(args.subject, args.repetition)
    data = np.load(preprocessed_data_path)

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
                # aligned_features = lanczosinterp2D(loaded.item()['char_feats'], loaded.item()['timestamps'], np.arange(0,num_TRs)*TR_duration, window=3)
                aligned_features = distributed_speaker_per_TR(loaded.item()['char_feats'], loaded.item()['timestamps'], np.arange(1,num_TRs+1)*TR_duration) # (1, num_TRs+1) because including end times of each TR rather than start time here 
            elif nlp_feat_type == 'binary_speech':
                loaded = np.load('/share/volume0/cneuromod_data/nlp_features/movie10/figures/character_feats/hidden_figures_seg{}.npy'.format(run),allow_pickle=True)
                aligned_features = lanczosinterp2D(np.ones(loaded.item()['char_feats'].shape), loaded.item()['timestamps'], np.arange(0,num_TRs)*TR_duration, window=3)
            elif nlp_feat_type == 'words_per_TR':
                loaded = np.load('/share/volume0/abollu/neuromod/data/words_per_TR/sub{}/hidden_figures_seg{}.npy'.format(subject, run))
                aligned_features = loaded # since word rate data is already collected with respect to TR times
            TR_aligned_features.append(aligned_features[START_TRIM_NUM:-END_TRIM_NUM])

            print('done loading features for run {}'.format(subj_run))

        stacked_features = np.vstack(TR_aligned_features)
        if stacked_features.shape[1] > 20:
            pca = PCA(n_components=10, svd_solver='full')
            pca.fit(stacked_features)
            stacked_features = pca.transform(stacked_features)
        delayed_features = delay_mat(stacked_features, np.arange(1,7))

        return delayed_features
    

    def _load_features_to_regress_out(feat_name_list): 
        if len(feat_name_list) == 0:
            return [],[]

        regress_features = []

        print(feat_name_list)
        for feat_name in feat_name_list:
            features = _get_TR_features(args.subject, feat_name, layer=1, sequence_length=25)
            regress_features.append(features)
                                    
        return np.hstack(regress_features)
    ###########
    delayed_predict_features = _get_TR_features(args.subject, args.nlp_feat_type, args.layer, args.sequence_length)
    if len(regress_feat_names_list) == 0:
        regress_features = []
    else:
        regress_features = _load_features_to_regress_out(regress_feat_names_list)
    
    results = run_class_time_CV_fmri_crossval_ridge_neuromod(data, delayed_predict_features, regress_features)
    corrs_t, preds_t, test_t = results['corrs_t'], results['preds_t'], results['test_t']
    regressfeat2predictfeat_preds_t, predictfeat_test_t = results['regressfeat2predictfeat_preds_t'], results['predictfeat_test_t']
    predictfeat2regressfeat_preds_t, regressfeat_test_t = results['predictfeat2regressfeat_preds_t'], results['regressfeat_test_t']

    # Save fold weights
    if args.predict_feat_type == 'elmo' or args.predict_feat_type == 'bert':
        if args.regress_feat_types != '0':
            fname = 'predict_{}_rep_{}_with_{}_layer_{}_len_{}_regress_out_{}'.format(args.output_fname_prefix,args.repetition, args.predict_feat_type, args.layer, args.sequence_length, '+'.join(regress_feat_names_list))
        else:
            fname = 'predict_{}_rep_{}_with_{}_layer_{}_len_{}'.format(args.output_fname_prefix, args.repetition,args.predict_feat_type, args.layer, args.sequence_length)
    elif args.predict_feat_type == 'speaker' or args.predict_feat_type == 'binary_speech' or args.predict_feat_type == 'words_per_TR':
        fname = 'predict_{}_rep_{}_with_{}'.format(args.output_fname_prefix,args.repetition, args.predict_feat_type) 
    else:
        if args.regress_feat_types != '0':
            fname = 'predict_{}_rep_{}_with_{}_regress_out_{}'.format(args.output_fname_prefix, args.repetition,args.predict_feat_type, '+'.join(regress_feat_names_list))
        else:
            fname = 'predict_{}_rep_{}_with_{}'.format(args.output_fname_prefix, args.repetition, args.predict_feat_type)
    
    if args.perm_shift > 0:
        fname = fname + '_shift_{}'.format(args.perm_shift)
    if args.perm_block > 0:
        fname = fname + '_block_{}'.format(args.perm_block)

    print('saving: {}'.format(args.output_dir + fname))
    np.save(args.output_dir + fname, {'corrs_t':corrs_t,'preds_t':preds_t,'test_t':test_t,'regressfeat2predictfeat_preds_t':regressfeat2predictfeat_preds_t,'predictfeat_test_t':predictfeat_test_t,'predictfeat2regressfeat_preds_t':predictfeat2regressfeat_preds_t,'regressfeat_test_t':regressfeat_test_t})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--predict_feat_type", required=True)
    parser.add_argument("--nlp_feat_type", required=True)
    parser.add_argument("--nlp_feat_dir", required=True)
    parser.add_argument("--layer", type=int, required=False)
    parser.add_argument("--sequence_length", type=int, required=False)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--output_fname_prefix", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--regress_feat_types", default=0)
    parser.add_argument("--perm_shift", type=int, default=0)
    parser.add_argument("--perm_block", type=int, default=0)

    args = parser.parse_args()
    print(args)

    save_brain_activity_preds(args)



