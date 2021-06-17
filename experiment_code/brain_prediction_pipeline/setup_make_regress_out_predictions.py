import os
import math
import numpy as np

#all_feat_combs = ['context', 'context','emb']
#regress_feat_combs = ['emb','','']

meg_subjects = ['A','B','C','D','E','G','H','I'] #['A','B','D'] #['H','K','L']
fmri_subjects = ['I', 'M', 'L','J', 'H','K','N','G','F'] #G
cneuromod_subjects = ['01', '02', '03', '04', '05', '06']
#process_slugs = ['trans-D_nsb-5_cb-0'];#_corr-.98_empty-4-10-2-2','trans-D_nsb-5_cb-0_corr-.98_empty-8-20-2-2']#, 'trans-D_nsb-5_cb-0_empty-4-10-2-2']

#repo_root = '/bigbrain/bigbrain.usr1/homes/mktoneva/src/trunk'
save_dir = '/share/volume0/abollu/neuromod/results/refresh_results'
#special_feat_dir = '/' + feat_type + '/' #no_segments_BERT_features/'


repetition = 2
nlp_feat_type = 'words_per_TR' #'elmo','speaker','binary_speech','words_per_TR'


nlp_feat_dir_format = '/home/abollu/cneuromod/brain_prediction_pipeline'
delay = 0
subjects = fmri_subjects

all_regress_feat_types = ['0'] #'0'

#other_feat_type = [0,'prev-1-emb',0,'emb'] #[0,'emb','prev-24-emb','emb+prev-24-emb',0,'prev-24-emb','prev-1-context','prev-24-emb+prev-1-context']


task = ''
token_type = ''


nlp_feat_dir = nlp_feat_dir_format #.format(predict_feat_type)

perm_shift = 0
perm_block = 0
layer = 1 
seq_len = 25


predict_feat_type = nlp_feat_type

for subject in cneuromod_subjects:
    for regress_feat_types in all_regress_feat_types:
        regress_feat_suffix = '' if (regress_feat_types == '0') else '_without_{}'.format(regress_feat_types)
        output_dir = '{}/{}{}/'.format(save_dir, predict_feat_type, regress_feat_suffix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_fname = subject
        qsubStr = 'qsub -l walltime=24:00:00 -l vmem="100gb" -p 0 -q gpu -v repetition=%d,perm_shift=%d,perm_block=%d,predict_feat_type=%s,nlp_feat_type=%s,nlp_feat_dir=%s,layer=%d,sequence_length=%d,output_prefix=%s,delay=%d,regress_feat_types=%s,subject=%s,output_dir=%s run_make_regress_out_predictions.sh' % (repetition,perm_shift,perm_block, predict_feat_type, nlp_feat_type, nlp_feat_dir, layer, seq_len, output_fname, delay, regress_feat_types, subject, output_dir)
        print(qsubStr)
        os.system(qsubStr)
