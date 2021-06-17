import os
import math
import numpy as np

cneuromod_subjects = ['01', '02', '03', '04', '05', '06']
repetition = 2
all_regress_feat_types = ['0'] #'0'
layer = 1 
seq_len = 25

for subject in cneuromod_subjects:
	for regress_feat_types in all_regress_feat_types:	
                qsubStr = 'qsub -l walltime=24:00:00 -l vmem="100gb" -p 0 -q gpu -v repetition=%d,layer=%d,sequence_length=%d,regress_feat_types=%s,subject=%s run_make_stacked_predictions.sh' % (repetition, layer, seq_len, regress_feat_types, subject)
                print(qsubStr)
                os.system(qsubStr)
