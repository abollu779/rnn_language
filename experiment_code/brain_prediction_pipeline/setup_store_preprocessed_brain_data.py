import os
import math
import numpy as np

cneuromod_subjects = ['01', '02', '03', '04', '05', '06']
repetition = 2

for subject in cneuromod_subjects:	
    qsubStr = 'qsub -l walltime=24:00:00 -l vmem="100gb" -p 0 -q gpu -v repetition=%d,subject=%s run_store_preprocessed_brain_data.sh' % (repetition, subject)
    print(qsubStr)
    os.system(qsubStr)