###########
# IMPORTS #
###########
import numpy as np
from os import path
from utils.utils import load_and_process

####################
# GLOBAL VARIABLES #
####################
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
####################

def store_preprocessed_brain_data(args):
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
    data = np.vstack(data)

    store_path = '/share/volume0/abollu/neuromod/data/preprocessed_brain_data/sub{}_rep{}.npy'.format(args.subject, args.repetition)
    np.save(store_path, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    args = parser.parse_args()
    print(args)

    store_preprocessed_brain_data(args)