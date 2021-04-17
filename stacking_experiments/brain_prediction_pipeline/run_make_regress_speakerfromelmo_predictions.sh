#!/bin/bash
source ~/.bashrc
conda activate /home/abollu/anaconda3
python3 /home/abollu/cneuromod/brain_prediction_pipeline/make_regress_speakerfromelmo_predictions.py --repetition=$repetition --perm_block=$perm_block --perm_shift=$perm_shift --subject=$subject  --predict_feat_type=$predict_feat_type --nlp_feat_type=$nlp_feat_type --nlp_feat_dir=$nlp_feat_dir --layer=$layer --sequence_length=$sequence_length --output_dir=$output_dir --delay $delay --output_fname_prefix $output_prefix  --regress_feat_types $regress_feat_types
