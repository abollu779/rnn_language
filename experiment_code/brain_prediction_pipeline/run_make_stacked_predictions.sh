#!/bin/bash
source ~/.bashrc
conda activate /home/abollu/anaconda3
python3 /home/abollu/cneuromod/brain_prediction_pipeline/make_stacked_predictions.py --repetition=$repetition --subject=$subject --layer=$layer --sequence_length=$sequence_length  --regress_feat_types $regress_feat_types
