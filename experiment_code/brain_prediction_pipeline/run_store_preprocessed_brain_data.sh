#!/bin/bash
source ~/.bashrc
conda activate /home/abollu/anaconda3
python3 /home/abollu/cneuromod/brain_prediction_pipeline/store_preprocessed_brain_data.py --repetition=$repetition --subject=$subject
