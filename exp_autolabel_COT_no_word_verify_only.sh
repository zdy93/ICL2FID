#!/bin/bash
#SBATCH --job-name=sbs		# job name
#SBATCH --mem 30G		# memory size
#SBATCH -n 4 			# number of cpu
#SBATCH -N 1 			# number of node
##SBATCH --gres=gpu:1		# number of gpu
##SBATCH -C H100|A100|A100-80G			# name of gpu that you want to use
##SBATCH -x gpu-4-01,gpu-4-04,gpu-4-11,gpu-4-13	# name of node that you don't want to request for
#SBATCH -p short		# maximum running time, long (7 days) or short (1 day)
##SBATCH -d afterany:375534
##SBATCH -o H-output     	# STDOUT
##SBATCH -e H-error      	# STDERR
##SBATCH --mail-type=BEGIN,END	# notifications for job done & fail
##SBATCH --mail-user=dzhang5@wpi.edu 

size=${1:-1.0}
agg_method=${2:-"expert"}
model_name=${3:-"refuel-llm"}
few_shot_selection=${4:-"semantic_similarity"}
temperature=${5:-0.1}
verify_few_shot_selection=${6:-"semantic_similarity"}
trans_model_name=$(echo "$model_name" | sed 's/\//_/g')
few_shot_path='/scratch/dzhang5/LLM/TWEET-FID/'${size}'.'${agg_method}'.devfortrain.short.csv'
data_path='/scratch/dzhang5/LLM/TWEET-FID/expert.traintestfortest.csv'
output_dir='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-'${agg_method}
label_column='sentence_class'
text_column='context'
explanation_column='sentence_explanation'
token_path="/home/dzhang5/.cache/huggingface/token"
few_shot_num=8

use_current_explanation=False
# use_current_explanation=True
# use_ground_explanation=True
use_ground_explanation=False
cache=False
console_output=False

output_dir=${output_dir}/${temperature}/${size}


data_name=$(basename "$data_path")
output_file=TC_output/${agg_method}/${size}/out_AutoLabel_COT_verify_only_${trans_model_name}_${few_shot_selection}_${data_name}_current_${use_current_explanation}_ground_${use_ground_explanation}.ipynb

papermill \
AutoLabel_COT_verify_only.ipynb \
${output_file} \
-k py3.9-visa \
-p few_shot_path ${few_shot_path} \
-p data_path ${data_path} \
-p model_name ${model_name} \
-p output_dir ${output_dir} \
-p label_column ${label_column} \
-p text_column ${text_column} \
-p explanation_column ${explanation_column} \
-p few_shot_num ${few_shot_num} \
-p few_shot_selection ${few_shot_selection} \
-p verify_few_shot_selection ${verify_few_shot_selection} \
-p use_current_explanation ${use_current_explanation} \
-p use_ground_explanation ${use_ground_explanation} \
-p token_path ${token_path} \
-p cache ${cache} \
-p console_output ${console_output} \
-p temperature ${temperature}