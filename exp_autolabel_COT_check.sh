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
verify=${6:-"True"}
check_mode=${7:-"strict"}
label_version=${8:-'v1'}
task_version=${9:-'v1'}
word_few_shot_selection=${10:-"semantic_similarity"}

trans_model_name=$(echo "$model_name" | sed 's/\//_/g')

# Set the Internal Field Separator to "/"
IFS='/' read -ra parts <<< "$model_name"
# Get the last part
last_model_name="${parts[-1]}"

few_shot_path='/scratch/dzhang5/LLM/TWEET-FID/'${size}'.'${agg_method}'.devfortrain.short.csv'
if [[ "$verify" == "True" ]]; then
data_path='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-ner-qa-'${agg_method}'/'${temperature}'/'${size}'/'${last_model_name}'_'${check_mode}'_'${word_few_shot_selection}'_aggregated_final_expert.traintestfortest.csv'
else
data_path='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-ner-qa-'${agg_method}'/'${temperature}'/'${size}'/'${last_model_name}'_'${check_mode}'_'${word_few_shot_selection}'_aggregated_expert.traintestfortest.csv'
fi
output_dir='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-'${agg_method}

label_column='sentence_class'
text_column='context'
explanation_column='sentence_check_explanation'
reference_column='sentence_check_reference'
example_selection_label_column='sentence_check_class'
token_path="/home/dzhang5/.cache/huggingface/token"
few_shot_num=8


cache=False
console_output=False

output_dir=${output_dir}/${temperature}/${size}
data_name=$(basename "$data_path")

# Specify the directory path
directory=TC_output/${agg_method}/${size}/lv_${label_version}_tv_${task_version}

# Check if the directory exists
if [ ! -d "$directory" ]; then
    # Create the directory
    mkdir -p "$directory"
fi

output_file=TC_output/${agg_method}/${size}/lv_${label_version}_tv_${task_version}/out_AutoLabel_COT_check_${trans_model_name}_${check_mode}_${few_shot_selection:0:3}_${word_few_shot_selection:0:3}_${data_name}_current_${use_current_explanation}_ground_${use_ground_explanation}.ipynb

papermill \
AutoLabel_COT_check.ipynb \
${output_file} \
-k py3.9-visa \
-p few_shot_path ${few_shot_path} \
-p data_path ${data_path} \
-p model_name ${model_name} \
-p output_dir ${output_dir} \
-p label_column ${label_column} \
-p text_column ${text_column} \
-p explanation_column ${explanation_column} \
-p reference_column ${reference_column} \
-p example_selection_label_column ${example_selection_label_column} \
-p few_shot_num ${few_shot_num} \
-p few_shot_selection ${few_shot_selection} \
-p token_path ${token_path} \
-p cache ${cache} \
-p console_output ${console_output} \
-p temperature ${temperature}