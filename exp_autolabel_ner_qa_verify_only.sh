#!/bin/bash
#SBATCH --job-name=sbs		# job name
#SBATCH --mem 30G		# memory size
#SBATCH -n 4 			# number of cpu
#SBATCH -N 1 			# number of node
##SBATCH --gres=gpu:1		# number of gpu
##SBATCH -C H100|A100|A100-80G|V100			# name of gpu that you want to use
##SBATCH -x gpu-4-01,gpu-4-04,gpu-4-11,gpu-4-13	# name of node that you don't want to request for
#SBATCH -p short		# maximum running time, long (7 days) or short (1 day)
##SBATCH -d afterany:375516
##SBATCH -o H-output     	# STDOUT
##SBATCH -e H-error      	# STDERR
##SBATCH --mail-type=BEGIN,END	# notifications for job done & fail
##SBATCH --mail-user=dzhang5@wpi.edu 

# for entity_type in "Food" "Location" "Symptom" "Keyword"
# # for entity_type in "Food"
# do
size=${1:-1.0}
agg_method=${2:-"expert"}
model_name=${3:-"google/flan-t5-small"}
few_shot_selection=${4:-"semantic_similarity"}
temperature=${5:-0.1}
entity_type=${6:-"Food"}
verify=${7:-"True"}
label_version=${8:-'v1'}
task_version=${9:-'v1'}
verify_few_shot_selection=${10:-"label_diversity_similarity"}
few_shot_path='/scratch/dzhang5/LLM/TWEET-FID/'${size}'.'${agg_method}'.devfortrain.short.csv'
verify_few_shot_path='/scratch/dzhang5/LLM/TWEET-FID/'${entity_type}'-verify.'${size}'.'${agg_method}'.devfortrain.short.csv'
data_path='/scratch/dzhang5/LLM/TWEET-FID/expert.traintestfortest.csv'
output_dir='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-ner-qa-'${agg_method}
text_column="context"
label_column=${entity_type}_answer
example_selection_label_column=has_${entity_type}
label_symbol="^^^^"
token_path="/home/dzhang5/.cache/huggingface/token"
few_shot_num=8
cache=False
console_output=False
output_dir=${output_dir}/${temperature}/${size}/lv_${label_version}_tv_${task_version}


trans_model_name=$(echo "$model_name" | sed 's/\//_/g')
data_name=$(basename "$data_path")

# Specify the directory path
directory=NER_output/${agg_method}/${size}/lv_${label_version}_tv_${task_version}

# Check if the directory exists
if [ ! -d "$directory" ]; then
    # Create the directory
    mkdir -p "$directory"
fi

output_file=NER_output/${agg_method}/${size}/lv_${label_version}_tv_${task_version}/out_AutoLabel_NER_QA_verification_only_${trans_model_name}_${few_shot_selection}_${entity_type}_${data_name}_${size}_${agg_method}.ipynb

papermill \
AutoLabel_NER_QA_verification_only.ipynb \
${output_file} \
-k py3.9-visa \
-p few_shot_path ${few_shot_path} \
-p verify_few_shot_path ${verify_few_shot_path} \
-p data_path ${data_path} \
-p output_dir ${output_dir} \
-p model_name ${model_name} \
-p label_column ${label_column} \
-p text_column ${text_column} \
-p few_shot_num ${few_shot_num} \
-p example_selection_label_column ${example_selection_label_column} \
-p label_symbol ${label_symbol} \
-p few_shot_selection ${few_shot_selection} \
-p verify_few_shot_selection ${verify_few_shot_selection} \
-p token_path ${token_path} \
-p cache ${cache} \
-p console_output ${console_output} \
-p temperature ${temperature} \
-p verify ${verify} \
-p label_version ${label_version} \
-p task_version ${task_version}

# done