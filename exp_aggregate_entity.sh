#!/bin/bash
#SBATCH --job-name=sbs		# job name
#SBATCH --mem 30G		# memory size
#SBATCH -n 4 			# number of cpu
#SBATCH -N 1 			# number of node
#SBATCH -p short		# maximum running time, long (7 days) or short (1 day)
##SBATCH -d afterany:375528
##SBATCH -o H-output     	# STDOUT
##SBATCH -e H-error      	# STDERR
##SBATCH --mail-type=BEGIN,END	# notifications for job done & fail
##SBATCH --mail-user=dzhang5@wpi.edu 

source /scratch/dzhang5/visa/visa39/bin/activate

size=${1:-1.0}
agg_method=${2:-"expert"}
model_name=${3:-"google/flan-t5-small"}
few_shot_selection=${4:-"semantic_similarity"}
temperature=${5:-0.1}
verified=${6:-"True"}
check_mode=${7:-"strict"}
label_version=${8:-'v1'}
task_version=${9:-'v1'}
verify_few_shot_selection=${10:-"label_diversity_similarity"}
data_path='/scratch/dzhang5/LLM/TWEET-FID/expert.traintestfortest.csv'
output_dir='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-ner-qa-'${agg_method}
label_symbol="^^^^"
text_column="context"
result_dir='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-ner-qa-'${agg_method}
output_dir=${output_dir}/${temperature}/${size}/lv_${label_version}_tv_${task_version}
result_dir=${result_dir}/${temperature}/${size}/lv_${label_version}_tv_${task_version}

python aggregate_entity_results.py \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --model_name ${model_name} \
    --few_shot_selection ${few_shot_selection} \
    --verify_few_shot_selection ${verify_few_shot_selection} \
    --label_symbol ${label_symbol} \
    --text_column ${text_column} \
    --verified ${verified} \
    --check_mode ${check_mode} \
    --result_dir ${result_dir}