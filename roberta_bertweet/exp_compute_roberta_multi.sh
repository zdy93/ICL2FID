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


source /scratch/dzhang5/visa/visa39/bin/activate

assign_token_weight=True
assign_seq_weight=True

for seed in 2022
do
# for size in 0.5
# for size in 1.0 0.5 0.25 0.125 0.0625 0.03125
for size in 1.0 0.5 0.25 0.125
do
for model in "roberta-base" 
# for model in "roberta-base"
# for model in "vinai/bertweet-base"
do
# for model_type in "bertweet-multi-crf" "bertweet-multi"
for model_type in "bertweet-multi"
do
for agg in "expert"
do
   
python compute_multi_performance.py \
   --seed $seed \
   --bert_model $model \
   --task_type relevant_entity_detection \
   --model_type ${model_type} \
   --data tweet-fid/LREC_${agg} \
   --train_file ${size}.expert.devfortrain.pkl \
   --val_file traintestfortest.p \
   --test_file traintestfortest.p \
   --label_map label_map.json \
   --assign_token_weight ${assign_token_weight} \
   --assign_seq_weight ${assign_seq_weight} \
   --n_epochs 20 \
   --learning_rate 1e-5 \
   --token_lambda 10 \
   --log_dir log-multi \
   --do_bootstrap True \
   --bootstrap_time 200 

done
done
done
done
done