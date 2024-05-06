#!/bin/bash
size=${1:-1.0}
# size=${1:-0.5}
# size=${1:-0.25}
# size=${1:-0.125}
# size=${1:-0.0625}
# size=${1:-0.03125}

# agg=${2:-"mv"}
agg=${2:-"expert"}

check_mode="strict"

# cot_col="one_step"
cot_col="two_step"

verify=False
label_version='v1'
task_version='v1'

# for ft in "semantic_similarity"  "label_diversity_similarity" 
# for ft in "label_diversity_similarity"
for ft in "semantic_similarity" 
# for ft in "fixed" 
do
# for mn in "google/flan-t5-small" "google/flan-t5-xxl" "google/pegasus-x-base" "microsoft/prophetnet-large-uncased" "gpt2" "tiiuae/falcon-40b" "meta-llama/Llama-2-13b-chat-hf" "mistralai/Mixtral-8x7B-Instruct-v0.1" "refuel-llm" "llama-13b-chat" "google/gemma-7b-it" "refuel-llm" "gpt-3.5-turbo"

# for mn in "refuel-llm"
for mn in "gpt-3.5-turbo"
# for mn in "meta-llama/Llama-2-13b-chat-hf"
do
   vft="label_diversity_similarity"
   # vft="fixed"
   # vft="semantic_similarity"
   
   # tft="label_diversity_similarity"
   tft="semantic_similarity"
   # tft="fixed"
   
   tvft="label_diversity_similarity"
   # tvft="semantic_similarity"
   
   # svft="label_diversity_similarity"
   svft="semantic_similarity"
   
   tmp=0.1
   ets=( "Food" "Location" "Symptom" "Keyword" )
   R_jobs=( 424435 425158 425161 424439 )
   G_jobs=( 443977 443978 443979 443980 )
   L_jobs=( 428769 429810 428767 428768 )
   djobstr=""
   for L_job in "${L_jobs[@]}"; do
   # Concatenate each element with ":" to the result string
   L_jobstr+=":$L_job"
   done
   # Remove leading ":" from the result string
   L_jobstr="${L_jobstr#:}"
   for G_job in "${G_jobs[@]}"; do
   # Concatenate each element with ":" to the result string
   G_jobstr+=":$G_job"
   done
   # Remove leading ":" from the result string
   G_jobstr="${G_jobstr#:}"
   for i in 0
   do
   et="${ets[$i]}"
   R_job="${R_jobs[$i]}"
   G_job="${G_jobs[$i]}"
   L_job="${L_jobs[$i]}"
   if [[ "$mn" == "refuel-llm" ]]; then
   echo $et $R_job 
   # sbatch --depend=afterany:${R_job} exp_aggregate_entity.sh $size $agg $mn $ft $tmp $verify $check_mode $label_version $task_version $vft
   # sbatch --depend=afterany:${R_job} exp_aggregate_COT_entity.sh $size $agg $mn $ft $tmp $verify ${cot_col} ${check_mode} $label_version $task_version $vft
   # sbatch --depend=afterany:${R_job} exp_aggregate_COT_VCOT_entity.sh $size $agg $mn $ft $tmp "True" ${cot_col} ${check_mode} $label_version $task_version $vft 
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_ner_qa_verify.sh $size $agg $mn $ft $tmp $et $verify $label_version $task_version
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_ner_COT_qa_verify.sh $size $agg $mn $ft $tmp $et $verify ${cot_col} $label_version $task_version
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_ner_qa_verify_relabel.sh $size $agg $mn $ft $tmp $et $verify $label_version $task_version
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_ner_COT_qa_verify_relabel.sh $size $agg $mn $ft $tmp $et $verify ${cot_col} $label_version $task_version
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_ner_qa_verify_only.sh $size $agg $mn $ft $tmp $et "True" $label_version $task_version $vft
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_ner_COT_qa_verify_only.sh $size $agg $mn $ft $tmp $et "True" ${cot_col} $label_version $task_version $vft
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_ner_qa_verify_COT_only.sh $size $agg $mn $ft $tmp $et "True" $label_version $task_version $vft
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_ner_COT_qa_verify_COT_only.sh $size $agg $mn $ft $tmp $et "True" ${cot_col} $label_version $task_version $vft
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_qa.sh $mn $ft $tmp
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel.sh $mn $ft $tmp
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_ner.sh $mn $ft $tmp
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_COT.sh $size $agg $mn $ft $tmp $verify $check_mode $label_version $task_version
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_COT_COT.sh $size $agg $mn $ft $tmp $verify $check_mode $cot_col $label_version $task_version $vft
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_COT_VCOT_COT.sh $size $agg $mn $ft $tmp "True" $check_mode $cot_col $label_version $task_version $vft
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_COT_check.sh $size $agg $mn $tft $tmp $verify $check_mode $label_version $task_version $ft
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_COT_COT_check.sh $size $agg $mn $tft $tmp $verify $check_mode $cot_col $label_version $task_version $ft $vft
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_COT_VCOT_COT_check.sh $size $agg $mn $tft $tmp "True" $check_mode $cot_col $label_version $task_version $ft $vft
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel_COT_no_word.sh $size $agg $mn $tft $tmp
   # sbatch --gres=gpu:1 --constraint="V100|H100|A100|A100-80G" --depend=afterany:${R_job} exp_autolabel.sh $size $agg $mn $tft $tmp
   elif [[ "$mn" == gpt* ]]; then
   echo $et $G_job
   # sbatch --depend=afterany:${G_job} exp_aggregate_entity.sh $size $agg $mn $ft $tmp $verify $check_mode $label_version $task_version $vft
   # sbatch --depend=afterany:${G_job} exp_aggregate_COT_entity.sh $size $agg $mn $ft $tmp $verify ${cot_col} ${check_mode} $label_version $task_version $vft
   # sbatch --depend=afterany:${G_jobstr} exp_aggregate_COT_VCOT_entity.sh $size $agg $mn $ft $tmp "True" ${cot_col} ${check_mode} $label_version $task_version $vft
   # sbatch --depend=afterany:${G_jobstr} exp_aggregate_COT_VCOT_AE_entity.sh $size $agg $mn $ft $tmp "True" ${cot_col} ${check_mode} $label_version $task_version $vft
   # sbatch --depend=afterany:${G_jobstr} exp_aggregate_COT_VCOT_entity_LONG.sh $size $agg $mn $ft $tmp "True" ${cot_col} ${check_mode} $label_version $task_version $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_qa_verify.sh $size $agg $mn $ft $tmp $et $verify $label_version $task_version
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_qa_verify.sh $size $agg $mn $ft $tmp $et $verify ${cot_col} $label_version $task_version
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_qa_verify_LONG.sh $size $agg $mn $ft $tmp $et $verify ${cot_col} $label_version $task_version
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_ae_verify.sh $size $agg $mn $ft $tmp $et $verify $label_version $task_version
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_qa_check_verify.sh $size $agg $mn $ft $tmp $et $verify $label_version $task_version $tft $tvft "True"
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_qa_check_verify.sh $size $agg $mn $ft $tmp $et $verify $label_version $task_version $tft $tvft "False"
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_qa_verify_only.sh $size $agg $mn $ft $tmp $et "True" $label_version $task_version $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_qa_verify_only.sh $size $agg $mn $ft $tmp $et "True" ${cot_col} $label_version $task_version $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_qa_verify_COT_only.sh $size $agg $mn $ft $tmp $et "True" $label_version $task_version $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_qa_verify_COT_only.sh $size $agg $mn $ft $tmp $et "True" ${cot_col} $label_version $task_version $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_qa_verify_COT_only_LONG.sh $size $agg $mn $ft $tmp $et "True" ${cot_col} $label_version $task_version $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_ae_verify_COT_only.sh $size $agg $mn $ft $tmp $et "True" $label_version $task_version $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_qa_verify_SECOND_COT_only.sh $size $agg $mn $ft $tmp $et "True" ${cot_col} $label_version $task_version $vft $tft $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_qa_check_verify_COT_only.sh $size $agg $mn $ft $tmp $et "True" $label_version $task_version $tft $tvft "True" $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner_COT_qa_check_verify_COT_only.sh $size $agg $mn $ft $tmp $et "True" $label_version $task_version $tft $tvft "False" $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_qa.sh $mn $ft $tmp
   # sbatch --depend=afterany:${G_job} exp_autolabel.sh $mn $ft $tmp
   # sbatch --depend=afterany:${G_job} exp_autolabel_ner.sh $mn $ft $tmp
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT.sh $size $agg $mn $ft $tmp $verify $check_mode
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_COT.sh $size $agg $mn $ft $tmp $verify $check_mode $cot_col $label_version $task_version $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_VCOT_COT.sh $size $agg $mn $tft $tmp "True" $check_mode $cot_col $label_version $task_version $ft $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_check.sh $size $agg $mn $tft $tmp $verify $check_mode $ft
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_COT_check.sh $size $agg $mn $tft $tmp $verify $check_mode $cot_col $label_version $task_version $ft $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_VCOT_COT_check.sh $size $agg $mn $tft $tmp "True" $check_mode $cot_col $label_version $task_version $ft $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_VCOT_COT_check_LONG.sh $size $agg $mn $tft $tmp "True" $check_mode $cot_col $label_version $task_version $ft $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_VCOT_COT_AE_check.sh $size $agg $mn $tft $tmp "True" $check_mode $cot_col $label_version $task_version $ft $vft
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_VCOT_COT_check_dummy.sh $size $agg $mn $tft $tmp "True" $check_mode $cot_col $label_version $task_version $ft $vft
   sbatch --depend=afterany:${G_job} exp_autolabel_COT_VCOT_COT_check_verify_only.sh $size $agg $mn $tft $tmp "True" $check_mode $cot_col $label_version $task_version $ft $vft $tvft
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_no_word.sh $size $agg $mn $tft $tmp
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_no_word_verify_only.sh $size $agg $mn $tft $tmp $tvft
   # sbatch --depend=afterany:${G_job} exp_autolabel.sh $size $agg $mn $tft $tmp
   # sbatch --depend=afterany:${G_job} exp_autolabel_COT_ae_all.sh $size $agg $mn $ft $tmp $label_version $task_version
   else
   echo $et $L_job
   # sbatch --depend=afterany:${L_jobstr} exp_aggregate_entity.sh $size $agg $mn $ft $tmp $verify $check_mode $label_version $task_version $vft
   # sbatch --depend=afterany:${L_jobstr} exp_aggregate_COT_entity.sh $size $agg $mn $ft $tmp $verify ${cot_col} ${check_mode} $label_version $task_version $vft
   # sbatch --depend=afterany:${L_jobstr} exp_aggregate_COT_VCOT_entity.sh $size $agg $mn $ft $tmp "True" ${cot_col} ${check_mode} $label_version $task_version $vft
   # sbatch --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_ner_qa_verify.sh $size $agg $mn $ft $tmp $et $verify $label_version $task_version
   # sbatch --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_ner_COT_qa_verify.sh $size $agg $mn $ft $tmp $et $verify ${cot_col} $label_version $task_version
   # sbatch --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_ner_qa_verify_only.sh $size $agg $mn $ft $tmp $et "True" $label_version $task_version $vft
   # sbatch --depend=afterany:${L_job} --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_ner_COT_qa_verify_only.sh $size $agg $mn $ft $tmp $et "True" ${cot_col} $label_version $task_version $vft
   # sbatch --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_ner_qa_verify_COT_only.sh $size $agg $mn $ft $tmp $et "True" $label_version $task_version $vft
   # sbatch --depend=afterany:${L_job} --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_ner_COT_qa_verify_COT_only.sh $size $agg $mn $ft $tmp $et "True" ${cot_col} $label_version $task_version $vft
   # sbatch --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_qa.sh $mn $ft $tmp
   # sbatch --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel.sh $mn $ft $tmp
   # sbatch --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_ner.sh $mn $ft $tmp
   # sbatch --depend=afterany:${L_job} --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_COT.sh $size $agg $mn $ft $tmp $verify $check_mode
   # sbatch --depend=afterany:${L_job} --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_COT_COT.sh $size $agg $mn $ft $tmp $verify $check_mode $cot_col $label_version $task_version $vft
   # sbatch --depend=afterany:${L_job} --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_COT_VCOT_COT.sh $size $agg $mn $ft $tmp "True" $check_mode $cot_col $label_version $task_version $vft
   # sbatch --depend=afterany:${L_job} --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_COT_check.sh $size $agg $mn $tft $tmp $verify $check_mode $ft
   # sbatch --depend=afterany:${L_job} --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_COT_COT_check.sh $size $agg $mn $tft $tmp $verify $check_mode $cot_col $label_version $task_version $ft $vft
   # sbatch --depend=afterany:${L_job} --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_COT_VCOT_COT_check.sh $size $agg $mn $tft $tmp "True" $check_mode $cot_col $label_version $task_version $ft $vft
   # sbatch --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel_COT_no_word.sh $size $agg $mn $tft $tmp
   # sbatch --gres=gpu:1 --constraint="H100|A100|A100-80G" exp_autolabel.sh $size $agg $mn $tft $tmp
   fi
   done
   
done
done