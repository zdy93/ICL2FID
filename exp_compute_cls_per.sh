source /scratch/dzhang5/visa/visa39/bin/activate
agg_method='expert'
output_dir='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-'${agg_method}

temperature=0.1

size=0.03125

# model_name="meta-llama/Llama-2-13b-chat-hf"
# model_name="refuel-llm"
model_name="gpt-3.5-turbo"

# word_few_shot_selection="label_diversity_similarity"
word_few_shot_selection="semantic_similarity"
# word_few_shot_selection="fixed"

verify_few_shot_selection="label_diversity_similarity"
# verify_few_shot_selection="semantic_similarity"
# verify_few_shot_selection="fixed"

# tweet_few_shot_selection="label_diversity_similarity"
tweet_few_shot_selection="semantic_similarity"
# tweet_few_shot_selection="fixed"

tweet_verify_few_shot_selection="label_diversity_similarity"
# tweet_verify_few_shot_selection="semantic_similarity"

IFS='/' read -ra parts <<< "$model_name"
# Get the last part
last_model_name="${parts[-1]}"

label_column="sentence_class"

check_mode="strict"
# check_mode="mention"

CoT=True
# CoT=False

# check=True
check=False

# verify=False
verify=True

tweet_verify=False
# tweet_verify=True


use_current_explanation=False
# use_current_explanation=True

# use_ground_explanation=True
use_ground_explanation=False

wordCoT=True
# wordCoT=False

verify_CoT=True
# verify_CoT=False


word_explanation_column="two_step"

word_use_current_explanation=False
word_use_ground_explanation=False

label_version='v1'
task_version='v1'

archive=False
# archive=True

few_shot_num=8

if [[ "$archive" == "True" ]]; then
output_dir=${output_dir}/archive
else
output_dir=${output_dir}
fi

if [[ "$verify_few_shot_selection" == "$word_few_shot_selection" ]]; then
v_tail=''
else
v_tail=${verify_few_shot_selection}'_'
fi

if [[ "$verify_CoT" == True ]]; then
v_head=_aggregated_final_COT_
else
v_head=_aggregated_final_
fi


if [[ "$few_shot_num" == 8 ]]; then
output_dir=${output_dir}/${temperature}/${size}/lv_${label_version}_tv_${task_version}
data_path_par='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-ner-qa-'${agg_method}'/'${temperature}'/'${size}'/lv_'${label_version}'_tv_'${task_version}
else
output_dir=${output_dir}/${temperature}/${size}/lv_${label_version}_tv_${task_version}/few_${few_shot_num}
data_path_par='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-ner-qa-'${agg_method}'/'${temperature}'/'${size}'/lv_'${label_version}'_tv_'${task_version}'/few_'${few_shot_num}
fi

if [[ "$CoT" == "False" || "$use_current_explanation" == "False" ]]; then
    data_path='/scratch/dzhang5/LLM/TWEET-FID/expert.traintestfortest.csv'
    output_dir='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-'${agg_method}
    if [[ "$archive" == "True" ]]; then
    output_dir=${output_dir}/archive
    else
    output_dir=${output_dir}
    fi
    output_dir=${output_dir}/${temperature}/${size}
elif [[ "$verify" == "True" ]]; then
    if [[ "$wordCoT" == "True" ]]; then 
    data_path=${data_path_par}'/'${last_model_name}'_'${check_mode}'_'${word_few_shot_selection}'_COT_'${word_explanation_column}'_cur_'${word_use_current_explanation}'_ground_'${word_use_ground_explanation}${v_head}${v_tail}'expert.traintestfortest.csv'
    else
    data_path=${data_path_par}'/'${last_model_name}'_'${check_mode}'_'${word_few_shot_selection}'_aggregated_final_expert.traintestfortest.csv'
    fi
else
    if [[ "$wordCoT" == "True" ]]; then
    data_path=${data_path_par}'/'${last_model_name}'_'${check_mode}'_'${word_few_shot_selection}'_COT_'${word_explanation_column}'_cur_'${word_use_current_explanation}'_ground_'${word_use_ground_explanation}'_aggregated_expert.traintestfortest.csv'
    else
    data_path=${data_path_par}'/'${last_model_name}'_'${check_mode}'_'${word_few_shot_selection}'_aggregated_expert.traintestfortest.csv'
    fi
fi

output_path=''
# output_path=/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-expert/0.1/1.0/lv_v1_tv_v1/dummy/gpt-3.5-turbo_semantic_similarity_COT_check_gpt-3.5-turbo_strict_semantic_similarity_COT_two_step_cur_False_ground_False_aggregated_final_COT_label_diversity_similarity_expert.traintestfortest.csv

# output_path=/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-all-ae-expert/0.1/1.0/lv_v1_tv_v1/few_8/gpt-3.5-turbo_semantic_similarity_COT_AE_entity_explanation_cur_False_ground_False_expert.traintestfortest.csv

output_path=/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-expert/0.1/1.0/lv_v1_tv_v1/gpt-3.5-turbo_semantic_similarity_COT_check_final_COT_label_diversity_similarity_gpt-3.5-turbo_strict_semantic_similarity_COT_two_step_cur_False_ground_False_aggregated_final_COT_label_diversity_similarity_expert.traintestfortest.csv
echo ${wordCoT}
echo ${data_path}


python compute_classification_performance.py \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --output_path=${output_path} \
    --model_name ${model_name} \
    --few_shot_selection ${tweet_few_shot_selection} \
    --label_column ${label_column} \
    --CoT ${CoT} \
    --check ${check} \
    --tweet_verify ${tweet_verify} \
    --tweet_verify_few_shot_selection ${tweet_verify_few_shot_selection} \
    --use_current_explanation ${use_current_explanation} \
    --use_ground_explanation ${use_ground_explanation} \
    --do_bootstrap True