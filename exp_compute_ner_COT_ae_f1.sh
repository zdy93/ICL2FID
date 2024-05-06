source /scratch/dzhang5/visa/visa39/bin/activate
agg_method='expert'
data_path='/scratch/dzhang5/LLM/TWEET-FID/expert.traintestfortest.csv'
output_dir='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-all-ae-'${agg_method}

model_name="gpt-3.5-turbo"
# model_name="refuel-llm"
# model_name="meta-llama/Llama-2-13b-chat-hf"

# few_shot_selection="label_diversity_similarity"
few_shot_selection="semantic_similarity"

label_type="all"
# label_type="Food"
# label_type="Location"
# label_type="Symptom"
# label_type="Keyword"

label_symbol="^^^^"
text_column="context"
# text_column="tweet"

AE=True

verified=True
# verified=False

temperature=0.1

size=1.0

CoT=True

verify_CoT=True
# verify_CoT=False

verify_few_shot_selection="label_diversity_similarity"
# verify_few_shot_selection="semantic_similarity"

explanation_column="entity_explanation"

use_current_explanation=False
use_ground_explanation=False

label_version='v1'
task_version='v1'

few_shot_num=8

archive=False
# archive=True

if [[ "$archive" == "True" ]]; then
output_dir=${output_dir}/archive/${temperature}/${size}/lv_${label_version}_tv_${task_version}/few_${few_shot_num}
else
output_dir=${output_dir}/${temperature}/${size}/lv_${label_version}_tv_${task_version}/few_${few_shot_num}
fi

do_bootstrap=False

python compute_f1_qa.py \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --model_name ${model_name} \
    --few_shot_selection ${few_shot_selection} \
    --verify_few_shot_selection ${verify_few_shot_selection} \
    --label_type ${label_type} \
    --label_symbol ${label_symbol} \
    --text_column ${text_column} \
    --verified ${verified} \
    --CoT ${CoT} \
    --AE ${AE} \
    --verify_CoT ${verify_CoT} \
    --explanation_column ${explanation_column} \
    --use_current_explanation ${use_current_explanation} \
    --use_ground_explanation ${use_ground_explanation} \
    --do_bootstrap ${do_bootstrap}