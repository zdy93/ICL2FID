source /scratch/dzhang5/visa/visa39/bin/activate
agg_method='expert'
data_path='/scratch/dzhang5/LLM/TWEET-FID/expert.traintestfortest.csv'
output_dir='/scratch/dzhang5/LLM/TWEET-FID/traintestfortest-results-autolabel-ner-qa-'${agg_method}
# model_name="gpt-3.5-turbo"
# model_name="refuel-llm"
model_name="meta-llama/Llama-2-13b-chat-hf"
# few_shot_selection="label_diversity_similarity"
few_shot_selection="semantic_similarity"
label_type="all"
# label_type="Food"
label_symbol="^^^^"
text_column="context"
# text_column="tweet"
verified=True
# verified=False
temperature=0.1
size=1.0
label_version='v1'
task_version='v1'
output_dir=${output_dir}/${temperature}/${size}/lv_${label_version}_tv_${task_version}

python compute_f1_qa.py \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --model_name ${model_name} \
    --few_shot_selection ${few_shot_selection} \
    --label_type ${label_type} \
    --label_symbol ${label_symbol} \
    --text_column ${text_column} \
    --verified ${verified}