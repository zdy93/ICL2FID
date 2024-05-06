# ICL2FID
Code for Paper: "LLM-based Two-Level Foodborne Illness Detection Label Annotation with Limited Labeled Samples"

## Requirement
### Language
* Python3 >= 3.9

### Modules
* papermill==2.4.0
* pandas==2.1.4
* matplotlib==3.6.2
* refuel-autolabel==0.0.16
* langchain-core==0.1.23
* transformers==4.38.2
* tqdm==4.64.1
* wandb==0.10.11

## Running
### LLM Annotation Steps
1. run [tweet_autolabel.ipynb](tweet_autolabel.ipynb) to preprocess tweet data
2. run [exp_autolabel_ner_COT_qa_verify.sh](exp_autolabel_ner_COT_qa_verify.sh) for word-level labeling (step 1)
3. run [exp_autolabel_ner_COT_ae_verify_COT_only.sh](exp_autolabel_ner_COT_ae_verify_COT_only.sh) for word-level verification (step 2)
5. run [exp_aggregate_COT_VCOT_entity.sh](exp_aggregate_COT_VCOT_entity.sh) to aggregate all types of word-level labels and generate explanation for post-level labeling
6. run [exp_autolabel_COT_VCOT_COT_check.sh](exp_autolabel_COT_VCOT_COT_check.sh) for post-level labeling (step 3)
7. run [exp_compute_ner_COT_qa_f1.sh](exp_compute_ner_COT_qa_f1.sh) for evaluating word-level labels
8. run [exp_compute_cls_per.sh](exp_compute_cls_per.sh) for evaluating post-level labels

### LLM Configs
* Step 1 few shot method: semantic_similarity
* Step 2 few shot method: label_diversity_similarity
* Step 3 few shot method: semantic_similarity
* temperature: 0.1

### Supervised Learning Steps
1. run roberta_bertweet/exp_[MODEL_NAME]_[multi/sequence/token].sh for two-level/post-level/word-level labeling
2. run roberta_bertweet/exp_compute_[MODEL_NAME]_[multi/sequence/token].sh for evaluating two-level/post-level/word-level labels

### Supervised Learning Configs
* specify roberta/bertweet in exp_roberta_XXX.sh
* use model version without "crf"

Change data path or model name if needed. You can use the [submit_exp.sh](submit_exp.sh) to manage LLM job submissions.

For more about AutoLabel, you are referred to [AutoLabel Introduction](https://docs.refuel.ai/autolabel/introduction/).
