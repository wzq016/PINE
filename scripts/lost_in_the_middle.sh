GPUS="0,1,2,3"
FILENAME="nq-open-10_total_documents_gold_at_4_random.json"
MODEL_NAME="meta-llama/Meta-Llama-3-70B-Instruct"

python eval/lost_in_middle_eval.py  \
    --eval_data lost_in_middle \
    --filename ${FILENAME} \
    --eval_num 3 \
    --gpu ${GPUS} \
    --mode ps \
    --model_name ${MODEL_NAME} \