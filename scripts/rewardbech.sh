GPUS="0"
CKPT="../checkpoints/Meta-Llama-3-8B-Instruct"

python eval/rewardbench_sxs.py \
    --model ${CKPT} \
    --force_local \
    --mode ps \
    --gpu ${GPUS} \
    --disable_beaker_save \
    --do_not_save