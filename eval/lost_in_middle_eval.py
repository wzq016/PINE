"""
This file is to eval lost-in-the-middle tasks. 

Evaluation function, prompts and data are borrowed from: https://github.com/nelson-liu/lost-in-the-middle
"""
import os

import re
import json
from tqdm import tqdm
import argparse

import string
from typing import List
import regex

from vllm import LLM, SamplingParams

# The login function is borrowed from RewardBench
# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def template_adjust(inputs, model_name):
    if 'Llama-3' in model_name:
        start_head = '<|start_header_id|>user<|end_header_id|>\n\n'
        end_head = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    elif 'Qwen' in model_name:
        start_head = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        end_head = '<|im_end|>\n<|im_start|>assistant\n'
    else:
        raise NotImplementedError
    
    if isinstance(inputs, str):
        return inputs.replace('[INST]', start_head).replace('[/INST]', end_head)
    else:
        inputs[0] = inputs[0].replace('[INST]', start_head)
        inputs[2] = inputs[2].replace('[/INST]', end_head)
        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='4,5,6,7')
    parser.add_argument('--eval_num', type=int, default=30)
    parser.add_argument('--mode', type=str, choices=['ps', 'origin'])
    parser.add_argument('--eval_data', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    import torch
    from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast, LlamaConfig
    from pine import LlamaForCausalLMWithPS, LlamaTokenizerWithPS, LlamaTokenizerFastWithPS
    
    
    data_path = './eval_data/{}/'.format(args.eval_data) + args.filename
    dump_path = './eval_res/{}/'.format(args.eval_data)
    dump_prefix='ps_' if args.mode=='ps' else 'origin_'
    model_name = args.model_name
    dump_prefix = model_name.split('/')[-1] + '_' + dump_prefix

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    dump_path = dump_path + dump_prefix + args.filename
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if 'Llama-3' in model_name:
        if args.mode == 'origin':
            num_gpus = len(args.gpu.split(','))
            model = LLM(model_name, trust_remote_code=False, tensor_parallel_size=num_gpus, max_model_len=6144)
            stop_token_ids = [128009]
            sampling_params = SamplingParams(
                n=1,
                temperature=0,
                top_p=1,
                max_tokens=500,
                stop_token_ids=stop_token_ids,
            )
        else:
            config = LlamaConfig.from_pretrained(model_name)
            config._attn_implementation = 'eager'
            tokenizer = LlamaTokenizerFastWithPS.from_pretrained(model_name)
            model = LlamaForCausalLMWithPS.from_pretrained(model_name, torch_dtype=torch.float16, config=config, device_map="auto")
            model.generation_config.do_sample = False # Greedy Decoding
            model.generation_config.max_new_tokens = 500
            model.generation_config.pad_token_id = 128004 # For llama 3
        generation_split = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
    elif 'Qwen' in model_name:
        if args.mode == 'origin':
            num_gpus = len(args.gpu.split(','))
            model = LLM(model_name, trust_remote_code=False, tensor_parallel_size=num_gpus, max_model_len=6144)
            stop_token_ids = []
            sampling_params = SamplingParams(
                n=1,
                temperature=0,
                top_p=1,
                max_tokens=500,
                stop_token_ids=stop_token_ids,
            )
        else:
            from transformers import Qwen2Config
            from pine import Qwen2ForCausalLMWithPS, Qwen2TokenizerWithPS
            config = Qwen2Config.from_pretrained(model_name)
            config._attn_implementation = 'eager'
            tokenizer = Qwen2TokenizerWithPS.from_pretrained(model_name)
            model = Qwen2ForCausalLMWithPS.from_pretrained(model_name, torch_dtype=torch.float16, config=config, device_map="auto")
            model.generation_config.do_sample = False # Greedy Decoding
            model.generation_config.max_new_tokens = 500
        generation_split = '<|im_start|>assistant'

    if args.mode == 'origin':
        prompts = [template_adjust(d[0], model_name) for d in data[:args.eval_num]]
        outputs = model.generate(prompts, sampling_params)
        answers = [o.outputs[0].text for o in outputs]
        writer = open(dump_path, 'a+')
        score = 0
        idx = 0
        for answer, gt in zip(answers, [d[2] for d in data[:args.eval_num]]):
            answer = answer.strip()
            score += best_subspan_em(answer, gt)
            writer.write(json.dumps({
                'prompt': prompts[idx],
                'model_ans': answer,
                'gt_ans': gt

            }, indent=4))
            writer.write('\n')
            idx+=1

        print('Score: ', score/len(answers))
    else:
        tot = 0
        score = 0  
        writer = open(dump_path, 'a+')
        for d in tqdm(data[:args.eval_num]):
            tot+=1
            origin_prompt = d[1]
            assert len(origin_prompt)==4
            origin_prompt = [origin_prompt[0], origin_prompt[1], origin_prompt[2] + origin_prompt[3]]
            origin_prompt = template_adjust(origin_prompt, model_name)
            inputs = tokenizer(origin_prompt)
            inputs = {k: torch.tensor(v).to("cuda") for k,v in inputs.items()}
            outputs = model.generate(**inputs)
            
            output_texts = tokenizer.batch_decode(outputs)[0].split(generation_split)[1].strip()
            answer = output_texts.strip()
            score += best_subspan_em(answer, d[2])

            
            writer.write(json.dumps({
                'prompt': origin_prompt,
                'outputs': output_texts,
                'model_ans': answer,
                'gt_ans': d[2]

            }, indent=4))
            writer.write('\n')

            print('Score: ', score/tot)
            
        writer.close()


if __name__=="__main__":
    main()