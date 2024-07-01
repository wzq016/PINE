# Eliminating Position Bias of Language Models: A Mechanistic Approach

This is the official repo of the paper "Eliminating Position Bias of Language Models: A Mechanistic Approach".

The paper proposes a novel method (PINE) to eliminate the position bias in LLMs from the angle of transformers' computation flows.

## Requirements
transformers==4.40.0

torch=2.3

[RewardBench](https://github.com/allenai/reward-bench) (Only if you want to run RewardBench)

## Usage
First, use ```pip install -e .``` to install this repo as a python package.

Then, the usage is similar to huggingface transformers:

```
from transformers import LlamaConfig
from pine import LlamaForCausalLMWithPS, LlamaTokenizerFastWithPS

# Initialize tokenizer and model
config = LlamaConfig.from_pretrained(model_name)
config._attn_implementation = 'eager'
tokenizer = LlamaTokenizerFastWithPS.from_pretrained(model_name)
model = LlamaForCausalLMWithPS.from_pretrained(model_name, torch_dtype=torch.float16, config=config, device_map="auto")
model.generation_config.do_sample = False # Suppose we sue Greedy Decoding
model.generation_config.max_new_tokens = 500
model.generation_config.pad_token_id = 128004 # For llama 3

# Suppose [X1 | X2 | X3 | X4 | X5] is the input string
# Xi is the substring, and we want X2, X3, X4 to be treated equally w/o position bias,
# For example, X1 is the system prompt for the LM-as-a-judge task, X2, X3, X4 are three options, X5 is the <eos> token.
# Then we reformat the input to be the following:
input_string = [
    X1,
    [
        X2,
        X3,
        X4
    ],
    X5
]

inputs = tokenizer(input_string)
inputs = {k: torch.tensor(v).to("cuda") for k,v in inputs.items()}
outputs = model.generate(**inputs)
```
## Implementation Efficiency
Since I am not really good at writing efficient implementations, and efficiency is not the core topic of the paper, the current implementation is slow:

(1) No batch inputs

(2) No vllm support

(3) Slow Inference due to the for loop in the implementation

(4) [Minor] A bit high memory cost.

## When should I use PINE?

As the discussion in the paper, PINE is useful when 

(1) The base model is capable, since position bias is not always a bad phenomenon when the task is too hard for models.

(2) The task has a position bias (which is the meaning of this paper!)

## Reproducing results in the paper

### RewardBench
```bash scripts/rewardbech_sxs.sh```
Remember to change gpu and ckpt config in the bash files before running.

### Lost-in-the-middle
We use and process the data provided [here](https://github.com/nelson-liu/lost-in-the-middle), and the processed data is located in ```./eval_data```

To test on the lost-in-the-middle task:
```bash scripts/rewardbech_sxs.sh```
Remember to change gpu, ckpt config and evaluation files in the bash files before running.

Unlike RewardBench that only have 2 options, the lost-in-the-middle tasks have 10 or 20 options, therefore it is harder for models. We suggest using capable models when running this experiment, otherwise you will observe a performance drop as the above section and the paper suggest.

## Citation
If you find the repo and paper helpful for you, please cite our paper with the following format:

```

```
(Will Update when the paper is annouced by Arxiv.)

## Misc
If you have any suggestions, find any mistakes, or want futhur discussion on the paper or the repo, feel free to utilize the issues, PR or directly contact XY@illinois.edu, where X=ziqiw, Y=9