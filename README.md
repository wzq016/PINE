# Eliminating Position Bias of Language Models: A Mechanistic Approach

This is the official repo of the paper "[Eliminating Position Bias of Language Models: A Mechanistic Approach](https://arxiv.org/abs/2407.01100)".

The paper proposes a novel method (PINE) to eliminate the position bias in LLMs from the angle of transformers' computation flows.

![image](./method.png)

## Requirements
transformers==4.40.0

torch=2.3

[RewardBench==0.1.1](https://github.com/allenai/reward-bench) (Only if you want to run RewardBench, the code is only tested on v0.1.1)

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
# For example, X1 is the system prompt for the LM-as-a-judge task, X2, X3, X4 are three options,
# X5 is the <eos> token.
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

* No batch inputs

* No vllm support

* Slow Inference due to 
  * 'for' loop in the implementation
  * No support to advanced operations such as FlashAttention, SdpaAttention.

* [Minor] A bit high memory cost.

## When should I use PINE?

As the discussion in the paper, PINE is useful when 

* The base model is capable, since position bias is not always a bad phenomenon when the task is too hard for models.

* The task has a position bias (which is the meaning of this paper!)

## Reproducing results in the paper

### RewardBench
```bash scripts/rewardbech_sxs.sh```
Remember to change gpu and ckpt config in the bash files before running.

### Lost-in-the-middle
We use and process the data provided [here](https://github.com/nelson-liu/lost-in-the-middle), and the processed data is located in ```./eval_data```

To test on the lost-in-the-middle task:
```bash scripts/lost_in_the_middle.sh```
Remember to change gpu, ckpt config and evaluation files in the bash files before running.

Unlike RewardBench that only have 2 options, the lost-in-the-middle tasks have 10 or 20 options, therefore it is harder for models. We suggest using capable models when running this experiment, otherwise you will observe a performance drop as the above section and the paper suggest.

## Citation
If you find the repo and paper helpful for you, please cite our paper with the following format:

```
@misc{wang2024eliminatingpositionbiaslanguage,
      title={Eliminating Position Bias of Language Models: A Mechanistic Approach}, 
      author={Ziqi Wang and Hanlin Zhang and Xiner Li and Kuan-Hao Huang and Chi Han and Shuiwang Ji and Sham M. Kakade and Hao Peng and Heng Ji},
      year={2024},
      eprint={2407.01100},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01100}, 
}
```

## Misc
If you have any suggestions, find any mistakes, or want further discussion on the paper or the repo, feel free to utilize the issues, PR, or directly contact XY@illinois.edu, where X=ziqiw, Y=9
