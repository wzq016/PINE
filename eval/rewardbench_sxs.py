"""Adapted from https://github.com/allenai/reward-bench"""
# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# run a generative RM. For now, this requires openai and anthropic to be installed
# Examples:
# python scripts/run_generative.py --model gpt-3.5-turbo
# python scripts/run_generative.py --model=claude-3-haiku-20240307

# note: for none API models, this script uses vllm
# pip install vllm

import argparse
import logging
import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

import random
import torch
def fix_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

fix_all_seeds(1234)

def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="name of OpenAI model to use (TODO add more providers/models)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Inference mode: ps or origin",
    )
    parser.add_argument("--chat_template", type=str, default="chatgpt", help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--gpu", type=str, help="gpu")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--force_local", action="store_true", default=False, help="force local run, even if model is on Together API"
    )
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


def main():
    args = get_args()
    import torch
    from fastchat.conversation import get_conv_template
    from transformers import AutoTokenizer

    from rewardbench import load_eval_dataset, save_to_hub
    from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
    from rewardbench.generative import (
        ANTHROPIC_MODEL_LIST,
        API_MODEL_LIST,
        OPENAI_MODEL_LIST,
        format_judge_answers,
        process_judgement,
        run_judge_pair,
    )
    from rewardbench.utils import calculate_scores_per_section
    ###############
    # Setup logging
    ###############
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    # load chat template
    conv = get_conv_template("raw")  # not used
    custom_dialogue = True  # to mirror other scripts, required here
    model_type = "Generative RM"

    # if model is list, make type + PoLL and check multiple is odd
    if isinstance(args.model, list):
        model_type += " + PoLL"
        # assert that is odd and > 1
        assert len(args.model) > 1 and len(args.model) % 2 == 1

    # define variable if is API or local
    is_api_models = isinstance(args.model, list) or args.model in API_MODEL_LIST or not args.force_local
    print(is_api_models)
    # if model isn't API, load via vllm
    if not is_api_models:
        # load model
        if "Llama-3" in args.model:
            if args.mode == "origin":
                from transformers import LlamaForCausalLM, LlamaTokenizerFast
                tokenizer = LlamaTokenizerFast.from_pretrained(args.model)
                model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
            else:
                assert args.mode == "ps"
                from transformers import LlamaConfig
                from pine import LlamaForCausalLMWithPS, LlamaTokenizerFastWithPS
                config = LlamaConfig.from_pretrained(args.model)
                config._attn_implementation = 'eager'
                tokenizer = LlamaTokenizerFastWithPS.from_pretrained(args.model)
                model = LlamaForCausalLMWithPS.from_pretrained(args.model, torch_dtype=torch.float16, config=config, device_map="auto")
            model.generation_config.do_sample = False # Greedy Decoding
            model.generation_config.max_new_tokens = 500
            model.generation_config.pad_token_id = 0 # For llama 3
            answer_split_tag = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        elif "Qwen" in args.model:
            if args.mode == "origin":
                raise NotImplementedError
            else:
                assert args.mode == "ps"
                from transformers import Qwen2Config
                from pine import Qwen2ForCausalLMWithPS, Qwen2TokenizerWithPS
                config = Qwen2Config.from_pretrained(args.model)
                config._attn_implementation = 'eager'
                tokenizer = Qwen2TokenizerWithPS.from_pretrained(args.model)
                model = Qwen2ForCausalLMWithPS.from_pretrained(args.model, config=config, torch_dtype=torch.float16, device_map='auto')
                model.generation_config.do_sample = False # Greedy Decoding
                model.generation_config.max_new_tokens = 500
                answer_split_tag = '<|im_end|>\n<|im_start|>assistant'

        else:
            raise NotImplementedError

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        tokenizer=None,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
        max_turns=4,
    )

    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    if is_api_models:
        ############################
        # Run inference via API
        ############################
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()

        def get_judgement(batch, debug=args.debug):
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["text_chosen"][0]["content"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if len(batch["text_chosen"]) <= 4:  # set up only for 1 or 2 turns
                winner, request, judgement = run_judge_pair(
                    prompt, answer_a, answer_b, args.model, multi_turn=mult_turn
                )
                if debug:
                    print(f"Prompt: {request}")
                    print(f"Judgement: {judgement}")

                # handle voting
                if isinstance(winner, list):
                    # print votes if debug
                    if debug:
                        print(winner)
                    winner = max(set(winner), key=winner.count)

                if winner == winner_text:
                    return 1
                elif winner == loser_text:
                    return 0
                else:  # if "error"
                    return 0.5  # effectively a tie
            else:
                return 0.5

        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Map 'my_function' across the vector, executing in parallel using threads
            # results = list(executor.map(get_judgement, dataset))

            # Progress bar version
            results = [None] * len(dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks

            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                # Submit all tasks and hold their futures in a list
                future_to_index = {executor.submit(get_judgement, x): i for i, x in enumerate(dataset)}

                # As tasks complete, update progress and store results in the original order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(dataset))

            # Print newline after progress bar
            print()
    else:
        ############################
        # Run model weights with vllm
        ############################

        def format_judgements(batch):
            # TODO expand this to include fastchat chat templates if needed
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            assert not mult_turn
            prompt = batch["text_chosen"][0]["content"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a
            
            system_prompt, user_prompt = format_judge_answers(prompt, answer_a, answer_b, multi_turn=mult_turn)
            
            if "Llama-3" in args.model:
                assert len(answer_a)==2 and answer_a[-1]['role'] == 'assistant'
                assert len(answer_b)==2 and answer_b[-1]['role'] == 'assistant'
                real_answer_a = answer_a[-1]['content']
                real_answer_b = answer_b[-1]['content']
                template1 = '<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n[User Question]\n {} \n'.format(system_prompt, prompt)
                template21 = "\n[The Start of Assistant A's Answer]\n{}\n[The End of Assistant A's Answer]\n".format(real_answer_a)
                template22 = "\n[The Start of Assistant B's Answer]\n{}\n[The End of Assistant B's Answer]\n".format(real_answer_b)
                template3 = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            elif "Qwen" in args.model:
                assert len(answer_a)==2 and answer_a[-1]['role'] == 'assistant'
                assert len(answer_b)==2 and answer_b[-1]['role'] == 'assistant'
                real_answer_a = answer_a[-1]['content']
                real_answer_b = answer_b[-1]['content']
                template1 = '<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n[User Question]\n {} \n'.format(system_prompt, prompt)
                template21 = "\n[The Start of Assistant A's Answer]\n{}\n[The End of Assistant A's Answer]\n".format(real_answer_a)
                template22 = "\n[The Start of Assistant B's Answer]\n{}\n[The End of Assistant B's Answer]\n".format(real_answer_b)
                template3 = '<|im_end|>\n<|im_start|>assistant\n'
            else:
                raise NotImplementedError

            batch["template1"] = template1
            batch["template21"] = template21
            batch["template22"] = template22
            batch["template3"] = template3
            batch["is_shuffled"] = is_shuffled
            return batch

        # format the dataset for the model
        dataset_prompts = dataset.map(format_judgements)

        # collect texts of dataset in list
        prompts1, prompts21, prompts22, prompts3 = dataset_prompts["template1"], dataset_prompts["template21"], dataset_prompts["template22"], dataset_prompts["template3"]
        is_shuffled = dataset_prompts["is_shuffled"]

        def process_shuffled(win, shuffle):
            if shuffle:
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if win == winner_text:
                return 1
            elif win == loser_text:
                return 0
            else:  # if "error"
                return 0.5  # effectively a tie

        answers = []
        for prompt1, prompt21, prompt22, prompt3 in tqdm(zip(prompts1, prompts21, prompts22, prompts3)):
            if args.mode == "origin":
                prompt = [prompt1 + prompt21 + prompt22 + prompt3]
            elif args.mode == "ps":
                prompt = [prompt1, [prompt21, prompt22], prompt3]
            else:
                raise ValueError('mode incorrect.')

            # generate
            inputs = tokenizer(prompt)
            inputs = {k: torch.tensor(v).to("cuda") for k, v in inputs.items()}
            
            outputs = model.generate(**inputs)
            
            answer = tokenizer.batch_decode(outputs)[0].split(answer_split_tag)[-1]
            answers.append(answer)
            
        winners = [process_judgement(a) for a in answers]

        results = [process_shuffled(w, s) for w, s in zip(winners, is_shuffled)]

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)

    # model name concat if list
    if isinstance(args.model, list):
        model_name = "_".join(args.model)
        model_name = "PoLL/" + model_name
    else:
        model_name = args.model
    # if model in openai or Anthropic list, append org to model name
    if args.model in OPENAI_MODEL_LIST:
        model_name = "openai/" + model_name
    if args.model in ANTHROPIC_MODEL_LIST:
        model_name = "anthropic/" + model_name

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = model_name
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = args.chat_template

    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    # log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)

    ############################
    # Upload results to hub
    #############################
    sub_path = "eval_res/rewardbench/custom/" if not args.pref_sets else "eval_res/rewardbench/pref-sets/"
    results_url = save_to_hub(
        results_grouped,
        args.mode + '_' + model_name.split('/')[-1],
        sub_path,
        args.debug,
        local_only=args.do_not_save,
        save_metrics_for_beaker=not args.disable_beaker_save,
    )
    if not args.do_not_save:
        logger.info(f"Uploaded reward model results to {results_url}")

    logger.info("Not uploading chosen-rejected text with scores due to model compatibility")

    ############################
    # Save per-prompt results to hub
    ############################
    # create new json with scores and upload
    scores_dict = out_dataset.to_dict()
    scores_dict["model"] = model_name
    scores_dict["model_type"] = model_type

    sub_path_scores = "eval_res/rewardbench/custom_scores/" if not args.pref_sets else "eval_res/rewardbench/pref-sets-scores/"

    scores_url = save_to_hub(scores_dict, args.mode + '_' + model_name.split('/')[-1], sub_path_scores, args.debug, local_only=args.do_not_save)
    logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")


if __name__ == "__main__":
    main()
