"""Tokenizer for Llama model with position stacking. ONLY single input is supported."""

import numpy as np
from transformers.models.llama import tokenization_llama, tokenization_llama_fast

def merge_encoded(encoded, append_encoded):
    for k, v in encoded.items():
        v.extend(append_encoded[k])
    return encoded

def ps_call(input_list, call_func):
    """
        There are several assumptions for the input_list:
        1. It is not a batch input. Although this could be implemented, I am lazy..
        2. Not a nested input, i.e., [XX, [XX, XX]] is ok, but [XX, [XX, [XX, XX]]] is not OK. 
            The method is applicable for such an input, but there are no available tasks.
            Therefore no implementation due to my lazyiness.
        3. Only one internal list, i.e., [XX, [XX, XX], XX, [XX, XX]] is not supported.
            Reasons are the same as above.
        4. The first token is not a list. This can always be satisfied since the existence of bos.
            This is not required by the method itself, but for the simplication of implementation.
        Therefore, the current implementation only supports input of format [XX, [XX, XX, ...], XX].
    """
    assert isinstance(input_list, list)
    assert isinstance(input_list[0], str)
    assert isinstance(input_list[1], list)
    assert isinstance(input_list[2], str)

    # Encode the first part
    encoded = call_func(input_list[0], add_special_tokens=True)
    # Position ids for the first part
    position_ids = list(range(len(encoded['input_ids'])))
    
    # Encode the second part
    invariant_text_len = []
    for position_invariant_text in input_list[1]:
        text_encoded = call_func(position_invariant_text, add_special_tokens=False)
        encoded = merge_encoded(encoded, text_encoded)
        invariant_text_len.append(len(text_encoded['input_ids']))
    # Position ids for the second part
    invariant_text_len_sum = sum(invariant_text_len)
    end_position = position_ids[-1] + invariant_text_len_sum + 1
    doc_st_posi = position_ids[-1] + 1
    for text_len in invariant_text_len:
        position_ids.extend(list(range(end_position - text_len, end_position)))
    # doc_range, doc_mask and doc_select compute
    max_doc_len = max(invariant_text_len)
    doc_range = []
    doc_mask = []
    doc_select = []
    self_region_mask = np.zeros([invariant_text_len_sum, invariant_text_len_sum], dtype=np.int32)
    new_doc_posi = 0
    for text_idx, text_len in enumerate(invariant_text_len):
        _doc_range = [0]*max_doc_len
        _doc_range[:text_len] = list(range(doc_st_posi, doc_st_posi + text_len))
        _doc_mask = [0] * max_doc_len
        _doc_mask[:text_len] = [1] * text_len

        doc_range.append(_doc_range)
        doc_mask.append(_doc_mask)
        doc_select.extend([text_idx]*text_len)
        
        self_region_mask[new_doc_posi: new_doc_posi + text_len,
                            new_doc_posi: new_doc_posi + text_len] = 1
        new_doc_posi += text_len
        doc_st_posi += text_len

    # Encode the third part
    text_encoded = call_func(input_list[2], add_special_tokens=False)
    encoded = merge_encoded(encoded, text_encoded)
    end_position = position_ids[-1] + 1
    # Position ids for the third part
    position_ids.extend(list(range(end_position, end_position + len(text_encoded['input_ids']))))

    encoded = {k: [v] for k,v in encoded.items()}
    encoded.update({
        'all_position_ids': [position_ids],
        'doc_range': doc_range,
        'doc_mask': doc_mask,
        'doc_select': doc_select,
        'self_region_mask': self_region_mask
    })

    return encoded


class LlamaTokenizerWithPS(tokenization_llama.LlamaTokenizer):
    def __call__(cls, input_list):
       return ps_call(input_list, super().__call__)

class LlamaTokenizerFastWithPS(tokenization_llama_fast.LlamaTokenizerFast):
    def __call__(cls, input_list):
       return ps_call(input_list, super().__call__)
