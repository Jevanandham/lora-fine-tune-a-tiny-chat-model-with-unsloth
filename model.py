"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
import unsloth
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # TODO: call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    model, tokeniser = unsloth.FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True
    )

    return (model, tokeniser)

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # TODO: sum p.numel() over every parameter tensor in the module
    sum = 0

    for i in model.parameters():
        sum += i.numel()
    
    return sum

# Step 3 - is_model_4bit_quantized
import bitsandbytes
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    target_class = bitsandbytes.nn.Linear4bit

    modules = torch.nn.Module.modules(model)

    for i in modules:
        if isinstance(i, target_class):
            return True

    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules

    available_adaptors = set(get_lora_target_modules())
    if target_modules is not None:
        available_adaptors = available_adaptors & set(target_modules)
    
    return unsloth.FastLanguageModel.get_peft_model(
        model,
        lora_alpha = lora_alpha,
        target_modules = list(available_adaptors),
        r = r,
    )

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    parameters = 0

    modules = model.parameters()

    for module in modules:
        if module.requires_grad == True:
            parameters += module.numel()

    return parameters

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # TODO: return the fraction of parameters that are trainable.
    return trainable_count / total_count

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # TODO: return a tiny hand-written list of instruction/response example dicts.
    return[
       {
        "instruction" : "Summarize the following sentence in five words or fewer: The quick brown fox jumps over the lazy dog.",
        "response" : "Fox jumps over dog."
       },
       {
        "instruction" : "When a customer requests a refund, first verify their purchase history in the database, confirm the item was bought within 30 days, and then issue the credit to their original payment method",
        "response" : "Received. I will look up their account history, check the 30-day window, and process the refund back to their card."
       },
       {
        "instruction" : "Turn left at the next traffic light.",
        "response" : "Got it, turning left ahead."
       },
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # TODO: combine example['instruction'] and example['response'] into one string
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # TODO: apply format_instruction_example to every example and return the list
    arr = list()

    for ex in examples:
        arr.append(format_instruction_example(ex))

    return arr

# Step 12 - build_text_dataset
import datasets
def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # TODO: return a datasets.Dataset with one 'text' column holding the given strings
    return datasets.Dataset.from_dict({
        'text': texts
    })

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # TODO: call the tokenizer on text and return its input_ids as a plain list
    return tokenizer(text)['input_ids']

# Step 14 - count_tokens (not yet solved)
# TODO: implement

# Step 15 - build_training_arguments (not yet solved)
# TODO: implement

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

