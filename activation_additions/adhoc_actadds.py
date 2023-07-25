import torch
from contextlib import contextmanager
from typing import Tuple, Callable, Optional

import numpy as np
import torch as t
from torch import nn
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import accelerate
from typing import List, Dict
accelerator = accelerate.Accelerator()

class SteeringVector:
    def __init__(self):
        self.vector: List[Dict[str, object]] = []

    def add_entry(self, prompt: str, layer: int, coefficient: float):
        entry = {"prompt": prompt, "layer": layer, "coefficient": coefficient}
        self.vector.append(entry)

    def __repr__(self):
        return str(self.vector)


# %%
# Declare hooking types.
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
Hook = Tuple[nn.Module, PreHookFn]
Hooks = list[Hook]


def tokenize(tokenizer,text: str) -> dict[str, t.Tensor]:
    """Tokenize prompts onto the appropriate devices."""
    tokens = tokenizer(text, return_tensors="pt")
    tokens = accelerator.prepare(tokens)
    return tokens

def gen_without_steering(model,
    tokenizer,
    prompt,
    NUM_CONTINUATIONS=5,
    DO_SAMPLE=True,
    MAX_NEW_TOKENS=100,
    sampling_kwargs: dict = {
        "temperature": 1,
        "top_p": 0.9,
        "repetition_penalty":  2.0,
    },
):
        
    base_tokens = accelerator.unwrap_model(
        model.generate(
            **tokenize(tokenizer,[prompt] * NUM_CONTINUATIONS),
            generation_config=GenerationConfig(
                **sampling_kwargs,
                do_sample=DO_SAMPLE,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
            ),
        )
    )
    base_strings = [tokenizer.decode(o) for o in base_tokens]
    return base_strings
def gen_with_steering(
    model,
    tokenizer,
    sentence_prompt,
    steering_vec_dict,
    NUM_CONTINUATIONS=5,
    DO_SAMPLE=True,
    MAX_NEW_TOKENS=100,
    sampling_kwargs: dict = {
        "temperature": 1,
        "top_p": 0.9,
        "repetition_penalty":  2.0,
    },
):

    activations = []

    # Check whether all layers are the same
    unique_layers = set(item["layer"] for item in steering_vec_dict)

    if len(unique_layers) > 1:
        raise ValueError("All layers in steering_vec_dict must be the same.")

    # Extract the common layer value
    layer = unique_layers.pop()

    for steering_vector_dict in steering_vec_dict:
        # Extract values from the dictionary
        prompt = steering_vector_dict["prompt"]
        coeff = steering_vector_dict["coefficient"]
        print(f"Prompt: {prompt}, Coefficient: {coeff}")

        # Compute activations using the given prompt and layer
        activations.append(get_resid_pre(model,tokenizer,prompt, layer) * coeff)
    # Return the sum of all activations
    activations = resize_tensors(*activations)
    steering_vec = sum(activations)  # %%

    # Run the model with the steering vector * COEFF.
    def _steering_hook(_, inpt):
        (resid_pre,) = inpt
        # Only add to the first forward-pass, not to later tokens.
        if resid_pre.shape[1] == 1:
            # Caching in `model.generate` for new tokens.
            return
        ppos, apos = resid_pre.shape[1], steering_vec.shape[1]
        assert (
            apos <= ppos
        ), f"More modified streams ({apos}) than prompt streams ({ppos})!"
        resid_pre[:, :apos, :] += steering_vec

    layer_name = get_blocks(model)[layer]
    with pre_hooks(hooks=[(layer_name, _steering_hook)]):
        steered_tokens = accelerator.unwrap_model(
            model.generate(
                **tokenize(tokenizer,[sentence_prompt] * NUM_CONTINUATIONS),
                generation_config=GenerationConfig(
                    **sampling_kwargs,
                    do_sample=DO_SAMPLE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=tokenizer.eos_token_id,
                ),
            )
        )
    steered_strings = [tokenizer.decode(o) for o in steered_tokens]
    return steered_strings


def resize_tensors(*tensors):
    """
    Adjusts the tensors in the list to match the size of the largest tensor.
    Smaller tensors are padded with zeros.
    """
    # Assuming all tensors have the same number of dimensions
    # and differ only in one of those dimensions (for simplicity, let's take the second dimension as in the original function)
    max_size = max(tensor.size(1) for tensor in tensors)

    resized_tensors = []
    for tensor in tensors:
        current_size = tensor.size(1)

        # If the tensor size matches the maximum, just append it as is
        if current_size == max_size:
            resized_tensors.append(tensor)
        else:
            # If tensor is smaller, pad it
            padding_size = max_size - current_size
            padding = torch.zeros(
                (tensor.size(0), padding_size, *tensor.size()[2:]),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            resized_tensor = torch.cat([tensor, padding], dim=1)
            resized_tensors.append(resized_tensor)

    return tuple(resized_tensors)


# %%
# Hooking functionality.
@contextmanager
def pre_hooks(hooks: Hooks):
    """Register pre-forward hooks with torch."""
    handles = []
    try:
        handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
        yield
    finally:
        for handle in handles:
            handle.remove()


def get_blocks(mod):
    """Get the blocks of a model."""
    if isinstance(mod, LlamaForCausalLM):
        return mod.model.layers
    raise ValueError(f"Unsupported model type: {type(mod)}.")


@contextmanager
def residual_stream(mod: LlamaForCausalLM, layers: Optional[list[int]] = None):
    """Actually build hooks for a model."""
    # TODO Plausibly could be replaced by "output_hidden_states=True" in model call.
    modded_streams = [None] * len(get_blocks(mod))

    # Factory function that builds the initial hooks.
    def _make_helper_hook(i):
        def _helper_hook(_, current_inputs):
            modded_streams[i] = current_inputs[0]

        return _helper_hook

    hooks = [
        (layer, _make_helper_hook(i))
        for i, layer in enumerate(get_blocks(mod))
        if i in layers
    ]
    # Register the hooks.
    with pre_hooks(hooks):
        yield modded_streams


def get_resid_pre(model,tokenizer,prompt: str, layer_num: int):
    """Get residual stream activations for a prompt, just before a layer."""
    # TODO: Automatic addition padding.
    with residual_stream(model, layers=[layer_num]) as unmodified_streams:
        model(**tokenize(tokenizer,prompt))
    return unmodified_streams[layer_num]
