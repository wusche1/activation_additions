from contextlib import contextmanager
from typing import Tuple, Callable, Optional
import numpy as np
import einops
import torch as t
from torch import nn
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import accelerate
from typing import List, Dict

accelerator = accelerate.Accelerator()


class SteeringVector:
    def __init__(self):
        self.vector: List[Dict[str, object]] = []

    def add_entry(
        self,
        prompt: str,
        layer: int,
        sub_stream: str,
        coefficient: float,
        location: int=0,
        spread_coeff: float=0,
        remove_EOS: bool=True,
    ):
        entry = {
            "prompt": prompt,
            "layer": layer,
            "coefficient": coefficient,
            "sub_stream": sub_stream,
            "location": location,
            "spread_coeff": spread_coeff,
            "remove_EOS": remove_EOS,
        }
        self.vector.append(entry)

    def __repr__(self):
        return str(self.vector)


# Declare hooking types.
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
Hook = Tuple[nn.Module, PreHookFn]
Hooks = list[Hook]


def tokenize(tokenizer, text: str) -> dict[str, t.Tensor]:
    """Tokenize prompts onto the appropriate devices."""
    tokens = tokenizer(text, return_tensors="pt")
    tokens = accelerator.prepare(tokens)
    return tokens


def gen_without_steering(
    model,
    tokenizer,
    prompt,
    NUM_CONTINUATIONS=1,
    DO_SAMPLE=True,
    MAX_NEW_TOKENS=100,
    sampling_kwargs: dict = {
        "temperature": 1,
        "top_p": 0.9,
        "repetition_penalty": 2.0,
    },
):
    base_tokens = accelerator.unwrap_model(
        model.generate(
            **tokenize(tokenizer, [prompt] * NUM_CONTINUATIONS),
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


# TODO: make this work
def resize_strings(string_list, tokenizer):
    tokens_list = [tokenizer.encode(string) for string in string_list]
    max_len = max(len(tokens) for tokens in tokens_list)
    for string in string_list:
        diff = max_len - len(tokenizer.encode(string))
        diff_pad = " " * diff
        string += diff_pad
    return string_list


def forward_pass_with_hooks(model, tokenizer, tokens, steering_vec_dict):
    accelerator.prepare(tokens)
    if len(steering_vec_dict.vector) == 0:
        return accelerator.unwrap_model(model.forward(tokens)["logits"])
    with residual_stream(model, layers=[0]) as unmodified_streams:
        model(tokens)
    prompt_vector = unmodified_streams[0]

    activations = []
    # Check whether all layers are the same
    unique_layers = set(item["layer"] for item in steering_vec_dict.vector)
    if len(unique_layers) > 1:
        raise ValueError("All layers in steering_vec_dict must be the same.")
    # Extract the common layer value
    layer = unique_layers.pop()
    string_list = [
        steering_vector["prompt"] for steering_vector in steering_vec_dict.vector
    ]

    for steering_vector_dict in steering_vec_dict.vector:
        # Extract values from the dictionary
        prompt = steering_vector_dict["prompt"]
        coeff = steering_vector_dict["coefficient"]
        location = steering_vector_dict["location"]
        spread_coeff = steering_vector_dict["spread_coeff"]
        remove_EOS = steering_vector_dict["remove_EOS"]
        base_vector = get_resid_pre(model, tokenizer, prompt, layer)
        if remove_EOS:
            base_vector = base_vector[:, 1:, :]
        base_vector_length = base_vector.shape[1]
        zeros_vector = t.zeros_like(prompt_vector)
        base_vector = base_vector.to(prompt_vector.device)
        zeros_vector = zeros_vector.to(prompt_vector.device)
        assert (
            base_vector_length + location < zeros_vector.shape[1]
        ), "The injection extends past the end of the prompt"
        zeros_vector[:, location : location + base_vector_length, :] += (
            base_vector * coeff
        )
        times_to_pad_right = (
            zeros_vector.shape[1] - (location + base_vector_length)
        ) // base_vector_length
        times_to_pad_left = location // base_vector_length
        right_vec = einops.repeat(
            base_vector,
            "batch tokens features -> batch (tokens times_to_pad_right) features",
            times_to_pad_right=times_to_pad_right,
        )
        left_vec = einops.repeat(
            base_vector,
            "batch tokens features -> batch (tokens times_to_pad_left) features",
            times_to_pad_left=times_to_pad_left,
        )
        right_vec = right_vec.to(prompt_vector.device)
        left_vec = left_vec.to(prompt_vector.device)
        zeros_vector[:, location - left_vec.shape[1] : location, :] += (
            left_vec * spread_coeff
        )
        zeros_vector[
            :,
            location
            + base_vector_length : location
            + base_vector_length
            + right_vec.shape[1],
            :,
        ] += (
            right_vec * spread_coeff
        )
        activations.append(zeros_vector)
        steering_vec = sum(activations)

    # Return the sum of all activations
    # Run the model with the steering vector * COEFF.
    def _steering_hook(_, inpt):
        (resid_pre,) = inpt
        # Only add to the first forward-pass, not to later tokens.
        if resid_pre.shape[1] == 1:
            # Caching in `model.generate` for new tokens.
            return
        resid_pre = resid_pre.to(steering_vec.device)
        resid_pre += steering_vec

    layer_name = get_blocks(model)[layer]
    with pre_hooks(hooks=[(layer_name, _steering_hook)]):
        return_logits = accelerator.unwrap_model(model.forward(tokens)["logits"])
    return return_logits


def gen_with_steering(
    model,
    tokenizer,
    sentence_prompt,
    steering_vec_dict,
    NUM_CONTINUATIONS=1,
    DO_SAMPLE=True,
    MAX_NEW_TOKENS=100,
    sampling_kwargs: dict = {
        "temperature": 1,
        "top_p": 0.9,
        "repetition_penalty": 2.0,
    },
):
    prompt_tokens = tokenizer.encode(sentence_prompt)
    prompt_vector = get_resid_pre(model, tokenizer, sentence_prompt, 0)
    activations = []
    # Check whether all layers are the same
    unique_layers = set(item["layer"] for item in steering_vec_dict.vector)
    if len(unique_layers) > 1:
        raise ValueError("All layers in steering_vec_dict must be the same.")
    # Extract the common layer value
    layer = unique_layers.pop()
    string_list = [
        steering_vector["prompt"] for steering_vector in steering_vec_dict.vector
    ]
    string_list = resize_strings(string_list, tokenizer)
    for n, steering_vector in enumerate(steering_vec_dict.vector):
        steering_vector["prompt"] = string_list[n]
    for steering_vector_dict in steering_vec_dict.vector:
        # Extract values from the dictionary
        prompt = steering_vector_dict["prompt"]
        coeff = steering_vector_dict["coefficient"]
        location = steering_vector_dict["location"]
        spread_coeff = steering_vector_dict["spread_coeff"]
        remove_EOS = steering_vector_dict["remove_EOS"]
        base_vector = get_resid_pre(model, tokenizer, prompt, layer)
        if remove_EOS:
            base_vector = base_vector[:, 1:, :]
        base_vector_length = base_vector.shape[1]
        zeros_vector = t.zeros_like(prompt_vector)
        base_vector = base_vector.to(prompt_vector.device)
        zeros_vector = zeros_vector.to(prompt_vector.device)
        assert (
            base_vector_length + location < zeros_vector.shape[1]
        ), "The injection extends past the end of the prompt"
        zeros_vector[:, location : location + base_vector_length, :] += (
            base_vector * coeff
        )
        times_to_pad_right = (
            zeros_vector.shape[1] - (location + base_vector_length)
        ) // base_vector_length
        times_to_pad_left = location // base_vector_length
        right_vec = einops.repeat(
            base_vector,
            "batch tokens features -> batch (tokens times_to_pad_right) features",
            times_to_pad_right=times_to_pad_right,
        )
        left_vec = einops.repeat(
            base_vector,
            "batch tokens features -> batch (tokens times_to_pad_left) features",
            times_to_pad_left=times_to_pad_left,
        )
        right_vec = right_vec.to(prompt_vector.device)
        left_vec = left_vec.to(prompt_vector.device)
        zeros_vector[:, location - left_vec.shape[1] : location, :] += (
            left_vec * spread_coeff
        )
        zeros_vector[
            :,
            location
            + base_vector_length : location
            + base_vector_length
            + right_vec.shape[1],
            :,
        ] += (
            right_vec * spread_coeff
        )
        activations.append(zeros_vector)
        steering_vec = sum(activations)

    # Return the sum of all activations
    # Run the model with the steering vector * COEFF.
    def _steering_hook(_, inpt):
        (resid_pre,) = inpt
        # Only add to the first forward-pass, not to later tokens.
        if resid_pre.shape[1] == 1:
            # Caching in `model.generate` for new tokens.
            return
        resid_pre = resid_pre.to(steering_vec.device)
        resid_pre +=steering_vec

    layer_name = get_blocks(model)[layer]
    with pre_hooks(hooks=[(layer_name, _steering_hook)]):
        steered_tokens = accelerator.unwrap_model(
            model.generate(
                **tokenize(tokenizer, [sentence_prompt] * NUM_CONTINUATIONS),
                generation_config=GenerationConfig(
                    **sampling_kwargs,
                    do_sample=DO_SAMPLE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=tokenizer.eos_token_id,
                ),
            )
        )

        # model.forward(
        # **tokenize(tokenizer, [sentence_prompt] * NUM_CONTINUATIONS)
        # )["logits"])

    # steered_strings = [tokenizer.decode(o) for o in steered_tokens]
    return steered_tokens


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


def get_resid_pre(model, tokenizer, prompt: str, layer_num: int):
    """Get residual stream activations for a prompt, just before a layer."""
    # TODO: Automatic addition padding.
    with residual_stream(model, layers=[layer_num]) as unmodified_streams:
        tokens=tokenize(tokenizer, prompt).to(model.device)
        model(**tokens)
    return unmodified_streams[layer_num]
