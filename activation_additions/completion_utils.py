""" Functions for generating completions from a model, using a prompt
and a list of ActivationAdditions. """

from functools import wraps
from typing import List, Optional, Dict, Callable, Union

from jaxtyping import Int, Float
import torch as t
import numpy as np
import pandas as pd
import prettytable
import einops

from transformer_lens.HookedTransformer import HookedTransformer, Output

from activation_additions.prompt_utils import ActivationAddition
from activation_additions import hook_utils, logging


def preserve_rng_state(func):
    """Decorator that preserves the `torch` RNG state before and after a
    function call."""

    @wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        # Save the current RNG state
        rng_state: t.Tensor = t.random.get_rng_state()

        # Call the original function
        result = func(*args, **kwargs)

        # Restore the RNG state
        t.random.set_rng_state(rng_state)

        return result

    return wrapper


@preserve_rng_state
@logging.loggable
def gen_using_model(
    model: HookedTransformer,
    prompt_batch: List[str],
    tokens_to_generate: int = 40,
    seed: Optional[int] = None,
    include_logits: bool = False,
    log: Union[bool, Dict] = False,  # pylint: disable=unused-argument
    **sampling_kwargs,
) -> pd.DataFrame:
    """Run `model` on `prompt_batch`
    Returns a `DataFrame` with the completions and losses.

    args:
        `model`: The model to use for completion.

        `prompt_batch`: The prompt batch to use for completion.

        `tokens_to_generate`: The number of additional tokens to generate.

        `seed`: A random seed to use for generation.

        `include_logits`: True to include the full logits tensors as a
        column in the returned DataFrame.

        `log`: To enable logging of this call to wandb, pass either
        True, or a dict contining any of ('tags', 'group', 'notes') to
        pass these keys to the wandb init call.  False to disable
        logging.

        `sampling_kwargs`: Keyword arguments to pass to the model's
        `generate` function.

    returns:
        A `DataFrame` with the completions and losses. The `DataFrame`
        has the following columns:
                `prompts`: The prompts used for completion.
                `completions`: The completions generated by the model.
                `loss`: The loss of the completions.
                `is_modified`: Whether the completion was modified by
                    any hook functions.
    """
    if seed is not None:
        t.manual_seed(seed)

    tokenized_prompts: Int[t.Tensor, "batch pos"] = model.to_tokens(
        prompt_batch
    )
    completions: Float[t.Tensor, "batch pos"] = model.generate(
        input=tokenized_prompts,
        max_new_tokens=tokens_to_generate,
        verbose=False,
        **sampling_kwargs,
    )

    # Compute the loss per token
    if include_logits:
        output: Output = model(
            completions.clone(), return_type="both", loss_per_token=True
        )
        loss, logits = output.loss.detach().cpu(), output.logits.detach().cpu()
    else:
        loss = (
            model(completions.clone(), return_type="loss", loss_per_token=True)
            .detach()
            .cpu()
        )
        logits = None
    average_loss: np.ndarray = einops.reduce(
        loss, "batch pos -> batch", "mean"
    ).numpy()  # NOTE why are we casting to numpy?

    # Remove the <EOS> token and the prompt tokens
    trimmed_completions: Int[t.Tensor, "batch pos"] = completions[
        :, tokenized_prompts.shape[1] :
    ]

    # Put the completions into a DataFrame and return
    results = pd.DataFrame(
        {
            "prompts": prompt_batch,
            "completions": model.to_string(trimmed_completions),
            "loss": list(average_loss),
        }
    )

    if logits is not None:
        results["logits"] = logits.tolist()

    return results


# Ensure that even if we set the seed, we don't change the RNG state globally
@preserve_rng_state
@logging.loggable
def gen_using_hooks(
    model: HookedTransformer,
    prompt_batch: List[str],
    hook_fns: Dict[str, List[Callable]],
    tokens_to_generate: int = 40,
    seed: Optional[int] = None,
    include_logits: bool = False,
    log: Union[bool, Dict] = False,  # pylint: disable=unused-argument
    **sampling_kwargs,
) -> pd.DataFrame:
    """Run `model` using the given `hook_fns`.
    Returns a `DataFrame` with the completions and losses.

    args:
        `model`: The model to use for completion.

        `prompt_batch`: The prompt batch to use for completion.

        `hook_fns`: A dictionary mapping activation names to hooks.

        `tokens_to_generate`: The number of additional tokens to generate.

        `seed`: A random seed to use for generation.

        `include_logits`: True to include the full logits tensors as a
        column in the returned DataFrame.

        `log`: To enable logging of this call to wandb, pass either
        True, or a dict contining any of ('tags', 'group', 'notes') to
        pass these keys to the wandb init call.  False to disable logging.

        `sampling_kwargs`: Keyword arguments to pass to the model's
        `generate` function.

    returns:
        A `DataFrame` with the completions and losses. The `DataFrame`
        has the following columns:
                `prompts`: The prompts used for completion.
                `completions`: The completions generated by the model.
                `loss`: The loss of the completions.
                `is_modified`: Whether the completion was modified by
                    any hook functions.
    """
    # Feels a bit strong to deprecate this, but using gen_using_model is more flexible.
    # warnings.warn("Deprecated: Use `gen_using_model` and `with model.hooks(...)` instead")

    fwd_hooks = [
        (name, hook_fn)
        for name, hook_fns in hook_fns.items()
        for hook_fn in hook_fns
    ]

    for hook in fwd_hooks:
        print("name:", hook[0])

    with model.hooks(fwd_hooks=fwd_hooks):  # type: ignore
        results = gen_using_model(
            model,
            prompt_batch,
            tokens_to_generate,
            seed,
            include_logits,
            log=log,
            **sampling_kwargs,
        )

    # Mark the completions as modified or not
    results["is_modified"] = hook_fns != {}

    return results


@logging.loggable
def gen_using_activation_additions(
    model: HookedTransformer,
    activation_additions: List[ActivationAddition],
    log: Union[bool, Dict] = False,  # pylint: disable=unused-argument
    addition_location: int = 0,
    res_stream_slice: slice = slice(None),
    remove_eos: bool = False,
    spread_coeff: float = 0.0,
    **kwargs,
) -> pd.DataFrame:
    """Generate completions using the given ActivationAdditions.

    args:
        `model`: The model to use for completion.

        `activation_additions`: A list of `ActivationAddition`s to use to create hooks.

        `log`: To enable logging of this call to `wandb`, pass either
        `True`, or a dict contining any of ('tags', 'group', 'notes') to
        pass these keys to the `wandb.init` call. `False` to disable
        logging.

        `addition_location`: An integer representing where in the prompt to add in the act_add

        `res_stream_slice`: A slice specifying which parts of the
        residual stream to add to

        'remove_eos': A boolean specifying whether to remove the EOS token from the beginning
        of the act_add

        `kwargs`: Keyword arguments to pass to `gen_using_hooks`.

    returns:
        A `DataFrame` with the completions and losses. The `DataFrame`
        will have the following columns:
                `prompts`: The prompts used to generate the completions.
                `completions`: The generated completions.
                `loss`: The average loss per token of the completions.
    """
    # Create the hook functions
    
    hook_fns: Dict[str, List[Callable]] = (
        hook_utils.hook_fns_from_activation_additions(
            model=model,
            activation_additions=activation_additions,
            addition_location=addition_location,
            res_stream_slice=res_stream_slice,
            remove_eos=remove_eos
        )
    )

    return gen_using_hooks(model=model, hook_fns=hook_fns, **kwargs)


# Display utils #
def bold_text(text: str) -> str:
    """Returns a string with ANSI bold formatting."""
    return f"\033[1m{text}\033[0m"


def _remove_eos(completion: str) -> str:
    """If completion ends with multiple <|endoftext|> strings, return a
    new string in which all but one are removed."""
    has_eos: bool = completion.endswith("<|endoftext|>")
    new_completion: str = completion.rstrip("<|endoftext|>")
    if has_eos:
        new_completion += "<|endoftext|>"
    return new_completion


def pretty_print_completions(
    results: pd.DataFrame,
    normal_title: str = "Unsteered completions",
    mod_title: str = "Steered completions",
    normal_prompt_override: Optional[str] = None,
    mod_prompt_override: Optional[str] = None,
) -> None:
    """Pretty-print the given completions.

    args:
        `results`: A `DataFrame` with the completions.

        `normal_title`: The title to use for the normal completions.

        `mod_title`: The title to use for the modified completions.

        `normal_prompt_override`: If not `None`, use this prompt for the
            normal completions.

        `mod_prompt_override`: If not `None`, use this prompt for the
            modified completions.
    """
    assert all(
        col in results.columns
        for col in ("prompts", "completions", "is_modified")
    )

    # Assert that an equal number of rows have `is_modified` True and
    # False
    n_rows_mod, n_rows_unmod = [
        len(results[results["is_modified"] == cond]) for cond in [True, False]
    ]
    all_modified: bool = n_rows_unmod == 0
    all_normal: bool = n_rows_mod == 0
    assert all_normal or all_modified or (n_rows_mod == n_rows_unmod), (
        "The number of modified and normal completions must be the same, or we"
        " must be printing all (un)modified completions."
    )

    # Figure out which columns to add
    completion_cols: List[str] = []
    completion_cols += [normal_title] if n_rows_unmod > 0 else []
    completion_cols += [mod_title] if n_rows_mod > 0 else []
    completion_dict: dict = {}
    for col in completion_cols:
        is_mod = col == mod_title
        completion_dict[col] = results[results["is_modified"] == is_mod][
            "completions"
        ]

    # Format the DataFrame for printing
    prompt: str = results["prompts"].tolist()[0]

    # Generate the table
    table = prettytable.PrettyTable()
    table.align = "c"
    table.field_names = map(bold_text, completion_cols)
    table.min_width = table.max_width = 60

    # Separate completions
    table.hrules = prettytable.ALL

    # Put into table
    for row in zip(*completion_dict.values()):
        # Bold the appropriate prompt
        normal_str = bold_text(
            prompt
            if normal_prompt_override is None
            else normal_prompt_override
        )
        mod_str = bold_text(
            prompt if mod_prompt_override is None else mod_prompt_override
        )
        if all_modified:
            new_row = [mod_str + _remove_eos(row[0])]
        elif all_normal:
            new_row = [normal_str + _remove_eos(row[0])]
        else:
            normal_str += _remove_eos(row[0])
            mod_str += _remove_eos(row[1])
            new_row = [normal_str, mod_str]

        table.add_row(new_row)
    print(table)


@logging.loggable
def print_n_comparisons(
    prompt: str,
    model: HookedTransformer,
    num_comparisons: int = 5,
    log: Union[bool, Dict] = False,  # pylint: disable=unused-argument
    activation_additions: Optional[List[ActivationAddition]] = None,
    addition_location: int = 0,
    res_stream_slice: slice = slice(None),
    remove_eos: bool = False,
    spread_coeff: float = 0.0,
    **kwargs,
) -> None:
    """Pretty-print generations from `model` using the appropriate hook
    functions.

    args:
        `prompt`: The prompt to use for completion.

        `model`: The model to use for completion.

        `num_comparisons`: The number of comparisons to make.

        `log`: To enable logging of this call to wandb, pass either
        True, or a dict contining any of ('tags', 'group', 'notes') to
        pass these keys to the wandb init call.  False to disable
        logging.

        `activation_additions`: A list of `ActivationAddition`s to use to create hooks.

        `addition_location`: An int specifying where in the prompt to add the activation addition

        `res_stream_slice`: A slice specifying which activation positions to add
        into the residual stream.

        'remove_eos': A boolean specifying whether to remove the EOS token from the beginning
        of the act_add

        `kwargs`: Keyword arguments to pass to
        `gen_using_hooks`.
    """

    print("activation addition:", activation_additions)
    print("addition location:", addition_location)

    print("remove_eos:", remove_eos)

    assert num_comparisons > 0, "num_comparisons must be positive"

    prompt_batch: List[str] = [prompt] * num_comparisons

    # Generate the completions from the normal model
    normal_df: pd.DataFrame = gen_using_hooks(
        prompt_batch=prompt_batch, model=model, hook_fns={}, **kwargs
    )
    data_frames: List[pd.DataFrame] = [normal_df]

    # Generate the completions from the modified model
    if activation_additions is not None:
        mod_df: pd.DataFrame = gen_using_activation_additions(
            prompt_batch=prompt_batch,
            model=model,
            activation_additions=activation_additions,
            addition_location=addition_location,
            res_stream_slice=res_stream_slice,
            remove_eos=remove_eos,
            spread_coeff=spread_coeff / activation_additions[0].coeff,
            **kwargs,
        )
        data_frames.append(mod_df)

    # Combine the completion results, ensuring that the indices are unique
    results: pd.DataFrame = pd.concat(data_frames, ignore_index=True)

    pretty_print_completions(results=results)
