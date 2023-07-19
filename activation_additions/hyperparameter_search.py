"""Basic demonstration of sweeps and metrics operation."""

# %%
# Imports, etc.

import numpy as np
from functools import partial
import torch
from typing import List, Dict, Callable, Optional, Union, Tuple
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


from transformer_lens import HookedTransformer

from activation_additions import (
    prompt_utils,
    utils,
    metrics,
    hook_utils
)
from activation_additions.prompt_utils import (
    ActivationAddition,
    pad_tokens_to_match_activation_additions,
    get_block_name,
)
utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)

from typing import List, Union,Dict
import pandas as pd


def add_actads_to_model(model: HookedTransformer,
                        ActAds: List[ActivationAddition]
                        ):
    hook_fns=hook_utils.hook_fns_from_activation_additions(model, ActAds)
    for act_name in hook_fns.keys():
        for hook_fn in hook_fns[act_name]:
            model.add_hook(act_name, hook_fn)
    return
    

def conditional_perplexity(
    model: HookedTransformer,
    prompt_tokens: torch.Tensor,
    completion_tokens: torch.Tensor
) -> float:
    completed_tokens=torch.cat((prompt_tokens, completion_tokens), dim=1)
    metric_func=metrics.get_logprob_metric(model)
    metric=metric_func([completed_tokens])
    completion_logprobs=metric["logprob_actual_next_token"].array[0][-completion_tokens.shape[1]:]
    return -sum(completion_logprobs)


def completion_perplexities(
    model: HookedTransformer,
    prompt_tokens: List[torch.Tensor],
    wanted_completion_tokens: List[torch.Tensor],
    unwanted_completion_tokens: List[torch.Tensor],
    ActAds: List[ActivationAddition]
) -> Tuple[List[float], List[float]]:
    add_actads_to_model(model,ActAds)
    perplexity_on_wanted=[conditional_perplexity(model, prompt, completion) for prompt, completion in zip(prompt_tokens, wanted_completion_tokens)]
    perplexity_on_unwanted=[conditional_perplexity(model, prompt, completion) for prompt, completion in zip(prompt_tokens, unwanted_completion_tokens)]
    model.remove_all_hook_fns()


    return (perplexity_on_wanted, perplexity_on_unwanted)


def layer_coefficient_gridsearch(
    model: HookedTransformer,
    prompts: Union[str, List[str]],
    weighted_steering_prompts: Dict[str, float],
    Layer_list: List[int],
    coefficient_list: List[float],
    wanted_completions: Union[str, List[str]],
    unwanted_completions: Union[str, List[str]],
) -> pd.DataFrame:

    prompt_tokens=[model.to_tokens(prompt)for prompt in prompts]
    wanted_completion_tokens=[model.to_tokens(wanted_completion)[:, 1:] for wanted_completion in wanted_completions]
    unwanted_completion_tokens=[model.to_tokens(unwanted_completion)[:, 1:] for unwanted_completion in unwanted_completions]

    layer_data = []
    coefficient_data = []
    perplexity_wanted_data = []
    perplexity_unwanted_data = []

    for layer in Layer_list:
        for coefficient in coefficient_list:
            ActAds =[prompt_utils.ActivationAddition(
                coeff=prompt_weighting*coefficient,
                act_name=layer,
                prompt=prompt) for prompt, prompt_weighting in weighted_steering_prompts.items()]

            perplexity_on_wanted,perplexity_on_unwanted=completion_perplexities(model,
                            prompt_tokens,
                            wanted_completion_tokens,
                            unwanted_completion_tokens,
                            ActAds)
            
            # Append data for this layer and coefficient to the lists
            layer_data.extend([layer] * len(prompts))
            coefficient_data.extend([coefficient] * len(prompts))
            perplexity_wanted_data.extend(perplexity_on_wanted)
            perplexity_unwanted_data.extend(perplexity_on_unwanted)

    # Create DataFrame
    df = pd.DataFrame({
        "Layer": layer_data,
        "Coefficient": coefficient_data,
        "Perplexity (wanted)": perplexity_wanted_data,
        "Perplexity (unwanted)": perplexity_unwanted_data,
    })

    return df


def create_perplexity_matrices(df):
    # Find unique values for layers and coefficients
    layers = df['Layer'].unique()
    coefficients = df['Coefficient'].unique()

    # Initialize matrices to hold results
    wanted_matrix = np.zeros((len(layers), len(coefficients)))
    unwanted_matrix = np.zeros((len(layers), len(coefficients)))

    # Iterate over all combinations of layer and coefficient
    for i, layer in enumerate(layers):
        for j, coefficient in enumerate(coefficients):
            # Select rows with this combination of layer and coefficient
            rows = df[(df['Layer'] == layer) & (df['Coefficient'] == coefficient)]
            
            # If any rows were found, compute averages and store in matrices
            if not rows.empty:
                wanted_matrix[i, j] = rows['Perplexity (wanted)'].mean()
                unwanted_matrix[i, j] = rows['Perplexity (unwanted)'].mean()

    return wanted_matrix, unwanted_matrix



def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
    # Sort the list in ascending order of Xs
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    pareto_front = [sorted_list[0]]
    
    # Loop through the sorted list
    for pair in sorted_list[1:]:
        if maxY: 
            if pair[1] >= pareto_front[-1][1]: # Look for higher
                pareto_front.append(pair)
        else: 
            if pair[1] <= pareto_front[-1][1]: # Look for lower
                pareto_front.append(pair)
                
    return pareto_front


def plot_pareto_frontier(wanted_matrix, unwanted_matrix):
    # Flatten the matrices
    wanted_points = wanted_matrix.flatten()
    unwanted_points = unwanted_matrix.flatten()

    # Get the Pareto frontier
    pareto_points = pareto_frontier(wanted_points, unwanted_points, maxX = False, maxY = True)

    # Separate X and Y coordinates for plotting
    pareto_X = [point[0] for point in pareto_points]
    pareto_Y = [point[1] for point in pareto_points]

    # Create a mask for the Pareto frontier points
    mask = np.isin(wanted_points, pareto_X) & np.isin(unwanted_points, pareto_Y)

    # Create a linear segmented color map
    cmap = LinearSegmentedColormap.from_list("mycmap", ["yellow", "purple"])

    # Get the corresponding color for each point on the Pareto frontier
    colors = cmap(np.linspace(0, 1, len(pareto_X)))

    # Plot the original points and the Pareto frontier
    fig, axs = plt.subplots(figsize=(7, 7))
    axs.scatter(wanted_points[~mask], unwanted_points[~mask], color='grey', alpha=0.3)
    sc = axs.scatter(pareto_X, pareto_Y, color=colors)
    plt.xlabel('Wanted Perplexity')
    plt.ylabel('Unwanted Perplexity')
    plt.show()

def plot_matrices_with_pareto(df, plot_pareto_points=True):
    wanted_matrix, unwanted_matrix = create_perplexity_matrices(df)
    
    # Extract the layers and coefficients from the dataframe for tick labels
    layers = df['Layer'].unique()
    coefficients = df['Coefficient'].unique()

    fig, axs = plt.subplots(ncols=2, figsize=(14, 6))

    # Plot the wanted matrix
    im1 = axs[0].imshow(wanted_matrix, cmap='Reds')
    axs[0].set_title('Wanted Perplexities')
    fig.colorbar(im1, ax=axs[0])

    # Plot the unwanted matrix
    im2 = axs[1].imshow(unwanted_matrix, cmap='Blues')
    axs[1].set_title('Unwanted Perplexities')
    fig.colorbar(im2, ax=axs[1])

    # Set x and y ticks and labels for both matrices
    for ax in axs:
        ax.set_xticks(np.arange(len(coefficients)))
        ax.set_yticks(np.arange(len(layers)))
        ax.set_xticklabels(coefficients)
        ax.set_yticklabels(layers)
        ax.set_xlabel('Coefficient')
        ax.set_ylabel('Layer')

    if plot_pareto_points:
        # Compute the Pareto frontier points
        wanted_points = wanted_matrix.flatten()
        unwanted_points = unwanted_matrix.flatten()
        pareto_points = pareto_frontier(wanted_points, unwanted_points, maxX=False, maxY=True)

        pareto_X_vals = [point[0] for point in pareto_points]
        pareto_Y_vals = [point[1] for point in pareto_points]

        # Find matrix indices of the Pareto points
        pareto_indices = []

        # Create a linear segmented color map
        cmap = LinearSegmentedColormap.from_list("mycmap", ["yellow", "purple"])

        # Get the corresponding color for each point on the Pareto frontier
        colors = cmap(np.linspace(0, 1, len(pareto_X_vals)))

        for x, y in zip(pareto_X_vals, pareto_Y_vals):
            index = np.where((wanted_matrix == x) & (unwanted_matrix == y))
            pareto_indices.append(index)

        pareto_X_indices = [index[1][0] for index in pareto_indices]
        pareto_Y_indices = [index[0][0] for index in pareto_indices]

        # Overlay the Pareto frontier on both matrices
        for ax in axs:
            ax.scatter(pareto_X_indices, pareto_Y_indices, color=colors, s=50, label="Pareto Frontier")
            ax.legend()

    plt.tight_layout()
    plt.show()