"""
Funcitons are helpful for doing a search over the hyperparameters 
of Steering vector injection location and strength. The results can
also be plotted in a way, that makes it easy to pick out the best
set of hyperparameters.
"""


import numpy as np
import torch as t
from functools import partial
import torch
from typing import List, Dict, Callable, Optional, Union, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn.functional as F

from activation_additions import prompt_utils, utils, metrics, adhoc_actadds_2
from activation_additions.adhoc_actadds import SteeringVector
from activation_additions.prompt_utils import (
    ActivationAddition,
    pad_tokens_to_match_activation_additions,
    get_block_name,
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)

from typing import List, Union, Dict
import pandas as pd


def conditional_perplexity(
    model,
    tokenizer,
    steering_vec: SteeringVector,
    prompt_tokens: torch.Tensor,
    completion_tokens: torch.Tensor
) -> float:
    completed_tokens = t.cat((prompt_tokens, completion_tokens), dim=1)

    logits = adhoc_actadds_2.forward(
        model,  steering_vec
    )

    completed_tokens = completed_tokens[0, 1:]
    logits = logits[0, :-1]

    # Convert logits to log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    # Extract log probabilities of actual tokens
    token_log_probs = log_probs.gather(1, completed_tokens.unsqueeze(-1)).squeeze()

    # Summing the log probabilities of the completion tokens
    log_prob_sum = token_log_probs[-completion_tokens.size(1) :].sum().item()
    return -log_prob_sum


def completion_perplexities(
    model,
    tokenizer,
    steering_vec: SteeringVector,
    prompt_tokens: List[torch.Tensor],
    wanted_completion_tokens: List[torch.Tensor],
    unwanted_completion_tokens: List[torch.Tensor],
) -> Tuple[List[float], List[float]]:
    """
    Computes the conditional perplexities of the wanted and unwanted completion tokens given the prompt tokens for a given model.

    Args:
        model (HookedTransformer): The model used to compute log probabilities.
        prompt_tokens (List[torch.Tensor]): List of tensors of tokens representing the prompt/context for the completion.
        wanted_completion_tokens (List[torch.Tensor]): List of tensors of tokens representing the wanted completion text.
        unwanted_completion_tokens (List[torch.Tensor]): List of tensors of tokens representing the unwanted completion text.
        ActAds (List[ActivationAddition]): List of activation vectors to be added to the model.
    Returns:
        Tuple[List[float], List[float]]: Tuple of lists of floats representing the perplexity of the wanted and unwanted completions.
    """

    perplexity_on_wanted = [
        conditional_perplexity(
            model, tokenizer, steering_vec, prompt, completion
        )
        for prompt, completion in zip(prompt_tokens, wanted_completion_tokens)
    ]
    perplexity_on_unwanted = [
        conditional_perplexity(
            model, tokenizer, steering_vec, prompt, completion
        )
        for prompt, completion in zip(prompt_tokens, unwanted_completion_tokens)
    ]

    return (perplexity_on_wanted, perplexity_on_unwanted)


def layer_coefficient_gridsearch(
    model,
    tokenizer,
    prompts: Union[str, List[str]],
    weighted_steering_prompts: Dict[str, float],
    Layer_list: List[int],
    coefficient_list: List[float],
    wanted_completions: Union[str, List[str]],
    unwanted_completions: Union[str, List[str]],
) -> pd.DataFrame:
    """
    Performs a grid search over specified layers and coefficients to steer model's behavior.

    This function conducts a grid search over the provided layers and coefficients to determine
    the impact on perplexity when using weighted steering prompts. It returns a DataFrame with the
    delta in perplexity for both wanted and unwanted completions.

    Args:
        model (HookedTransformer): The model used for token conversion and perplexity computation.
        prompts (Union[str, List[str]]): Prompt(s) used for completion.
        weighted_steering_prompts (Dict[str, float]): Dictionary of steering prompts and their weights.
        Layer_list (List[int]): List of layers to explore in the grid search.
        coefficient_list (List[float]): List of coefficients to explore in the grid search.
        wanted_completions (Union[str, List[str]]): List of completions that are desired.
        unwanted_completions (Union[str, List[str]]): List of completions that are not desired.

    Returns:
        pd.DataFrame: A DataFrame containing the layer, coefficient, and delta perplexities for
                      both wanted and unwanted completions.

    Example:
        result_df = layer_coefficient_gridsearch(
            model_instance, ["What is the capital"], {"of France": 1.5}, [8, 10], [0.1, 0.2],
            ["Paris"], ["Berlin"]
        )

    Notes:
        - The function computes the change in perplexity (delta perplexity) by comparing the perplexity
          after applying steering to the initial perplexity.
        - The DataFrame returned contains the columns: "Layer", "Coefficient", "Delta Perplexity (wanted)",
          and "Delta Perplexity (unwanted)".
    """
    prompt_tokens = [
        tokenizer(prompt, return_tensors="pt")["input_ids"] for prompt in prompts
    ]
    wanted_completion_tokens = [
        tokenizer(wanted_completion, return_tensors="pt")["input_ids"][:, 1:]
        for wanted_completion in wanted_completions
    ]
    unwanted_completion_tokens = [
        tokenizer(unwanted_completion, return_tensors="pt")["input_ids"][:, 1:]
        for unwanted_completion in unwanted_completions
    ]

    layer_data = []
    coefficient_data = []
    perplexity_wanted_data = []
    perplexity_unwanted_data = []
    steering_vec = adhoc_actadds.SteeringVector()

    prior_perplexity_on_wanted, prior_perplexity_on_unwanted = completion_perplexities(
        model,
        tokenizer,
        steering_vec,
        prompt_tokens,
        wanted_completion_tokens,
        unwanted_completion_tokens,
    )

    for layer in Layer_list:
        for coefficient in coefficient_list:
            steering_vec = adhoc_actadds.SteeringVector()
            for prompt, prompt_weighting in weighted_steering_prompts.items():
                coeff = coefficient * prompt_weighting
                steering_vec.add_entry(prompt, layer, "sub_stream", coeff, 0, 0, True)

            perplexity_on_wanted, perplexity_on_unwanted = completion_perplexities(
                model,
                tokenizer,
                steering_vec,
                prompt_tokens,
                wanted_completion_tokens,
                unwanted_completion_tokens,
            )

            # Append data for this layer and coefficient to the lists
            layer_data.extend([layer] * len(prompts))
            coefficient_data.extend([coefficient] * len(prompts))
            delta_perplexity_on_wanted = [
                post - prior
                for post, prior in zip(perplexity_on_wanted, prior_perplexity_on_wanted)
            ]
            delta_perplexity_on_unwanted = [
                post - prior
                for post, prior in zip(
                    perplexity_on_unwanted, prior_perplexity_on_unwanted
                )
            ]

            perplexity_wanted_data.extend(delta_perplexity_on_wanted)
            perplexity_unwanted_data.extend(delta_perplexity_on_unwanted)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Layer": layer_data,
            "Coefficient": coefficient_data,
            "Delta Perplexity (wanted)": perplexity_wanted_data,
            "Delta Perplexity (unwanted)": perplexity_unwanted_data,
        }
    )

    return df


def create_perplexity_matrices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate matrices for average delta perplexities for wanted and unwanted completions.

    Given a DataFrame with layer, coefficient, and delta perplexities for both wanted and
    unwanted completions, this function creates two matrices. Each matrix's rows correspond
    to unique layers, and columns correspond to unique coefficients. The value in each cell
    represents the average delta perplexity for the corresponding layer and coefficient
    combination.

    Args:
        df (pd.DataFrame): DataFrame containing columns "Layer", "Coefficient",
                           "Delta Perplexity (wanted)", and "Delta Perplexity (unwanted)".

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - wanted_matrix (np.ndarray): Matrix with average delta perplexities for wanted completions.
            - unwanted_matrix (np.ndarray): Matrix with average delta perplexities for unwanted completions.

    Example:
        df = pd.DataFrame({
            'Layer': [1, 2, 1, 2],
            'Coefficient': [0.1, 0.1, 0.2, 0.2],
            'Delta Perplexity (wanted)': [1, 2, 3, 4],
            'Delta Perplexity (unwanted)': [2, 3, 1, 3]
        })
        wanted, unwanted = create_perplexity_matrices(df)

    Notes:
        - The matrices returned are filled with the average delta perplexity values for each combination
          of layer and coefficient. If no data exists for a particular combination, the corresponding
          cell in the matrix will be zero.
    """
    # Find unique values for layers and coefficients
    layers = df["Layer"].unique()
    coefficients = df["Coefficient"].unique()

    # Initialize matrices to hold results
    wanted_matrix = np.zeros((len(layers), len(coefficients)))
    unwanted_matrix = np.zeros((len(layers), len(coefficients)))

    # Iterate over all combinations of layer and coefficient
    for i, layer in enumerate(layers):
        for j, coefficient in enumerate(coefficients):
            # Select rows with this combination of layer and coefficient
            rows = df[(df["Layer"] == layer) & (df["Coefficient"] == coefficient)]

            # If any rows were found, compute averages and store in matrices
            if not rows.empty:
                wanted_matrix[i, j] = rows["Delta Perplexity (wanted)"].mean()
                unwanted_matrix[i, j] = rows["Delta Perplexity (unwanted)"].mean()

    return wanted_matrix, unwanted_matrix


def pareto_frontier(
    Xs: List[float], Ys: List[float], maxX: bool = True, maxY: bool = True
) -> List[Tuple[float, float]]:
    """
    Compute the Pareto frontier given a set of points.

    The Pareto frontier represents the set of points that are not dominated by any other points in
    a multi-objective optimization context. For two objectives, one point dominates another if it is
    better or equal in both objectives.

    Args:
        Xs (List[float]): List of x-coordinates of the points.
        Ys (List[float]): List of y-coordinates of the points.
        maxX (bool, optional): If True, the function will look for larger X values as better.
                               Defaults to True.
        maxY (bool, optional): If True, the function will look for larger Y values as better.
                               Defaults to True.

    Returns:
        List[Tuple[float, float]]: List of points on the Pareto frontier, where each point is represented
                                   as a tuple of (x, y).

    Example:
        Xs = [1, 2, 3, 4]
        Ys = [4, 3, 2, 1]
        frontier = pareto_frontier(Xs, Ys)
        # Expected result: [(4, 1), (3, 2)]

    Notes:
        - This function assumes that the input lists `Xs` and `Ys` are of the same length.
        - Points are sorted based on `maxX` and then checked for Pareto efficiency based on `maxY`.
        - The function currently operates in O(n log n) time due to sorting. There exist more efficient
          algorithms for computing the Pareto frontier, especially for larger datasets.
    """
    # Sort the list in ascending order of Xs
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    pareto_front = [sorted_list[0]]

    # Loop through the sorted list
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:  # Look for higher
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:  # Look for lower
                pareto_front.append(pair)

    return pareto_front


def plot_pareto_frontier(df: pd.DataFrame):
    """
    Visualize the Pareto frontier using the provided perplexity data.

    This function plots a scatter plot of delta wanted vs. delta unwanted perplexities. The Pareto frontier
    points are highlighted and the rest of the points are shown in a lighter shade. Hovering over the Pareto
    points reveals the associated layer and coefficient values.

    Args:
        df (pd.DataFrame): DataFrame containing columns "Layer", "Coefficient", "Delta Perplexity (wanted)",
                           and "Delta Perplexity (unwanted)".

    Returns:
        None: The function directly displays a Plotly figure and does not return any value.

    Example:
        df = pd.DataFrame({
            'Layer': [1, 2, 1, 2],
            'Coefficient': [0.1, 0.1, 0.2, 0.2],
            'Delta Perplexity (wanted)': [1, 2, 3, 4],
            'Delta Perplexity (unwanted)': [2, 3, 1, 3]
        })
        plot_pareto_frontier(df)

    Notes:
        - Assumes that `create_perplexity_matrices` and `pareto_frontier` functions are available in the same context.
        - Requires `plotly` library for visualization.
    """
    wanted_matrix, unwanted_matrix = create_perplexity_matrices(df)
    layers = df["Layer"].unique()
    coefficients = df["Coefficient"].unique()

    # Flatten the matrices
    wanted_points = wanted_matrix.flatten()
    unwanted_points = unwanted_matrix.flatten()

    # Get the Pareto frontier
    pareto_points = pareto_frontier(
        wanted_points, unwanted_points, maxX=False, maxY=True
    )

    # Separate X and Y coordinates for plotting
    pareto_X = [point[0] for point in pareto_points]
    pareto_Y = [point[1] for point in pareto_points]

    # Find matrix indices of the Pareto points
    pareto_indices = []
    for x, y in zip(pareto_X, pareto_Y):
        index = np.where((wanted_matrix == x) & (unwanted_matrix == y))
        pareto_indices.append(index)

    # Convert indices to hover text (strings)
    hover_texts = [
        "Layer: "
        + str(layers[idx[0][0]])
        + "\n Coefficent"
        + str(coefficients[idx[1][0]])
        for idx in pareto_indices
    ]

    # Create a mask for the Pareto frontier points
    mask = np.isin(wanted_points, pareto_X) & np.isin(unwanted_points, pareto_Y)

    # Define custom color scale and interpolate the colors
    colorscale = [[0, "yellow"], [1, "purple"]]
    colors = [
        color_interpolation(colorscale, c) for c in np.linspace(0, 1, len(pareto_X))
    ]

    # Create Plotly figure
    fig = go.Figure()

    # Add the original scatter points
    fig.add_trace(
        go.Scatter(
            x=wanted_points[~mask],
            y=unwanted_points[~mask],
            mode="markers",
            marker=dict(color="grey", opacity=0.3),
            hoverinfo="text",
            name="Other points",
        )
    )

    # Add the Pareto frontier points with hover texts
    fig.add_trace(
        go.Scatter(
            x=pareto_X,
            y=pareto_Y,
            mode="markers",
            marker=dict(color=colors),
            hovertext=hover_texts,
            hoverinfo="text",
            name="Pareto Frontier",
        )
    )

    # Label axes
    fig.update_layout(
        xaxis_title="Delta Wanted Perplexity",
        yaxis_title="Delta Unwanted Perplexity",
        title="Pareto Frontier",
    )

    # Show figure
    fig.show()


def color_interpolation(colorscale: List[List[Union[float, str]]], value: float) -> str:
    """
    Compute the interpolated color based on a given value and colorscale.

    Given a colorscale consisting of two points and a value between 0 and 1,
    this function returns an interpolated color. The interpolation is linear
    between the RGB components of the colors defined at the extremes of the colorscale.

    Args:
        colorscale (List[List[Union[float, str]]]): A list of two lists. Each inner list
            contains a float (either 0 or 1) and a color string (either a recognized name
            like "yellow" or a hex value like "#FFFF00").
        value (float): The interpolation value. Must be in the range [0, 1]. A value of
            0 will return the first color in the colorscale, and a value of 1 will
            return the second color.

    Returns:
        str: The interpolated color in rgb string format, e.g., "rgb(255,255,0)".

    Raises:
        KeyError: If a color name provided in the colorscale is not recognized.

    Example:
        >>> color_interpolation([[0, "yellow"], [1, "purple"]], 0.5)
        'rgb(191,127,64)'

    Notes:
        - Currently, the function only supports colors "yellow" and "purple" as named colors.
          If other named colors are used, it will raise a KeyError unless those names are added
          to the `color_name_to_hex` inner function.
        - The colorscale is currently limited to two points. For more complex color interpolations,
          the function would need enhancements.
    """

    # Convert color names to their RGB hex values
    def color_name_to_hex(color_name):
        colors = {
            "yellow": "#FFFF00",
            "purple": "#800080",
        }
        return colors.get(color_name, color_name)

    lower_color, upper_color = colorscale[0][1], colorscale[1][1]

    # Convert color names to hex if they are not already
    lower_color = color_name_to_hex(lower_color)
    upper_color = color_name_to_hex(upper_color)

    # Convert hex to RGB values
    r1, g1, b1 = [int(lower_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    r2, g2, b2 = [int(upper_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]

    # Calculate the interpolated RGB values
    r = r1 + value * (r2 - r1)
    g = g1 + value * (g2 - g1)
    b = b1 + value * (b2 - b1)

    return f"rgb({int(r)},{int(g)},{int(b)})"


def plot_matrices_with_pareto(
    df: pd.DataFrame, plot_pareto_points: bool = True
) -> None:
    """
    Visualize Wanted and Unwanted Perplexities matrices with optional Pareto points.

    This function creates a side-by-side heatmap of 'Wanted' and 'Unwanted' perplexities
    derived from a given dataframe. Optionally, Pareto points can be overlaid on these heatmaps
    to show optimal trade-offs.

    Args:
        df (pd.DataFrame): A dataframe with columns "Layer", "Coefficient",
            "Delta Perplexity (wanted)", and "Delta Perplexity (unwanted)".
        plot_pareto_points (bool, optional): A flag to determine whether to overlay Pareto
            points on the heatmaps. Defaults to True.

    Returns:
        None: The function displays the plots but does not return any values.

    Notes:
        - This function assumes that the dataframe provided has been appropriately pre-processed
          and contains the relevant columns.
        - The function will display the heatmaps in a side-by-side fashion, with an optional
          overlay of Pareto points if `plot_pareto_points` is set to True. The Pareto points are
          color-coded based on a linear interpolation between yellow and purple.
        - The plot requires Plotly's `go` and `make_subplots` functions for visualization.

    Example:
        >>> df = pd.DataFrame({...})  # Some example data
        >>> plot_matrices_with_pareto(df, True)  # Display heatmaps with Pareto points
    """
    wanted_matrix, unwanted_matrix = create_perplexity_matrices(df)

    # Extract the layers and coefficients from the dataframe for tick labels
    layers = df["Layer"].unique()
    coefficients = df["Coefficient"].unique()

    pareto_X_indices = []
    pareto_Y_indices = []
    colors = []
    if plot_pareto_points:
        wanted_points = wanted_matrix.flatten()
        unwanted_points = unwanted_matrix.flatten()
        pareto_points = pareto_frontier(
            wanted_points, unwanted_points, maxX=False, maxY=True
        )
        pareto_X_vals = [point[0] for point in pareto_points]
        pareto_Y_vals = [point[1] for point in pareto_points]

        for i, (x, y) in enumerate(zip(pareto_X_vals, pareto_Y_vals)):
            index = np.where((wanted_matrix == x) & (unwanted_matrix == y))
            pareto_X_indices.append(index[1][0])
            pareto_Y_indices.append(index[0][0])
        colorscale = [[0, "yellow"], [1, "purple"]]
        colors = np.linspace(0, 1, len(pareto_X_indices))
        color_values = [color_interpolation(colorscale, c) for c in colors]

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Wanted Perplexities", "Unwanted Perplexities")
    )

    # Add heatmap for Wanted Perplexities
    fig.add_trace(
        go.Heatmap(
            z=wanted_matrix,
            x=coefficients,
            y=layers,
            colorscale="Reds",
            colorbar=dict(len=0.5, yanchor="top", y=0.5, title="Wanted"),
            showscale=True,
        ),
        row=1,
        col=1,
    )

    # Add heatmap for Unwanted Perplexities
    fig.add_trace(
        go.Heatmap(
            z=unwanted_matrix,
            x=coefficients,
            y=layers,
            colorscale="Blues",
            colorbar=dict(len=0.5, yanchor="top", y=1, title="Unwanted"),
            showscale=True,
        ),
        row=1,
        col=2,
    )

    # Overlay the Pareto frontier points on each matrix
    if plot_pareto_points:
        for i, (x_index, y_index) in enumerate(zip(pareto_X_indices, pareto_Y_indices)):
            fig.add_trace(
                go.Scatter(
                    x=[coefficients[x_index]],
                    y=[layers[y_index]],
                    mode="markers",
                    marker=dict(color=color_values[i]),  # Use the color value directly
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[coefficients[x_index]],
                    y=[layers[y_index]],
                    mode="markers",
                    marker=dict(color=color_values[i]),  # Use the color value directly
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    # Update x and y axis titles
    fig.update_xaxes(title_text="Coefficient", row=1, col=1)
    fig.update_xaxes(title_text="Coefficient", row=1, col=2)
    fig.update_yaxes(title_text="Layer", row=1, col=1)
    fig.update_yaxes(title_text="Layer", row=1, col=2)

    fig.update_layout(margin=dict(t=20, b=50, l=50, r=50))

    fig.show()
