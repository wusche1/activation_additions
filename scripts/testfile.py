import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from activation_additions import prompt_utils, hook_utils, adhoc_actadds, adhoc_hyperparameter_search
import numpy as np
from contextlib import contextmanager
from typing import Tuple, Callable, Optional

import numpy as np
import torch as t
from torch import nn
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

import accelerate
import importlib

MODEL_DIR: str = "/mnt/ssd-2/mesaoptimizer/llama/hf/7B"
MAX_NEW_TOKENS: int = 50
NUM_CONTINUATIONS: int = 5
SEED: int = 0
DO_SAMPLE: bool = True
TEMPERATURE: float = 1.0
TOP_P: float = 0.9
REP_PENALTY: float = 2.0
PLUS_PROMPT, MINUS_PROMPT = "Harry Potter ", " Lord of the Rings"
CHAT_PROMPT: str = "My favourite book is "
ACT_NUM: int = 6
COEFF: int = 4

sampling_kwargs: dict = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "repetition_penalty": REP_PENALTY,
}

# Set torch and numpy seeds.
t.manual_seed(SEED)
np.random.seed(SEED)


t.set_grad_enabled(False)
# An accelerate wrapper does all the parallelization across devices.
accelerator = accelerate.Accelerator()
model = LlamaForCausalLM.from_pretrained(MODEL_DIR, device_map="cuda:4")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)
model, tokenizer = accelerator.prepare(model, tokenizer)
model.tie_weights()
# model.half()
model.eval()

#Define the kind of behaviour you want: for example: speaking british instead of american english
#a set of [prompt, wanted completion, unwanted completion]

US_UK_dict = {
    "1": ["The delivery driver parked the", " lorry outside the warehouse.", " truck outside the warehouse."],
    "2": ["Can you please turn off the", " tap in the kitchen?", " faucet in the kitchen?"],
    "3": ["I recently rented my first", " flat.", " apartment."],
    "4": ["We are planning to go on a", " holiday to Europe next summer.", " vacation to Europe next summer."],
    "5": ["Would you like a", " biscuit with your tea?", " cookie with your tea?"],
    "6": ["I'm thinking of making a dish with", " aubergine tonight.", " eggplant tonight."],
    "7": ["It's getting chilly; I'll wear my", " jumper today.", " sweater today."],
    "8": ["She bought a bag of salt and vinegar", " crisps from the store.", " chips from the store."],
    "9": ["It's too dark here. Do you have a", " torch?", " flashlight?"],
    "10": ["My kids enjoy playing", " football every weekend.", " soccer every weekend."]
}
prompts=[]
UK_completions=[]
US_completions=[]
for key, example in US_UK_dict.items():
    prompts.append(example[0])
    UK_completions.append(example[1])
    US_completions.append(example[2])

resolution=3
#when you have your steerig vector(here for example: "British" - "American"), do a grid search over layers and coefficents
#this might take a while
weighted_steering_prompts={" British":1," American":-1}
layer_list=[int(i) for i in np.linspace(1,len(model.model.layers)-1,resolution)]
coefficent_list= [i for i in np.linspace(0,10,resolution,dtype=int)]

df=adhoc_hyperparameter_search.layer_coefficient_gridsearch(
    model,
    tokenizer,
    prompts,
    weighted_steering_prompts,
    layer_list,
    coefficent_list,
    UK_completions,
    US_completions
)