import torch
class SteeringVector:
    def __init__(self):
        self.vector: List[Dict[str, object]] = []

    def add_entry(
        self,
        tokenizer,
        prompt: str,
        layer: int,
        coefficient: float,
        sub_stream: str="residual_stream",
        location: int=0,
        spread_coeff: float=0,
        remove_EOS: bool=True,
    ):
        entry = {
            "prompt": prompt,
            "tokens": tokenizer(prompt, return_tensors="pt")["input_ids"],
            "layer": layer,
            "sub_stream": sub_stream,
            "coefficient": coefficient,
            "location": location,
            "spread_coeff": spread_coeff,
            "remove_EOS": remove_EOS,
        }
        self.vector.append(entry)
    def get_activations(self,model):
        for entry in self.vector:
            activations=get_residual_activation(model, entry["tokens"], entry["layer"])
            entry["activations"]=activations
        return 

    def __repr__(self):
        return str(self.vector)

def tensor_addition_with_padding(tensor1, tensor2):
    # Get the sequence lengths (2nd dimension)
    seq_len1 = tensor1.size(1)
    seq_len2 = tensor2.size(1)
    
    # Determine the max sequence length
    max_seq_len = max(seq_len1, seq_len2)
    
    # Initialize tensors with the max size and pad with zeros
    padded_tensor1 = torch.zeros(tensor1.size(0), max_seq_len, tensor1.size(2))
    padded_tensor2 = torch.zeros(tensor2.size(0), max_seq_len, tensor2.size(2))
    
    # Copy the original tensors into the padded tensors
    padded_tensor1[:, :seq_len1, :] = tensor1
    padded_tensor2[:, :seq_len2, :] = tensor2

    # Add the tensors
    result = padded_tensor1 + padded_tensor2
    return result
def forward(model, input_ids: torch.LongTensor, steering_vec=None) -> torch.Tensor:
    # Determine the batch size and sequence length from the input_ids
    batch_size, seq_length = input_ids.shape

    # Generate position ids
    device = input_ids.device
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    # Convert input_ids to embeddings
    inputs_embeds = model.embed_tokens(input_ids)

    # Generate a full attention mask
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=inputs_embeds.device)

    # The past_key_values_length is set to 0, as we aren't using past_key_values
    past_key_values_length = 0
    attention_mask = model._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)

    hidden_states = inputs_embeds

    addition_idx_list=[]
    addition_activations_list=[]
    if steering_vec is not None:
        for entry in steering_vec.vector:
            addition_idx_list.append(entry["layer"])
            addition_activations_list.append(entry["activations"])

    # Go through the model's layers
    for idx, decoder_layer in enumerate(model.layers):
        layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = layer_outputs[0]
        if idx in addition_idx_list:
            addition_idx=addition_idx_list.index(idx)
            addition_activations=addition_activations_list[addition_idx]
            #addition_activations=addition_activations.unsqueeze(0).repeat(batch_size,1,1)
            hidden_states=tensor_addition_with_padding(hidden_states,addition_activations)

    hidden_states = model.norm(hidden_states)

    return hidden_states

def get_residual_activation(model, input_ids: torch.LongTensor, layer: int) -> torch.Tensor:
    # Determine the batch size and sequence length from the input_ids
    batch_size, seq_length = input_ids.shape

    # Generate position ids
    device = input_ids.device
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    # Convert input_ids to embeddings
    inputs_embeds = model.embed_tokens(input_ids)

    # Generate a full attention mask
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=inputs_embeds.device)

    # The past_key_values_length is set to 0, as we aren't using past_key_values
    past_key_values_length = 0
    attention_mask = model._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)

    hidden_states = inputs_embeds

    # Go through the model's layers
    for idx, decoder_layer in enumerate(model.layers):
        layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = layer_outputs[0]
        if idx == layer:
            return hidden_states

    print("ERROR: Layer not found")

    return 