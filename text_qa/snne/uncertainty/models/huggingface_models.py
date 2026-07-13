"""Implement HuggingfaceModel models."""
import copy
import logging
from collections import Counter

import torch
import accelerate
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from PIL import Image

from snne.uncertainty.models.base_model import BaseModel, STOP_SEQUENCES


LIST_SUPPORT_MODELS = ['llama', 'falcon', 'mistral', 'phi', 'gemma', 'qwen', 'deepseek']


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token (supports batched generation)."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None, batch_size=1):
        super().__init__()
        self.stops = [s for s in stops if s is not None]
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        self.batch_size = batch_size
        self.finished = [False] * batch_size
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        current_batch_size = input_ids.shape[0]
        if len(self.finished) != current_batch_size:
            self.finished = [False] * current_batch_size

        active_indices = [i for i, f in enumerate(self.finished) if not f]
        if not active_indices:
            return True

        if self.match_on == 'text':
            active_tokens = input_ids[active_indices, self.initial_length:]
            generations = self.tokenizer.batch_decode(active_tokens, skip_special_tokens=False)
            for idx, generation in zip(active_indices, generations):
                if any(stop in generation for stop in self.stops):
                    self.finished[idx] = True
        elif self.match_on == 'tokens':
            for idx in active_indices:
                if any(stop in input_ids[idx][-len(stop):] for stop in self.stops):
                    self.finished[idx] = True
        else:
            raise ValueError(f"Unknown match_on type: {self.match_on}")

        return all(self.finished)


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer.
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # Remove split for that layer.
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None, token_limit=4096):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
            
        eightbit = False
        fourbit = False
        
        if model_name.endswith('-8bit'):
            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}
            model_name = model_name[:-len('-8bit')]
            eightbit = True
        elif model_name.endswith('-4bit'):
            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,)}
            model_name = model_name[:-len('-4bit')]
            fourbit = True
        else:
            kwargs = {}

        if "checkpoint" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, device_map="auto",
                token_type_ids=None)
            config = LoraConfig.from_pretrained(model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, torch_dtype="auto", device_map="auto",
                max_memory={0: '80GIB'})
            # resize embeddings if needed (e.g. for LlamaTokenizer)
            embedding_size = base_model.get_input_embeddings().weight.shape[0]
            if len(self.tokenizer) > embedding_size:
                base_model.resize_token_embeddings(len(self.tokenizer))
            self.model = PeftModel.from_pretrained(
                base_model, model_name, device_map="auto")
            print(f"Load finetuned model from {model_name} successfully!")
        elif 'deepseek' in model_name.lower():
            model_id = f'deepseek-ai/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map="auto",
                token_type_ids=None)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto",
                torch_dtype=torch.bfloat16, trust_remote_code=True, **kwargs,)
        elif 'llama' in model_name.lower():
            if 'Llama-2' in model_name:
                base = 'meta-llama'
                model_name = model_name + '-hf'
            elif 'Llama-3' in model_name:
                base = 'meta-llama'
            else:
                base = 'huggyllama'

            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{model_name}", device_map="auto",
                token_type_ids=None)

            llama65b = '65b' in model_name.lower() and base == 'huggyllama'
            llama_large = '70b' in model_name.lower() and base == 'meta-llama'

            if llama_large and (eightbit or fourbit):
                # Quantized large model — fits on single GPU
                logging.info(f'Loading quantized large Llama model on single GPU: {model_name}')
                # self.tokenizer = AutoTokenizer.from_pretrained(f"{base}/{model_name}")
                # if self.tokenizer.pad_token is None:
                #     self.tokenizer.pad_token = self.tokenizer.eos_token
                # self.tokenizer.padding_side = "left"   # required for left-padded batch generation

                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}", device_map="auto",
                    torch_dtype=torch.bfloat16, trust_remote_code=True, quantization_config=kwargs.get('quantization_config'))

            elif llama_large or llama65b:
                path = snapshot_download(
                    repo_id=f'{base}/{model_name}',
                    allow_patterns=['*.json', '*.model', '*.safetensors'],
                    ignore_patterns=['pytorch_model.bin.index.json']
                )
                config = AutoConfig.from_pretrained(f"{base}/{model_name}")
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config)
                self.model.tie_weights()
                max_mem = 15 * 4686198491

                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype='float16'
                )
                device_map = remove_split_layer(device_map)
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0

                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model, path, device_map=full_model_device_map,
                    dtype='float16', skip_keys='past_key_values')
            else:
                # Small Llama models (7B, 8B, 13B, etc.) — fit on single GPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}", device_map="auto",
                    torch_dtype=torch.bfloat16, trust_remote_code=True, **kwargs,)

        elif 'mistral' in model_name.lower():
            model_id = f'mistralai/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto',
                clean_up_tokenization_spaces=False)

            try: 
                import accelerate
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map='auto',
                    max_memory={0: '80GIB'},
                    torch_dtype="auto",
                **kwargs,
            )
            except (ImportError, ValueError) as e:
                logging.warning(f"Could not use device_map='auto', loading to cuda manually: {e}")
                logging.warning(f"Loading model to cuda manually", torch.cuda.is_available())
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
        elif 'falcon' in model_name:
            model_id = f'tiiuae/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype="auto",
                **kwargs,
            )
        elif 'phi' in model_name.lower():
            model_id = f'microsoft/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map="auto",
                token_type_ids=None)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype="auto", 
                max_memory={0: '80GIB'}, **kwargs,)
        elif 'gemma' in model_name.lower():
            model_id = f'google/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map="auto",
                token_type_ids=None)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map='auto',
                torch_dtype=torch.bfloat16, trust_remote_code=True, **kwargs,
            )

        elif 'qwen' in model_name.lower():
            model_id = f'Qwen/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map="auto",
                token_type_ids=None)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto",
                torch_dtype=torch.bfloat16, trust_remote_code=True, **kwargs,)
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = token_limit
        # Setting padding for open-ended generation
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        print(f"Setting pad_token_id to {self.tokenizer.eos_token_id}")

    def predict(self, input_data, temperature, min_p=0.0, return_full=False, return_logits=False):

        # Implement prediction.
        if 'mistral' in self.model_name.lower():
            inputs = self.tokenizer(input_data, return_tensors="pt", return_token_type_ids=False).to("cuda")
        else:
            inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        # with torch.no_grad():
        #     outputs = self.model.generate(
        #         **inputs,
        #         max_new_tokens=self.max_new_tokens, 
        #         return_dict_in_generate=True,
        #         output_scores=True,
        #         output_hidden_states=True,
        #         temperature=temperature,
        #         min_p=min_p,
        #         do_sample=True,
        #         stopping_criteria=stopping_criteria,
        #     )
        with torch.no_grad():
            if temperature == 0.0:
                # Deterministic generation
                logging.info(f'Deterministic generation called, do_sample=False, temperature={temperature}')
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    temperature=temperature,
                    # min_p=min_p,
                    # top_p=0.95,
                    do_sample=False,
                    stopping_criteria=stopping_criteria,
                )
            else:
                logging.info(f'Stochastic generation called, do_sample=True, temperature={temperature}')
                # Sampling with temperature
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    temperature=temperature,
                    # min_p=min_p,
                    top_p=0.95,
                    do_sample=True,
                    stopping_criteria=stopping_criteria,
                )

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.startswith(input_data) or 'Llama-3' in self.model_name:
            # Llama-3 auto fix some typo in the input
            input_data_offset = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        else:
            raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            earliest_idx = len(sliced_answer)
            for stop in self.stop_sequences:
                if stop is not None and stop in sliced_answer:
                    idx = sliced_answer.find(stop)
                    if idx < earliest_idx:
                        earliest_idx = idx
            if earliest_idx < len(sliced_answer):
                sliced_answer = sliced_answer[:earliest_idx]
                # else:
                #     logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()

        # Get the number of tokens until the stop word comes up.
        # Note: Indexing with `stop_at` already excludes the stop_token.
        # Note: It's important we do this with full answer, since there might be
        # non-trivial interactions between the input_data and generated part
        # in tokenization (particularly around whitespaces.)
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated <= 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1

        # Get the last hidden state (last layer) and the last token's embedding of the answer.
        # Note: We do not want this to be the stop token.

        # outputs.hidden_state is a tuple of len = n_generated_tokens.
        # The first hidden state is for the input tokens and is of shape
        #     (n_layers) x (batch_size, input_size, hidden_size).
        # (Note this includes the first generated token!)
        # The remaining hidden states are for the remaining generated tokens and is of shape
        #    (n_layers) x (batch_size, 1, hidden_size).

        # Note: The output embeddings have the shape (batch_size, generated_length, hidden_size).
        # We do not get embeddings for input_data! We thus subtract the n_tokens_in_input from
        # token_stop_index to arrive at the right output.

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 0:
            embedding_size = self.model.get_input_embeddings().weight.shape[1]
            logging.warning(f"Generate {sliced_answer}. Hidden size is {hidden.size()}. Embedding size is {embedding_size}")
            last_input = torch.zeros_like(1, 1, 1, embedding_size)
        elif len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer,
                )
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            # If access idx is larger/equal.
            logging.error(
                'Taking last state because n_generated is too large'
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s, slice_answer: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer, sliced_answer
                )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]
            # try:
            #     last_input = hidden[n_generated - 1]
            # except:
            #     logging.error(f'stop_at = {stop_at}, answer len = {len(answer)}, token_stop_index = {token_stop_index}, n_input_token = {n_input_token}')
            #     logging.error(f'n_generated = {n_generated}, hidden size = {len(hidden)}')

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input.
        last_token_embedding = last_layer[:, -1, :].cpu()

        # Get log_likelihoods.
        # outputs.scores are the logits for the generated token.
        # outputs.scores is a tuple of len = n_generated_tokens.
        # Each entry is shape (bs, vocabulary size).
        # outputs.sequences is the sequence of all tokens: input and generated.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        # Transition_scores[0] only contains the scores for the first generated tokens.

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        if return_logits:
            all_logits = [s.cpu() for s in outputs.scores[:n_generated]]
            return sliced_answer, log_likelihoods, last_token_embedding, all_logits

        return sliced_answer, log_likelihoods, last_token_embedding

    def predict_batch(self, input_data_list, temperature, min_p=0.0):
        """
        Batch prediction for multiple prompts at once.
        Significantly faster than calling predict() multiple times.
        
        Args:
            input_data_list: List of prompts to generate from
            temperature: Sampling temperature (applies to all)
            min_p: Minimum probability threshold
            
        Returns:
            List of (answer, log_likelihoods, embedding) tuples
        """
        if not input_data_list:
            return []
        
        batch_size = len(input_data_list)
        logging.info(f'Batch prediction called with {batch_size} prompts, temperature={temperature}')
        
        # Ensure left padding for decoder-only batched generation
        orig_padding_side = getattr(self.tokenizer, 'padding_side', 'right')
        self.tokenizer.padding_side = "left"

        # Tokenize all inputs with padding
        if 'mistral' in self.model_name.lower():
            inputs = self.tokenizer(
                input_data_list, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                return_token_type_ids=False
            ).to("cuda")
        else:
            inputs = self.tokenizer(
                input_data_list, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to("cuda")

        self.tokenizer.padding_side = orig_padding_side
        
        # Store input lengths for each sequence (before padding)
        input_lengths = [
            (inputs['attention_mask'][i] == 1).sum().item() 
            for i in range(batch_size)
        ]
        
        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=inputs['input_ids'].shape[1],
                tokenizer=self.tokenizer,
                batch_size=batch_size)])
        else:
            stopping_criteria = None

        # Generate
        with torch.no_grad():
            if temperature == 0.0:
                logging.debug('Batch deterministic generation, do_sample=False')
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                logging.debug(f'Batch stochastic generation, do_sample=True, temperature={temperature}')
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        
        # Compute transition scores for log likelihoods
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        
        prompt_token_len = inputs['input_ids'].shape[1]

        # Process each output in the batch
        results = []
        for i in range(batch_size):
            input_data = input_data_list[i]
            
            # Slice newly generated tokens directly (from prompt_token_len onwards)
            generated_tokens = outputs.sequences[i, prompt_token_len:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Remove stop words
            sliced_answer = answer
            if self.stop_sequences is not None:
                earliest_idx = len(sliced_answer)
                for stop in self.stop_sequences:
                    if stop is not None and stop in sliced_answer:
                        idx = sliced_answer.find(stop)
                        if idx < earliest_idx:
                            earliest_idx = idx
                if earliest_idx < len(sliced_answer):
                    sliced_answer = sliced_answer[:earliest_idx]
            
            sliced_answer = sliced_answer.strip()
            
            # Get log likelihoods for this sequence
            # Find how many tokens were actually generated (non-padding)
            seq_len = outputs.sequences[i].shape[0]
            n_generated = seq_len - inputs['input_ids'].shape[1]
            
            if n_generated > 0:
                log_lls = [score.item() for score in transition_scores[i][:n_generated]]
            else:
                log_lls = []
            
            # No embeddings in batch mode to save memory
            embedding = None
            
            results.append((sliced_answer, log_lls, embedding))
        
        return results

    def predict_without_stop(self, input_data, temperature, min_p=0.0 , return_full=False):
        # Implement prediction.
        if 'mistral' in self.model_name.lower():
            inputs = self.tokenizer(input_data, return_tensors="pt", return_token_type_ids=False).to("cuda")
        else:
            inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        ## do sample False
        ## temperature 0.0
        with torch.no_grad():
            if temperature == 0.1 or temperature == 0.0:
                # Deterministic generation
                logging.debug('Deterministic generation called')
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    temperature=temperature,
                    min_p=min_p,
                    do_sample=False,
                    stopping_criteria=stopping_criteria,
                )
            else:
                logging.debug('Stochastic generation called')
                # Sampling with temperature
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    temperature=temperature,
                    min_p=min_p,
                    do_sample=False,
                    stopping_criteria=stopping_criteria,
                )
        
        # with torch.no_grad():
        #     outputs = self.model.generate(
        #         **inputs,
        #         max_new_tokens=self.max_new_tokens,
        #         return_dict_in_generate=True,
        #         output_scores=True,
        #         output_hidden_states=True,
        #         temperature=temperature,
        #         min_p=min_p,
        #         do_sample=True,
        #         stopping_criteria=stopping_criteria,
        #     )

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.startswith(input_data) or 'Llama-3' in self.model_name:
            # Llama-3 auto fix some typo in the input
            input_data_offset = len(input_data)
        else:
            raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]
        
        # Remove stop tokens
        answer = answer.split("### Instruction")[0]

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = answer.strip()
        stop_at = len(sliced_answer)

        # Get the number of tokens until the stop word comes up.
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated <= 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1

        # Get the last hidden state (last layer) and the last token's embedding of the answer.
        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 0:
            embedding_size = self.model.get_input_embeddings().weight.shape[1]
            logging.warning(f"Generate {sliced_answer}. Hidden size is {hidden.size()}. Embedding size is {embedding_size}")
            last_input = torch.zeros_like(1, 1, 1, embedding_size)
        elif len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer,
                )
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            # If access idx is larger/equal.
            logging.error(
                'Taking last state because n_generated is too large'
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s, slice_answer: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer, sliced_answer
                )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input.
        last_token_embedding = last_layer[:, -1, :].cpu()

        # Get log_likelihoods.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        # Transition_scores[0] only contains the scores for the first generated tokens.

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        return sliced_answer, log_likelihoods, last_token_embedding
    
    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input."""

        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']
        # The computation of the negative log likelihoods follows:
        # https://huggingface.co/docs/transformers/perplexity.

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()


class LlavaVLModel(BaseModel):
    """LLaVA Vision-Language Model."""

    def __init__(self, model_name='llava-hf/llava-v1.6-mistral-7b-hf', stop_sequences=None, max_new_tokens=None, token_limit=4096):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        if max_new_tokens is None:
            max_new_tokens = 128
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        
        logging.info(f'Loading LLaVA model: {model_name}')
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        try:
            import accelerate
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map='auto'
            )
        except (ImportError, ValueError) as e:
            logging.warning(f"Could not use device_map='auto', loading to cuda manually: {e}")
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_name = model_name
        self.stop_sequences = stop_sequences if stop_sequences else []
        self.token_limit = token_limit
        logging.info(f'LLaVA model loaded successfully')

    def predict(self, input_data, temperature, min_p=0.0, return_full=False, image_path=None):
        """Predict answer given prompt and image.
        
        Args:
            input_data: Text prompt
            temperature: Sampling temperature
            min_p: Minimum probability threshold
            return_full: Whether to return full output
            image_path: Path to image file
            
        Returns:
            tuple: (answer, log_likelihoods, embedding)
        """
        if image_path is None:
            raise ValueError("image_path is required for LlavaVLModel")
        
        # Hardcoded few-shot prompt from VQAv2 metadata.csv
        FEW_SHOT_PROMPT = """
Question: Where is he looking?
Answer: down

Question: What are the people in the background doing?
Answer: spectating

Question: What is he on top of?
Answer: table

Question: Is this a creamy soup?
Answer: no

Question: Is this rice noodle soup?
Answer: yes

Answer the following question as briefly as possible.\n
"""
        
        # Prepend few-shot prompt to input
        full_input = FEW_SHOT_PROMPT + input_data
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            raise
        
        # Construct conversation
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': full_input},
                    {'type': 'image'},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors='pt').to(self.model.device)
        
        # Generate
        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            if temperature == 0.0:
                # Deterministic generation
                logging.info(f'Deterministic generation called, do_sample=False, temperature={temperature}')
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                )
            else:
                # Sampling with temperature
                logging.info(f'Stochastic generation called, do_sample=True, temperature={temperature}')
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                )
        
        # Decode only the generated tokens
        input_len = inputs["input_ids"].shape[-1]
        gen_only = outputs.sequences[:, input_len:]
        answer = self.processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
        
        if return_full:
            return answer
        
        # Get log likelihoods
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        log_likelihoods = [score.item() for score in transition_scores[0]]
        
        # Get embeddings from hidden states
        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states
        
        # Extract last token embedding from last layer
        if len(hidden) > 0:
            last_layer = hidden[-1][-1]  # last generation step, last layer
            last_token_embedding = last_layer[:, -1, :].cpu()
        else:
            # Fallback: zero embedding
            embedding_size = self.model.language_model.get_input_embeddings().weight.shape[1]
            last_token_embedding = torch.zeros(1, embedding_size)
            logging.warning("No hidden states available, using zero embedding")
        
        return answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """Get the probability of the model answering A (True) for the given input.
        
        Note: This is a placeholder for VL models. Not typically used for VQA tasks.
        """
        raise NotImplementedError("get_p_true is not implemented for LlavaVLModel")


class QwenVLModel(BaseModel):
    """Qwen Vision-Language Model."""

    def __init__(self, model_name='Qwen/Qwen2.5-VL-7B-Instruct', stop_sequences=None, max_new_tokens=None, token_limit=4096):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        if max_new_tokens is None:
            max_new_tokens = 128
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        
        logging.info(f'Loading QwenVL model: {model_name}')
        try:
            import accelerate
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
        except (ImportError, ValueError) as e:
            logging.warning(f"Could not use device_map='auto', loading to cuda manually: {e}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.process_vision_info = process_vision_info
        
        self.model_name = model_name
        self.stop_sequences = stop_sequences if stop_sequences else []
        self.token_limit = token_limit
        logging.info(f'QwenVL model loaded successfully')

    def predict(self, input_data, temperature, min_p=0.0, return_full=False, image_path=None):
        """Predict answer given prompt and image.
        
        Args:
            input_data: Text prompt
            temperature: Sampling temperature
            min_p: Minimum probability threshold
            return_full: Whether to return full output
            image_path: Path to image file
            
        Returns:
            tuple: (answer, log_likelihoods, embedding)
        """
        if image_path is None:
            raise ValueError("image_path is required for QwenVLModel")
        
        # Hardcoded few-shot prompt from VQAv2 metadata.csv
        FEW_SHOT_PROMPT = """
        Question: Where is he looking?
Answer: down

Question: What are the people in the background doing?
Answer: spectating

Question: What is he on top of?
Answer: table

Question: Is this a creamy soup?
Answer: no

Question: Is this rice noodle soup?
Answer: yes

Answer the following question as briefly as possible.\n
"""
        
        # Prepend few-shot prompt to input
        full_input = FEW_SHOT_PROMPT + input_data
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            raise
        
        # Construct messages
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": full_input},
            ]}
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors='pt').to(device)
        
        # Generate
        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            if temperature == 0.0:
                # Deterministic generation
                logging.info(f'Deterministic generation called, do_sample=False, temperature={temperature}')
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                )
            else:
                # Sampling with temperature
                logging.info(f'Stochastic generation called, do_sample=True, temperature={temperature}')
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                )
        
        # Decode only the generated tokens
        input_len = inputs['input_ids'].shape[1]
        gen_only = outputs.sequences[:, input_len:]
        answer = self.processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
        
        if return_full:
            return answer
        
        # Get log likelihoods
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        log_likelihoods = [score.item() for score in transition_scores[0]]
        
        # Get embeddings from hidden states
        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states
        
        # Extract last token embedding from last layer
        if len(hidden) > 0:
            last_layer = hidden[-1][-1]  # last generation step, last layer
            last_token_embedding = last_layer[:, -1, :].cpu()
        else:
            # Fallback: zero embedding
            embedding_size = self.model.get_input_embeddings().weight.shape[1]
            last_token_embedding = torch.zeros(1, embedding_size)
            logging.warning("No hidden states available, using zero embedding")
        
        return answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """Get the probability of the model answering A (True) for the given input.
        
        Note: This is a placeholder for VL models. Not typically used for VQA tasks.
        """
        raise NotImplementedError("get_p_true is not implemented for QwenVLModel")


class AudioFlamingo3Model(BaseModel):
    """Wrapper for Audio Flamingo 3 audio->text model.

    This class loads `nvidia/audio-flamingo-3-hf` (or another HF repo) and
    exposes a simple `predict(prompt, audio_path, ...)` method that returns
    the decoded text. We keep this separate so the rest of the pipeline can
    call it the same way as other models.
    """

    def __init__(self, model_name='nvidia/audio-flamingo-3-hf', max_new_tokens=500):
        # Import inside init so loading the file doesn't require the audio deps
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        logging.info(f'Loading Audio Flamingo model: {model_name}')
        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_name, device_map='auto'
            )
        except Exception:
            # fallback to cpu/gpu manual load
            self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_name
            ).to('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        logging.info('AudioFlamingo3 model loaded successfully')

    def predict(self, input_text, audio_path, temperature=1.0, max_new_tokens=None):
        """Run AF3 on a single audio input.

        Args:
            input_text: The text prompt to prepend (few-shot + question)
            audio_path: URL or local path to the audio file
            temperature: sampling temperature (currently forwarded)
            max_new_tokens: override default max tokens

        Returns:
            tuple: (decoded_text, None, None)
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]

        # Build inputs using the processor
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode only generated tokens
        decoded = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        txt = decoded[0] if isinstance(decoded, (list, tuple)) else str(decoded)

        # Return in the same (answer, log_likelihoods, embedding) shape as other models
        return txt.strip(), None, None
