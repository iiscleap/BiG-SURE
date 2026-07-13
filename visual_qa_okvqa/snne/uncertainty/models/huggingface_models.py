"""Implement HuggingfaceModel models."""
import os
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
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


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
        
        if model_name.endswith('-8bit'):
            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}
            model_name = model_name[:-len('-8bit')]
            eightbit = True
        if model_name.endswith('-4bit'):
            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_4bit=True,)}
            model_name = model_name[:-len('-4bit')]
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
                model_id, device_map="auto", torch_dtype="auto", 
                max_memory={0: '80GIB'}, **kwargs,)
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

            llama65b = '65b' in model_name and base == 'huggyllama'
            llama2_70b = '70b' in model_name and base == 'meta-llama'

            if ('7b' in model_name or '13b' in model_name or '8B' in model_name or '1B' in model_name or '3B' in model_name) or eightbit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}", device_map="auto", torch_dtype="auto",  
                    max_memory={0: '80GIB'}, **kwargs,)

            elif llama2_70b or llama65b:
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
                raise ValueError

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
            try:
                import accelerate
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, device_map='auto', torch_dtype="auto"
                )
            except (ImportError, ValueError) as e:
                logging.warning(f"Could not use device_map='auto', loading to cuda manually: {e}")
                logging.warning(f"Loading model to cuda manually", torch.cuda.is_available())
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float16
                ).to('cuda' if torch.cuda.is_available() else 'cpu')

        elif 'qwen' in model_name.lower():
            model_id = f'Qwen/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map="auto",
                token_type_ids=None)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype="auto", 
                max_memory={0: '80GIB'}, **kwargs,)
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = token_limit
        # Setting padding for open-ended generation
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        print(f"Setting pad_token_id to {self.tokenizer.eos_token_id}")

    def predict(self, input_data, temperature, min_p=0.0, return_full=False):

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
            input_data_offset = len(input_data)
        else:
            raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                # if 'falcon' not in self.model_name.lower():
                #     raise ValueError(error_msg)
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

        return sliced_answer, log_likelihoods, last_token_embedding

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
                    top_p=0.95,
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
                    top_p=0.95,
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


class Qwen3VLModel(BaseModel):
    """Qwen3 Vision-Language Model."""

    def __init__(self, model_name='Qwen/Qwen3-VL-8B-Instruct', stop_sequences=None, max_new_tokens=None, token_limit=4096):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        if max_new_tokens is None:
            max_new_tokens = 128
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        
        logging.info(f'Loading Qwen3VL model: {model_name}')
        try:
            import accelerate
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
        except (ImportError, ValueError) as e:
            logging.warning(f"Could not use device_map='auto', loading to cuda manually: {e}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.model_name = model_name
        self.stop_sequences = stop_sequences if stop_sequences else []
        self.token_limit = token_limit
        logging.info(f'Qwen3VL model loaded successfully')

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
            raise ValueError("image_path is required for Qwen3VLModel")
        
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
        
        # Construct messages
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": full_input},
            ]}
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
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
                    top_p=0.95,
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
        raise NotImplementedError("get_p_true is not implemented for Qwen3VLModel")


class Gemma3VLModel(BaseModel):
    """Gemma 3 Vision-Language Model (google/gemma-3-12b-it)."""

    def __init__(self, model_name='google/gemma-3-12b-it', stop_sequences=None, max_new_tokens=None, token_limit=4096):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        
        if max_new_tokens is None:
            max_new_tokens = 128
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        
        logging.info(f'Loading Gemma3 VL model: {model_name}')
        try:
            import accelerate
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name, device_map='auto', torch_dtype=torch.bfloat16
            ).eval()
        except (ImportError, ValueError) as e:
            logging.warning(f"Could not use device_map='auto', loading to cuda manually: {e}")
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            ).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.model_name = model_name
        self.stop_sequences = stop_sequences if stop_sequences else []
        self.token_limit = token_limit
        logging.info(f'Gemma3 VL model loaded successfully')

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
            raise ValueError("image_path is required for Gemma3VLModel")
        
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

Answer the following question as briefly as possible.
"""
        
        # Prepend few-shot prompt to input
        full_input = FEW_SHOT_PROMPT + input_data
        
        # Construct messages for Gemma 3
        # Gemma 3 supports image as a local file path
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant that answers questions about images briefly and accurately."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": full_input},
                ]
            }
        ]
        
        # Process inputs using chat template
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate
        logging.debug('temperature: %f', temperature)
        with torch.inference_mode():
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
                    top_p=0.95,
                )
        
        # Decode only the generated tokens
        gen_only = outputs.sequences[:, input_len:]
        answer = self.processor.decode(gen_only[0], skip_special_tokens=True).strip()
        
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
        raise NotImplementedError("get_p_true is not implemented for Gemma3VLModel")


class Phi4VLModel(BaseModel):
    """Phi-4 Vision-Language Model."""

    def __init__(self, model_name='microsoft/Phi-4-multimodal-instruct', stop_sequences=None, max_new_tokens=None, token_limit=4096):
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        if max_new_tokens is None:
            max_new_tokens = 128
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        
        logging.info(f'Loading Phi4 VL model: {model_name}')
        try:
            import accelerate
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map='auto', torch_dtype='auto', trust_remote_code=True,
                _attn_implementation='flash_attention_2'
            ).eval()
        except (ImportError, ValueError) as e:
            logging.warning(f"Could not use flash_attn or auto device, loading to cuda manually: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype='auto', trust_remote_code=True
            ).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        self.model_name = model_name
        self.stop_sequences = stop_sequences if stop_sequences else []
        self.token_limit = token_limit
        logging.info(f'Phi4 VL model loaded successfully')

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
            raise ValueError("image_path is required for Phi4VLModel")
        
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

Answer the following question as briefly as possible.
"""
        
        # Prepend few-shot prompt to input
        full_input = FEW_SHOT_PROMPT + input_data
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            raise

        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        
        # The prompt format: <|user|><|image_1|>Text prompt<|end|><|assistant|>
        prompt = f'{user_prompt}<|image_1|>{full_input}{prompt_suffix}{assistant_prompt}'
        
        inputs = self.processor(text=prompt, images=image, return_tensors='pt').to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate
        logging.debug('temperature: %f', temperature)
        with torch.inference_mode():
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
                    top_p=0.95,
                )
        
        # Decode only the generated tokens
        gen_only = outputs.sequences[:, input_len:]
        answer = self.processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
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
        raise NotImplementedError("get_p_true is not implemented for Phi4VLModel")


class PixtralVLModel(BaseModel):
    """Pixtral Vision-Language Model."""

    def __init__(self, model_name='mistralai/Pixtral-12B-2409', stop_sequences=None, max_new_tokens=None, token_limit=8192):
        from vllm import LLM
        
        if max_new_tokens is None:
            max_new_tokens = 128
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        
        logging.info(f'Loading Pixtral VL model via vLLM: {model_name}')
        
        # Initialize the vLLM engine
        self.llm = LLM(
            model=model_name, 
            tokenizer_mode="mistral",
            max_model_len=token_limit,  # Configure based on token limits or defaults
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1
        )
        
        self.model_name = model_name
        self.stop_sequences = stop_sequences if stop_sequences else []
        self.token_limit = token_limit
        logging.info(f'Pixtral VL model loaded successfully')

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
        from vllm.sampling_params import SamplingParams
        
        if image_path is None:
            raise ValueError("image_path is required for PixtralVLModel")
        
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

Answer the following question as briefly as possible.
"""
        
        # Prepend few-shot prompt to input
        full_input = FEW_SHOT_PROMPT + input_data
        
        # Convert local image to base64 data URI (vLLM only accepts data:image or http URLs)
        import base64
        import mimetypes
        
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        
        with open(image_path, "rb") as img_file:
            image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        file_url = f"data:{mime_type};base64,{image_b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_input}, 
                    {"type": "image_url", "image_url": {"url": file_url}}
                ]
            },
        ]
        
        # Determine sampling params based on temperature
        if temperature == 0.0:
            logging.info(f'Deterministic generation called, temperature={temperature}')
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self.max_new_tokens,
                skip_special_tokens=True,
                logprobs=1
            )
        else:
            logging.info(f'Stochastic generation called, temperature={temperature}')
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.95,
                max_tokens=self.max_new_tokens,
                skip_special_tokens=True,
                logprobs=1
            )
            
        # Optional: Add stop sequences if processing supports it
        if self.stop_sequences:
            sampling_params.stop = self.stop_sequences

        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        
        # Process output
        response_output = outputs[0].outputs[0]
        answer = response_output.text.strip()
        
        if return_full:
            return answer
            
        # Extract log likelihoods directly from vLLM output token_ids and logprobs
        log_likelihoods = []
        if response_output.logprobs:
            for token_logprob_dict, token_id in zip(response_output.logprobs, response_output.token_ids):
                if token_id in token_logprob_dict:
                    log_likelihoods.append(token_logprob_dict[token_id].logprob)
                else:
                    # Fallback if token wasn't in top logprobs
                    log_likelihoods.append(-100.0)
                    
        # Get embeddings from hidden states (vLLM doesn't generally expose hidden states via chat API)
        # We'll return a zero embedding as fallback like other models do when hidden states are unavailable
        embedding_size = 4096  # Using Pixtral default embedding size approx or zero fallback
        last_token_embedding = torch.zeros(1, embedding_size)
        
        return answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """Get the probability of the model answering A (True) for the given input."""
        raise NotImplementedError("get_p_true is not implemented for PixtralVLModel")


class GeminiModel(BaseModel):
    """Wrapper for Gemini models using Google GenAI SDK."""

    def __init__(self, model_name='gemini-2.5-flash', stop_sequences=None, max_new_tokens=None, token_limit=4096):
        if max_new_tokens is None:
            max_new_tokens = 1024

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = stop_sequences if stop_sequences else []
        self.token_limit = token_limit

        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError('GOOGLE_API_KEY environment variable not set for GeminiModel.')

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError('GeminiModel requires `google-genai`. Please install it in this environment.') from exc

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.safety_settings = [
            types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_CIVIC_INTEGRITY', threshold='BLOCK_NONE'),
        ]
        logging.info(f'Gemini model initialized successfully: {self.model_name}')

    def _build_config(self, temperature):
        # Gemini often needs a larger budget than HF `max_new_tokens` to avoid
        # finishing with MAX_TOKENS before producing a visible final answer.
        max_output_tokens = max(int(self.max_new_tokens), 256)
        return self._types.GenerateContentConfig(
            temperature=float(temperature),
            top_p=1.0,
            max_output_tokens=max_output_tokens,
            safety_settings=self.safety_settings,
            thinking_config=self._types.ThinkingConfig(include_thoughts=False),
        )

    def _extract_text_from_response(self, response):
        text = (getattr(response, 'text', None) or '').strip()
        if text:
            return text

        candidates = getattr(response, 'candidates', None) or []
        for candidate in candidates:
            content = getattr(candidate, 'content', None)
            if content is None:
                continue
            parts = getattr(content, 'parts', None) or []
            part_texts = []
            for part in parts:
                part_text = getattr(part, 'text', None)
                if part_text:
                    part_texts.append(str(part_text))
            if part_texts:
                return ''.join(part_texts).strip()

        return ''

    def _trim_with_stop_sequences(self, text):
        if not text:
            return text
        trimmed = text.strip()
        for stop in self.stop_sequences:
            if not stop:
                continue
            idx = trimmed.find(stop)
            if idx != -1:
                trimmed = trimmed[:idx]
        return trimmed.strip()

    def predict(self, input_data, temperature, min_p=0.0, return_full=False, image_path=None):
        del min_p  # Unused for Gemini SDK generation.

        # Use the exact incoming prompt to stay consistent with other models.
        contents = [input_data]
        if image_path is not None:
            # Text-only path is used by SNNE generate scripts; keeping optional image support.
            img = Image.open(image_path).convert('RGB')
            contents.append(img)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=self._build_config(temperature),
            )
        except Exception as exc:
            logging.exception('Gemini API call failed: %s', exc)
            text = 'error'
        else:
            text = self._extract_text_from_response(response)
            if not text:
                finish_reason = None
                if getattr(response, 'candidates', None):
                    finish_reason = response.candidates[0].finish_reason
                logging.warning('Gemini returned empty response. finish_reason=%s', finish_reason)
                text = 'blocked_safety' if 'SAFETY' in str(finish_reason) else 'blocked_unknown'

        # Remove common assistant-style prefixes.
        lowered = text.lower()
        if lowered.startswith('answer:'):
            text = text[len('answer:'):].strip()

        text = self._trim_with_stop_sequences(text)
        if not text:
            text = 'no_answer'

        if return_full:
            return text

        return text, None, None

    def get_p_true(self, input_data):
        del input_data
        raise NotImplementedError('get_p_true is not implemented for GeminiModel')

