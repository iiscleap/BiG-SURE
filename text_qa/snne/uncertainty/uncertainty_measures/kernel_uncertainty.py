import os
import pickle
import logging
from collections import defaultdict

import wandb
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from snne.uncertainty.utils import utils


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:

    def init_prediction_cache(self, entailment_cache_id):
        if entailment_cache_id is None:
            return dict()

        logging.info('Restoring prediction cache from %s', entailment_cache_id)

        api = wandb.Api()
        run = api.run(entailment_cache_id)
        run.file(self.entailment_file).download(
            replace=True, exist_ok=False, root=wandb.run.dir)

        with open(f'{wandb.run.dir}/{self.entailment_file}', "rb") as infile:
            return pickle.load(infile)

    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    entailment_file = 'deberta_entailment_cache.pkl'

    def __init__(self, entailment_cache_id, entailment_cache_only):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(DEVICE)
        self.prediction_cache = self.init_prediction_cache(entailment_cache_id)
        self.entailment_cache_only = entailment_cache_only

    def check_implication(self, text1, text2, *args, batch_size=128, **kwargs):
        # Support both single pairs and batched pairs
        is_batch = isinstance(text1, list) and isinstance(text2, list)
        
        if not is_batch:
            # Single pair - original behavior with caching
            hashed = utils.md5hash(f"Text1 for DeBerta: {text1}, Text2 for DeBerta: {text2}")
            if hashed in self.prediction_cache:
                logging.info('Restoring hashed instead of predicting with model.')
                prediction, confidence = self.prediction_cache[hashed]
            else:
                if self.entailment_cache_only:
                    raise ValueError
                inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
                # The model checks if text1 -> text2, i.e. if text2 follows from text1.
                # check_implication('The weather is good', 'The weather is good and I like you') --> 1
                # check_implication('The weather is good and I like you', 'The weather is good') --> 2
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits
                # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
                activations = F.softmax(logits, dim=1)
                largest_index = torch.argmax(activations)  # pylint: disable=no-member
                confidence = torch.max(activations)
                prediction = largest_index.cpu().item()
                if os.environ.get('DEBERTA_FULL_LOG', False):
                    logging.info('Deberta Input: %s -> %s', text1, text2)
                    logging.info('Deberta Prediction: %s', prediction)
                    logging.info('Deberta Prediction Prob: %s', confidence)
                self.prediction_cache[hashed] = (prediction, confidence.cpu().item())
            return prediction, confidence
        else:
            # Batched pairs
            assert len(text1) == len(text2), "Batch text1 and text2 must have same length"
            all_predictions = []
            all_confidences = []
            num_pairs = len(text1)
            
            # Check cache first and track what needs computation
            uncached_indices = []
            uncached_text1 = []
            uncached_text2 = []
            
            for idx in range(num_pairs):
                hashed = utils.md5hash(f"Text1 for DeBerta: {text1[idx]}, Text2 for DeBerta: {text2[idx]}")
                if hashed in self.prediction_cache:
                    pred, conf = self.prediction_cache[hashed]
                    all_predictions.append(pred)
                    all_confidences.append(conf)
                else:
                    uncached_indices.append(idx)
                    uncached_text1.append(text1[idx])
                    uncached_text2.append(text2[idx])
                    all_predictions.append(None)  # Placeholder
                    all_confidences.append(None)  # Placeholder
            
            # Batch process uncached pairs
            if uncached_indices:
                if self.entailment_cache_only:
                    raise ValueError("entailment_cache_only is True but uncached pairs found")
                
                for start_idx in range(0, len(uncached_text1), batch_size):
                    end_idx = min(start_idx + batch_size, len(uncached_text1))
                    batch_text1 = uncached_text1[start_idx:end_idx]
                    batch_text2 = uncached_text2[start_idx:end_idx]
                    
                    inputs = self.tokenizer(batch_text1, batch_text2, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    logits = outputs.logits
                    activations = F.softmax(logits, dim=1)
                    largest_indices = torch.argmax(activations, dim=1)
                    confidences = torch.max(activations, dim=1)[0]
                    
                    # Fill in results and cache
                    for batch_offset in range(len(batch_text1)):
                        uncached_idx = start_idx + batch_offset
                        global_idx = uncached_indices[uncached_idx]
                        pred = largest_indices[batch_offset].cpu().item()
                        conf = confidences[batch_offset].cpu().item()
                        
                        all_predictions[global_idx] = pred
                        all_confidences[global_idx] = conf
                        
                        # Cache the result
                        hashed = utils.md5hash(f"Text1 for DeBerta: {text1[global_idx]}, Text2 for DeBerta: {text2[global_idx]}")
                        self.prediction_cache[hashed] = (pred, conf)
            
            return all_predictions, all_confidences
    
    def save_prediction_cache(self):
        # Write the dictionary to a pickle file.
        utils.save(self.prediction_cache, self.entailment_file)


def get_entailment_graph(strings_list, model, is_weighted=False, example=None, weight_strategy="manual"):
    """
    Get graph of entailment (optimized with batching)
    """
    n = len(strings_list)
    nodes = range(n)
    edges = []
    
    # Collect all pairs to check
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    
    if len(pairs) == 0:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        return G
    
    # Prepare batch inputs
    text1_list = [strings_list[i] for i, j in pairs]
    text2_list = [strings_list[j] for i, j in pairs]
    
    # Batch compute forward implications (i -> j)
    impl_forward, prob_forward = model.check_implication(text1_list, text2_list, example=example)
    
    # Batch compute backward implications (j -> i)
    impl_backward, prob_backward = model.check_implication(text2_list, text1_list, example=example)
    
    # Build edges from results
    for idx, (i, j) in enumerate(pairs):
        implication_1 = impl_forward[idx]
        implication_2 = impl_backward[idx]
        prob_impl1 = prob_forward[idx]
        prob_impl2 = prob_backward[idx]
        
        assert (implication_1 in [0, 1, 2])
        weight = int(implication_1 == 2) + int(implication_2 == 2) + 0.5 * int(implication_1 == 1) + 0.5 * int(implication_2 == 1)
        
        if is_weighted:
            if weight_strategy == "manual":
                edge_weight = weight
            elif weight_strategy == "deberta":
                edge_weight = prob_impl1 + prob_impl2
            else:
                raise ValueError(f"Unknown weight strategy {weight_strategy}")
            
            if edge_weight:
                edges.append((i, j, edge_weight))
        else:
            if weight:
                edges.append((i, j))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    if is_weighted:
        G.add_weighted_edges_from(edges)
    else:
        G.add_edges_from(edges)
    return G


def get_semantic_ids_graph(strings_list, model, semantic_ids, ordered_ids, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning (optimized with batching)."""
    nodes = ordered_ids
    weights = defaultdict(list)  # (i, j) -> weight
    
    n = len(strings_list)
    
    # Collect all pairs to check
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    
    if len(pairs) == 0:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        return G
    
    # Prepare batch inputs
    text1_list = [strings_list[i] for i, j in pairs]
    text2_list = [strings_list[j] for i, j in pairs]
    
    # Batch compute forward implications (i -> j)
    impl_forward, prob_forward = model.check_implication(text1_list, text2_list, example=example)
    
    # Batch compute backward implications (j -> i)
    impl_backward, prob_backward = model.check_implication(text2_list, text1_list, example=example)
    
    # Build edges from results
    for idx, (i, j) in enumerate(pairs):
        implication_1 = impl_forward[idx]
        implication_2 = impl_backward[idx]
        
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
        
        edge_weight = (implication_1 == 2) + (implication_1 == 1) * 0.5 + \
                      (implication_2 == 2) + (implication_2 == 1) * 0.5
        
        if edge_weight > 0:
            node_i = semantic_ids[i]
            node_j = semantic_ids[j]
            weights[(node_i, node_j)].append(edge_weight)
    
    for k, v in weights.items():
        weights[k] = np.sum(v)
    
    assert -1 not in semantic_ids
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from([(i, j, w) for (i, j), w in weights.items()])
    return G