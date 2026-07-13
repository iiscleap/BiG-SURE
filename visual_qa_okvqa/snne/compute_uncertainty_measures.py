"""Compute uncertainty measures after generating answers."""
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Ignore all warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

from collections import defaultdict
import logging
import os
import gc
import copy
import pickle
import numpy as np
import torch
import wandb

from snne.analyze_results import analyze_run
from snne.uncertainty.utils.data_utils import load_ds
from snne.uncertainty.utils.metric_utils import get_metric
from snne.uncertainty.uncertainty_measures.semantic_entropy import (
    get_semantic_ids_using_entailment, 
    get_semantic_ids_using_embedding, 
    get_semantic_ids_using_exact_match, 
    get_semantic_ids_using_metric, 
    logsumexp_by_id, 
    predictive_entropy, 
    predictive_entropy_rao, 
    cluster_assignment_entropy, 
    context_entails_response, 
    soft_nearest_neighbor_loss, 
    weighted_cluster_assignment_entropy, 
    EntailmentDeberta, 
    EntailmentLlama, 
    Qwen2Embedding, 
    SFR2Embedding
)
from snne.uncertainty.uncertainty_measures import p_true as p_true_utils
from snne.uncertainty.utils import utils


utils.setup_logger()

EXP_DETAILS = 'experiment_details.pkl'


def main(args):

    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    user = os.environ['USER']
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    project = "snne" if not args.debug else "snne_debug"
    if args.assign_new_wandb_id:
        logging.info('Assign new wandb_id.')
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        args.run_name = utils.get_run_name("compute_uncertainty_measures", args, old_config=old_run.config)
        
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            name=args.run_name,
            notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
            # For convenience, keep any 'generate_answers' configs from old run,
            # but overwrite the rest!
            # NOTE: This means any special configs affecting this script must be
            # called again when calling this script!
            config={**old_run.config, **args.__dict__},
            tags=["eval_only", args.experiment_lot, 
                      f"entailment={args.entailment_model}",
                      f"metric={args.metric}-{args.metric_model}"],
            resume="allow"
        )

        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'

            return Restored
    else:
        logging.info('Reuse active wandb id.')

        def restore(filename):
            class Restored:
                name = f'{wandb.run.dir}/{filename}'
            return Restored

    # Load entailment model.
    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')
    
    if (args.snn_similarity_model == "embedding") or (args.semantic_similarity == "embedding"):
        if args.embedding_model == 'qwen':
            embedding_model = Qwen2Embedding()
        elif args.embedding_model == 'sfr':
            embedding_model = SFR2Embedding()
        else:
            raise ValueError
        logging.info('Embedding model loading complete.')
    else:
        embedding_model = None

    if args.compute_p_true_in_compute_stage:
        # This is usually not called.
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp['args'])

        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp['args'].dataset, 
            add_options=old_exp['args'].use_mc_options,
            seed=args.random_seed
        )
        del pt_validation_dataset

        # Reduce num generations used in p_true if needed!
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=pt_model,
            dataset=pt_train_dataset,
            indices=old_exp['p_true_indices'],
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=utils.get_make_prompt(old_exp['args']),
            num_generations=num_gen,
            metric=get_metric(old_exp['args'].metric))
        del p_true_responses
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info('Generated few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')
        
    # Define metric
    metric = None

    if args.recompute_accuracy:
        # This is usually not enabled.
        logging.warning('Recompute accuracy enabled. This does not apply to precomputed p_true!')
        metric = get_metric(args.metric)
        
        if args.metric_model is not None:
            metric_model = utils.init_model_from_name(args.metric_model)
        else:
            metric_model = None

    # Restore outputs from `generate_answrs.py` run.
    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict['semantic_ids'] = []

    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in enumerate(validation_generations):
        if (idx + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if args.recompute_accuracy:
            logging.info('Recomputing accuracy!')
            if utils.is_answerable(example):
                try:
                    acc = metric(most_likely_answer['response'], example, metric_model)
                except Exception as e:
                    logging.error("Unable to calculate metric due to an error from the model. Rollback to previous acc")
                    logging.error(str(e))
                    acc = most_likely_answer['accuracy']
            else:
                acc = 0.0  # pylint: disable=invalid-name
            validation_generations[tid]['most_likely_answer']["accuracy"] = acc
            validation_is_true.append(acc)
            logging.info('Recomputed accuracy!')

        else:
            validation_is_true.append(most_likely_answer['accuracy'])

        validation_answerable.append(utils.is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])
        logging.info('validation_is_true: %f', validation_is_true[-1])

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.compute_context_entails_response:
                # Compute context entails answer baseline.
                entropies['context_entails_response'].append(context_entails_response(
                    context, responses, entailment_model))
            
            responses_wo_context = copy.deepcopy(responses)

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids.
            # TODO: Add RougeL and BertScore
            similarity_matrix = None
            if args.semantic_similarity == "entailment":
                semantic_ids = get_semantic_ids_using_entailment(
                    responses, entailment_model,
                    strict_entailment=args.strict_entailment, 
                    cluster_method=args.cluster_method, 
                    example=example)
            elif args.semantic_similarity == "embedding":
                semantic_ids, similarity_matrix = get_semantic_ids_using_embedding(
                    responses, embedding_model, 
                    cluster_method=args.cluster_method, 
                    threshold=args.cosine_threshold)
            elif args.semantic_similarity == "exact_match":
                semantic_ids = get_semantic_ids_using_exact_match(
                    responses,
                    cluster_method=args.cluster_method)
            elif args.semantic_similarity == "metric":
                if metric is None:
                    metric = get_metric(args.metric)
                semantic_ids = get_semantic_ids_using_metric(
                    responses_wo_context,
                    metric,
                    copy.deepcopy(example['reference']),
                    cluster_method=args.cluster_method,
                )

            result_dict['semantic_ids'].append(semantic_ids)

            # Compute entropy from frequencies of cluster assignments.
            if args.compute_cluster_assignment_entropy:
                entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))
            
            # Compute soft-nearest neighbor loss
            if args.compute_snn:
                snn_responses = responses_wo_context if args.snn_wo_context else responses
                pe = soft_nearest_neighbor_loss(
                        snn_responses, entailment_model, embedding_model, semantic_ids,
                        similarity_matrix=similarity_matrix,
                        variant=args.snn_variant, 
                        similarity_model=args.snn_similarity_model,
                        temperature=args.snn_temperature, 
                        exclude_diagonal=not args.self_similarity, 
                        strict_entailment=not args.include_neutral)
                entropies['soft_nearest_neighbor'].append(pe)

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Compute naive entropy.
            if args.compute_regular_entropy:
                entropies['regular_entropy'].append(predictive_entropy(log_liks_agg))
                
            # Compute normalized cluster probability
            if args.compute_semantic_entropy or args.compute_weighted_cluster_assignment_entropy:
                log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')

            # Compute semantic entropy.
            if args.compute_semantic_entropy:
                pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
                entropies['semantic_entropy'].append(pe)
            
            # Compute weighted cluster assignment entropy
            if args.compute_weighted_cluster_assignment_entropy:
                pe = weighted_cluster_assignment_entropy(semantic_ids, log_likelihood_per_semantic_id)
                entropies['weighted_cluster_assignment_entropy'].append(pe)
                
            # Compute weighted soft-nearest neighbor loss
            if args.compute_wsnn:
                weight_pe = torch.exp(torch.tensor(log_liks_agg))
                weight_pe = weight_pe / weight_pe.mean()
                snn_responses = responses_wo_context if args.snn_wo_context else responses
                pe = soft_nearest_neighbor_loss(
                        snn_responses, entailment_model, embedding_model, semantic_ids,
                        similarity_matrix=similarity_matrix,
                        variant=args.snn_variant,
                        similarity_model=args.snn_similarity_model,
                        temperature=args.snn_temperature, 
                        exclude_diagonal=not args.self_similarity,
                        strict_entailment=not args.include_neutral, weight=weight_pe)
                entropies['weighted_soft_nearest_neighbor'].append(pe)

            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            logging.info(80*'#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Context:')
            logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(example['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            logging.info('p_true: %s', np.exp(p_true))

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    if args.recompute_accuracy:
        logging.info('Saving new generations')
        utils.save(validation_generations, f'validation_generations.pkl')
    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    if args.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    utils.save(result_dict, 'uncertainty_measures.pkl')

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        # Follow up with computation of aggregate performance metrics.
        logging.info(50 * '#X')
        logging.info('STARTING `analyze_run`!')
        analyze_run(wandb.run.id)
        logging.info(50 * '#X')
        logging.info('FINISHED `analyze_run`!')


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
