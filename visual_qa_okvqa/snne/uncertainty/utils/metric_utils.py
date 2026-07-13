import evaluate
from rouge_score import tokenizers
import os
import time


def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    list_answer = []
    
    # Filter reference
    if isinstance(answers['text'], str) or isinstance(answers['text'], int):
        list_answer = [answers['text']]
    elif isinstance(answers['text'], list):
        for text in answers['text']:
            if isinstance(text, str) or isinstance(text, int):
                list_answer.append(text)
    
    reference = {'answers': {'answer_start': answer_starts, 'text': list_answer}, 'id': example['id']}
    
    return reference


def run_eval(expression, output):
    try:
        # Safely evaluate the expression
        result = eval(expression)
        output.put(result)
    except Exception as e:
        output.put(e)
        
        
def model_based_metric(predicted_answer, example, model):
    if 'answers' in example:
        correct_answers = example['answers']['text']
    elif 'reference' in example:
        correct_answers = example['reference']['answers']['text']
    else:
        raise ValueError

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"

    if 'gpt' in model.model_name.lower():
        predicted_answer = model.predict(prompt, 0.01)
    else:
        predicted_answer, _, _ = model.predict(prompt, 0.01)

    if 'yes' in predicted_answer.lower():
        return 1.0
    elif 'no' in predicted_answer.lower():
        return 0.0
    else:
        print('Redo llm check.')
        predicted_answer, _, _ = model.predict(prompt, 1)
        if 'yes' in predicted_answer.lower():
            return 1.0
        elif 'no' in predicted_answer.lower():
            return 0.0

        print('Answer neither no nor yes. Defaulting to no!')
        return 0.0


def llm_metric(predicted_answer, example, model):
    return model_based_metric(predicted_answer, example, model)


def _get_sys_prompt_for_dataset(dataset_name: str):
    """Return a short system prompt / examples tailored to a dataset name.

    Keep prompts minimal: question, ground truth, prediction examples and yes/no answers.
    """
    dataset_name = (dataset_name or '').lower()
    if 'nq' in dataset_name or 'naturalquestions' in dataset_name:
        return (
            "You are evaluating open-domain question answering (NaturalQuestions style). "
            "Determine if the Predicted answer is semantically equivalent to the Ground truth. "
            "Check if the predicted answers matches any of all of the expected answers."
            "Respond ONLY with 'yes' or 'no'.\n"
            "\nExamples:\n"
            "Q: 'Who is the president of the United States in 2021?' | GT: 'Joe Biden' | Pred: 'Joe Biden' -> yes\n"
            "Q: 'Who is the president of the United States in 2021?' | GT: 'Joe Biden' | Pred: 'The current president is Joe Biden.' -> yes\n"
            "Q: 'Who is the president of the United States in 2021?' | GT: 'Joe Biden' | Pred: 'Donald Trump' -> no\n"
            "Q: 'What is the capital of France?' | GT: 'Paris' | Pred: 'Paris, France' -> yes\n"
            "Q: 'What is the capital of France?' | GT: 'Paris' | Pred: 'London' -> no\n"
        )
    elif 'bioasq' in dataset_name or 'bio' in dataset_name:
        return (
            "You are evaluating biomedical QA (BioASQ style). Determine if the Predicted answer "
            "correctly corresponds to the Ground truth. Check if the predicted answers matches any of all of the expected answers. Respond ONLY with 'yes' or 'no'.\n"
            "\nExamples:\n"
            "Q: 'What is the function of insulin?' | GT: 'Regulates blood glucose' | Pred: 'It regulates blood glucose levels.' -> yes\n"
            "Q: 'What is the function of insulin?' | GT: 'Regulates blood glucose' | Pred: 'A hormone involved in digestion.' -> no\n"
            "Q: 'Which virus causes COVID-19?' | GT: 'SARS-CoV-2' | Pred: 'SARS-CoV-2' -> yes\n"
            "Q: 'Which virus causes COVID-19?' | GT: 'SARS-CoV-2' | Pred: 'Influenza virus' -> no\n"
            "Q: 'What is the main symptom of diabetes?' | GT: 'High blood sugar' | Pred: 'Elevated blood sugar levels.' -> yes\n"
        )
    elif 'squad' in dataset_name:
        return (
            "You are evaluating extractive question answering (SQuAD style). Determine if the Predicted answer "
            "is semantically equivalent to the Ground truth. Check if the predicted answers matches any of all of the expected answers. Respond ONLY with 'yes' or 'no'.\n"
            "\nExamples:\n"
            "Q: 'How many dogs are in the image?' | GT: '3' | Pred: 'There are three dogs.' -> yes\n"
            "Q: 'How many dogs are in the image?' | GT: '3' | Pred: '2' -> no\n"
            "Q: 'What color is the car?' | GT: 'red' | Pred: 'The car is red.' -> yes\n"
            "Q: 'What color is the car?' | GT: 'red' | Pred: 'blue' -> no\n"
            "Q: 'Who wrote Hamlet?' | GT: 'William Shakespeare' | Pred: 'Shakespeare' -> yes\n"
            "Q: 'Who wrote Hamlet?' | GT: 'William Shakespeare' | Pred: 'Christopher Marlowe' -> no\n"
        )
    else:
        # Default (general) prompt
        return (
            "You are evaluating whether a Predicted answer correctly matches the Ground truth for a question. "
            "Respond ONLY with 'yes' or 'no'.\n"
            "\nExamples:\n"
            "Q: 'How many dogs?' | GT: '3' | Pred: 'There are three dogs.' -> yes\n"
            "Q: 'What color is the car?' | GT: 'red' | Pred: 'blue' -> no\n"
        )


def gemini_vqa_metric(predicted_answer, example, gemini_model=None, dataset='vqav2', max_retries=3):
    """Evaluate VQA answer correctness using Gemini API.
    
    Args:
        predicted_answer: The model's predicted answer
        example: Example dict with 'question' and 'answers' keys
        gemini_model: Optional pre-initialized Gemini model
        max_retries: Maximum number of retry attempts
        
    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    import google.generativeai as genai

    import logging
    
    logger = logging.getLogger(__name__)
    
    # Get ground truth answers
    gt_answers = example.get('answers', {}).get('text', [])
    question = example.get('question', '')
    dataset_arg = dataset or example.get('dataset')

    logger.info(f"GEMINI METRIC - Dataset: {dataset_arg}")
    logger.info(f"GEMINI METRIC - Question: {question}")
    logger.info(f"GEMINI METRIC - Ground truth: {gt_answers}")
    logger.info(f"GEMINI METRIC - Predicted: {predicted_answer}")

    if not gt_answers:
        logger.warning("GEMINI METRIC - No ground truth answers, returning 0.0")
        return 0.0

    # Initialize Gemini model if not provided
    if gemini_model is None:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable must be set")
        genai.configure(api_key=api_key)

        # Get system prompt based on dataset
        sys_prompt = _get_sys_prompt_for_dataset(dataset_arg)

        gemini_model = genai.GenerativeModel(
            'gemini-2.0-flash',
            system_instruction=sys_prompt
        )

    # Build ground truth string
    if len(gt_answers) == 1:
        gt_str = str(gt_answers[0])
    else:
        gt_str = ', '.join([str(a) for a in gt_answers[:5]])

    # Simple user prompt
    user_prompt = f"Question: {question}\nGround truth: {gt_str}\nPredicted: {predicted_answer}\n\nIs the predicted answer correct? Answer YES or NO."

    # logger.info(f"GEMINI METRIC - User prompt:\n{user_prompt}\n{'='*40}")

    # Try to get response with retries
    for attempt in range(max_retries):
        try:
            # logger.info(f"GEMINI METRIC - Attempt {attempt + 1}/{max_retries}: Calling Gemini API...")
            
            response = gemini_model.generate_content(
                user_prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 10,
                }
            )

            if response.candidates and response.candidates[0].content.parts:
                result = response.candidates[0].content.parts[0].text.strip().lower()
                logger.info(f"GEMINI METRIC - Response: '{result}'")
                
                if 'yes' in result:
                    # logger.info("GEMINI METRIC - Found 'yes' in response, returning 1.0")
                    return 1.0
                elif 'no' in result:
                    # logger.info("GEMINI METRIC - Found 'no' in response, returning 0.0")
                    return 0.0
                else:
                    # Unexpected response, retry
                    logger.warning(f"GEMINI METRIC - Unexpected response (no 'yes' or 'no'): '{result}'")
                    if attempt < max_retries - 1:
                        logger.info(f"GEMINI METRIC - Retrying after {2 ** attempt} seconds...")
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        # Default to incorrect if uncertain
                        logger.warning("GEMINI METRIC - Max retries reached with unexpected response, defaulting to 0.0")
                        return 0.0
            else:
                # Response blocked or empty
                logger.warning(f"GEMINI METRIC - Response blocked or empty")
                
                if attempt < max_retries - 1:
                    logger.info(f"GEMINI METRIC - Retrying after {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error("GEMINI METRIC - Max retries reached with empty/blocked response, returning 0.0")
                    return 0.0
                    
        except Exception as e:
            logger.error(f"GEMINI METRIC - Exception on attempt {attempt + 1}: {type(e).__name__}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"GEMINI METRIC - Retrying after {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
                continue
            else:
                logger.error("GEMINI METRIC - Max retries reached, returning 0.0")
                return 0.0
    
    return 0.0


def get_metric(metric):
    if metric == 'squad':
        squad_metric = evaluate.load("squad_v2")

        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError

            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}
            score = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])['f1']
            
            return 1.0 if (score >= 50.0) else 0.0
    
    elif metric == 'squad_raw':
        squad_metric = evaluate.load("squad_v2")

        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError

            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}
            score = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])['f1']
            
            return score / 100

    # Reuses the globally active model for these.
    elif metric == 'llm':
        metric = llm_metric
    
    # Gemini-based VQA evaluation
    elif metric == 'gemini_vqa':
        # Initialize Gemini model once for efficiency
        import google.generativeai as genai

        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable must be set for gemini_vqa metric")

        # Note: actual model will be created per-call with dataset-specific system prompt
        def metric(response, example, *args, **kwargs):
            # Dataset can be passed via kwargs or via environment variable 'EVAL_DATASET'
            dataset = kwargs.get('dataset') or os.environ.get('EVAL_DATASET')
            return gemini_vqa_metric(response, example, gemini_model=None, dataset=dataset)
    
    # Entailment
    elif metric == 'entail':
        def metric(response, example, model, strict_entailment, *args, **kwargs):
            is_true = False
            list_reference = example['reference']['answers']['text']
            question = example['question']
            
            for reference in list_reference:
                implication_1 = model.check_implication(f'{question} {response}', f'{question} {reference}', example=example)
                implication_2 = model.check_implication(f'{question} {reference}', f'{question} {response}', example=example)
                assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
                if strict_entailment:
                    is_true = (implication_1 == 2) and (implication_2 == 2)
                else:
                    implications = [implication_1, implication_2]
                    # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
                    is_true = (0 not in implications) and ([1, 1] != implications)
                if is_true:
                    break
            
            return 1.0 if is_true else 0.0
    # Rouge-L
    elif metric == 'rougel':
        rouge = evaluate.load('rouge', keep_in_memory=True)
        tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize
        
        def metric(response, example, *args, **kwargs):            
            score = rouge.compute(
                predictions=[response], 
                references=example['answers']['text'], 
                rouge_types=['rougeL'], 
                tokenizer=tokenizer)['rougeL']
            
            return score
    # BertScore
    elif metric == 'bertscore':
        bert_score = evaluate.load('bertscore')
        
        def metric(response, example, *args, **kwargs):            
            score = bert_score.compute(
                predictions=[response], 
                references=example['answers']['text'], 
                model_type='microsoft/deberta-v2-xlarge-mnli')['f1']
            
            return score[0]

    elif metric == 'vqa_acc':
        evaluator = VQAAccuracyEvaluator()
        
        def metric(response, example, *args, **kwargs):
            # Extract ground truths
            if 'answers' in example and 'text' in example['answers']:
                gt = example['answers']['text']
            elif 'reference' in example and 'answers' in example['reference']:
                gt = example['reference']['answers']['text']
            else:
                return 0.0
            
            return evaluator.compute_accuracy(response, gt)

    elif metric == 'vqarad_exact':
        evaluator = VQARadExactMatchEvaluator()
        
        def metric(response, example, *args, **kwargs):
            # Extract ground truth (VQA-RAD has single ground truth)
            if 'answers' in example and 'text' in example['answers']:
                gt = example['answers']['text']
            elif 'reference' in example and 'answers' in example['reference']:
                gt = example['reference']['answers']['text']
            else:
                return 0.0
            
            # Get first answer if it's a list
            if isinstance(gt, list) and len(gt) > 0:
                gt = gt[0]
            
            return evaluator.compute_exact_match(response, gt)

    else:
        raise ValueError

    return metric


import re

def check_vqa_match(p_norm, g_norm):
    if not p_norm or not g_norm: return False
    p_norm = str(p_norm).strip()
    g_norm = str(g_norm).strip()
    
    if p_norm == g_norm: return True
    
    # Plural check
    if p_norm + 's' == g_norm or g_norm + 's' == p_norm: return True
    if p_norm + 'es' == g_norm or g_norm + 'es' == p_norm: return True
    
    # Heuristic: extract from full sentences.
    prefixes = [
        "the answer is ", "my answer is ", "i think it is ",
        "it is ", "there is a ", "there is ", "there are ", 
        "it looks like a ", "it looks like ",
        "i can see a ", "i can see ",
        "i see a ", "i see "
    ]
    for p in prefixes:
        if p_norm.startswith(p):
            p_norm_stripped = p_norm[len(p):].strip()
            if p_norm_stripped == g_norm: return True
            if p_norm_stripped + 's' == g_norm or g_norm + 's' == p_norm_stripped: return True
            if p_norm_stripped + 'es' == g_norm or g_norm + 'es' == p_norm_stripped: return True
            
    # Substring extraction with word boundaries as fallback for short form reference
    if g_norm and len(g_norm) > 0:
        if re.search(r'\b' + re.escape(g_norm) + r'\b', p_norm):
            return True
        if re.search(r'\b' + re.escape(g_norm) + r's\b', p_norm):
            return True
            
    return False

class VQAAccuracyEvaluator:
    """VQA accuracy evaluator based on official VQAv2 evaluation code."""
    
    def __init__(self):
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
            "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
            "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
            "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
            "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
            "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
            "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
            "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
            "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
            "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
            "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
            "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
            "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
            "somebodys": "somebody's", "someoned": "someone'd",
            "someoned've": "someone'd've", "someone'dve": "someone'd've",
            "someonell": "someone'll", "someones": "someone's",
            "somethingd": "something'd", "somethingd've": "something'd've",
            "something'dve": "something'd've", "somethingll": "something'll",
            "thats": "that's", "thered": "there'd", "thered've": "there'd've",
            "there'dve": "there'd've", "therere": "there're", "theres": "there's",
            "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
            "theyll": "they'll", "theyre": "they're", "theyve": "they've",
            "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
            "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
            "whatll": "what'll", "whatre": "what're", "whats": "what's",
            "whatve": "what've", "whens": "when's", "whered": "where'd",
            "wheres": "where's", "whereve": "where've", "whod": "who'd",
            "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
            "whos": "who's", "whove": "who've", "whyll": "why'll",
            "whyre": "why're", "whys": "why's", "wont": "won't",
            "wouldve": "would've", "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
            "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
            "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
            "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
            "youve": "you've"
        }
        
        self.manualMap = {
            'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
            'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
            'nine': '9', 'ten': '10',
            'y': 'yes', 'true': 'yes',
            'n': 'no', 'false': 'no'
        }
        
        self.articles = ['a', 'an', 'the']
        
        # Regex patterns from official code
        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']
    
    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) is not None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText
    
    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.get(word, word)
            if word not in self.articles:
                outText.append(word)
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText
    
    def normalize_answer(self, answer):
        if not answer:
            return ''
        answer = str(answer).replace('\n', ' ')
        answer = answer.replace('\t', ' ')
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        return answer
    
    def compute_accuracy(self, predicted, ground_truths):
        if not ground_truths:
            return 0.0
        
        gtAnswers = []
        for ans in ground_truths:
            if isinstance(ans, dict):
                ans = ans.get('answer', str(ans))
            ans = str(ans).replace('\n', ' ').replace('\t', ' ').strip()
            gtAnswers.append(ans)
        
        resAns = str(predicted).replace('\n', ' ').replace('\t', ' ').strip()
        
        # Always normalize answers to catch true/false and numbers!
        gtAnswers = [self.normalize_answer(ans) for ans in gtAnswers]
        resAns = self.normalize_answer(resAns)
        
        gtAcc = []
        for i, gtAns in enumerate(gtAnswers):
            otherGTAns = gtAnswers[:i] + gtAnswers[i+1:]
            matchingAns = [ans for ans in otherGTAns if check_vqa_match(resAns, ans)]
            acc = min(1.0, float(len(matchingAns)) / 3.0)
            gtAcc.append(acc)
        
        avgGTAcc = float(sum(gtAcc)) / len(gtAcc) if gtAcc else 0.0
        return avgGTAcc


class VQARadExactMatchEvaluator(VQAAccuracyEvaluator):
    """VQA-RAD exact match evaluator inheriting normalization from VQAAccuracyEvaluator.
    
    This performs:
    - Lowercase conversion
    - Punctuation removal
    - Article removal ("the", "a", "an")
    - Sentence extraction heuristics for robust exact match
    """
    def compute_exact_match(self, predicted, ground_truth):
        normalized_pred = self.normalize_answer(predicted)
        normalized_gt = self.normalize_answer(ground_truth)
        return 1.0 if check_vqa_match(normalized_pred, normalized_gt) else 0.0