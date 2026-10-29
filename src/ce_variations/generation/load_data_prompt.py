import json
import os
import random
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset
import ast
seed = 42
random.seed(seed)

# Directory with {nq,sciq,triviaqa}/train.csv and the test-question jsonl files.
data_path = os.environ.get('CE_DATA_DIR', 'data')
confidence_expressions = ['Almost No Chance', 'Highly Unlikely', 'Chances are Slight', 'Little Chance', 
                          'Unlikely', 'Probably Not', 'About Even', 'Better than Even', 'Likely',
                          'Probably', 'Very Good Chance', 'Highly Likely', 'Almost Certain']
confidence_expressions_probs = [0.02,0.05,0.1,0.1,0.2,0.25,0.5,0.6,0.7,0.7,0.8,0.9,0.95]
confidence_expressions_choices = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
model_list = ['gpt4o', "mistral", "llama70b", 'qwen72b', 'qwen32b', 'qwen14b', 'olmo32b', 'olmo13b', 'olmo13b-dpo', 'olmo13b-sft', 'olmo', 'olmo-dpo', 'olmo-sft']
formal_model_list = ["GPT-4o", "Mistral-123B", "Llama-70B", 'Qwen-72B', 'Qwen-32B', 'Qwen-14B',  'OLMo-32B INS', 'OLMo-13B INS', 'OLMo-13B DPO', 'OLMo-13B SFT', 'OLMo-7B INS', 'OLMo-7B DPO', 'OLMo-7B SFT']
dataname_list = ['nq', 'sciq', 'triviaqa', 'popqa']
formal_dataname_list = ['NQ', 'SciQ', 'TriviaQA', 'TruthfulQA', 'PopQA']
formal_prompt_names =['P(1)', 'P(%)', 'P(10)', 'CF(1)', 'CT(1)', 'L.', 'L. MC', 'RP(1)', 'RP(%)', 'RP(10)']
# dataname_list = ['popqa']

abstain_answers = ["I don't know the answer.", "I cannot be sure about my answer.", "That’s outside my current knowledge base."]
def ans0_prompt():

    prompt = f"""Answer the question, give ONLY the answer, no other words or explanation:\n\n"""
    
    return prompt

def ans1_prompt():

    prompt = f"""Answer the question, give ONLY the answer without explanation:\n\n"""
    
    return prompt

def ans2_prompt():

    prompt = f"""Answer the question, give ONLY the answer:\n\n"""
    
    return prompt

def ans3_prompt():

    prompt = f"""Answer the question as short as possible:\n\n"""
    
    return prompt

def ans4_prompt():

    prompt = f"""Answer the question with minimal words:\n\n"""
    
    return prompt

def ans5_prompt():

    prompt = f"""Provide an answer for the question, give ONLY the answer, no other words or explanation:\n\n"""
    
    return prompt

def ans6_prompt():

    prompt = f"""Provide an answer for the question without explanation:\n\n"""
    
    return prompt

def ans7_prompt():

    prompt = f"""Provide an answer for the question, give ONLY the answer:\n\n"""
    
    return prompt

def ans8_prompt():

    prompt = f"""Provide as short an answer as possible to the question:\n\n"""
    
    return prompt

def ans9_prompt():

    prompt = f"""Provide an answer with minimal words for the question:\n\n"""
    
    return prompt

def origin_prob_prompt():

    prompt = f"""Provide the probability that your answer is correct. """
    prompt += f"""Give ONLY the probability between 0.0 and 1.0, no other words or explanation."""

    return prompt


def confi_prompt():

    prompt = f"""Provide the confidence that your answer is correct. """ 
    prompt += f"""Give ONLY the confidence between 0.0 and 1.0, no other words or explanation."""
    
    return prompt

def certain_prompt():

    prompt = f"""Provide the certainty that your answer is correct. """ 
    prompt += f"""Give ONLY the certainty between 0.0 and 1.0, no other words or explanation."""
    
    return prompt

def origin_adv_prob_prompt():

    prompt = f"""Provide the probability that your answer is correct. """ 
    prompt += f"""Give ONLY the probability between 0.0 and 1.0 (where 0.0 means that the answer is completely correct, 1.0 means completely wrong), no other words or explanation."""
    
    return prompt

def _0_to_10_prob_prompt():
    
    prompt = f"""Provide the probability that your answer is correct. """ 
    prompt += f"""Give ONLY the probability between 0 and 10, no other words or explanation."""
    
    return prompt

def _0_to_10_confi_prompt():
    
    prompt = f"""Provide the confidence that your answer is correct. """ 
    prompt += f"""Give ONLY the confidence between 0 and 10, no other words or explanation."""
    
    return prompt

def _0_to_10_adv_prob_prompt():

    prompt = f"""Provide the probability that your answer is correct. """ 
    prompt += f"""Give ONLY the probability between 0 and 10 (where 0 means that the answer is completely correct, 10 means completely wrong), no other words or explanation."""
    
    return prompt

def percent_prob_prompt():

    prompt = f"""Provide the probability that your answer is correct. """
    prompt += f"""Give ONLY the probability between 0% and 100%, no other words or explanation."""
    
    return prompt

def percent_adv_prob_prompt():

    prompt = f"""Provide the probability that your answer is correct. """
    prompt += f"""Give ONLY the probability between 0% and 100% (where 0% means that the answer is completely correct, 100% means completely wrong), no other words or explanation."""
    
    return prompt

def ling_likely_prompt():
    
    prompt = f"""Describe how likely it is that your answer is correct as one of the following expressions: """
    prompt += f"""['Almost No Chance', 'Highly Unlikely', 'Chances are Slight', 'Little Chance', 'Unlikely', 'Probably Not', 'About Even', 'Better than Even', 'Likely', 'Probably', 'Very Good Chance','Highly Likely', 'Almost Certain']. """
    prompt += f"""Give ONLY the chosen expression, no other words or explanation."""

    return prompt

def multichoice_likely_prompt():
    
    prompt = f"""Describe how likely it is that your answer is correct by choosing one of the following options: """
    prompt += f"""[a: 'Almost No Chance', b: 'Highly Unlikely', c: 'Chances are Slight', d: 'Little Chance', e: 'Unlikely', f: 'Probably Not', g: 'About Even', h: 'Better than Even', i: 'Likely', j: 'Probably', k; 'Very Good Chance', l: 'Highly Likely', m: 'Almost Certain']. """
    prompt += f"""Give ONLY the chosen option, no other words or explanation."""

    return prompt


prompt_type_dict = {
    "prob": origin_prob_prompt,
    "per": percent_prob_prompt,
    "0-10": _0_to_10_prob_prompt,
    "confi": confi_prompt,
    "certain": certain_prompt,
    "ling": ling_likely_prompt,
    "mc": multichoice_likely_prompt,
    # "adv_prob": origin_adv_prob_prompt,
    # "per_adv": percent_adv_prob_prompt,
    # "0-10_adv": _0_to_10_adv_prob_prompt,   
}

answer_type_dict = {
    "ans0": ans0_prompt,
    "ans1": ans1_prompt,
    "ans2": ans2_prompt,
    "ans3": ans3_prompt,
    "ans4": ans4_prompt,
    "ans5": ans5_prompt,
    "ans6": ans6_prompt,
    "ans7": ans7_prompt,
    "ans8": ans8_prompt,
    "ans9": ans9_prompt,
    
}


def load_data(dataname):
    
    train_data = []
    test_data = []

    if dataname == 'truthfulqa':
        ## load test data
        test = load_dataset("truthfulqa/truthful_qa", "generation", split='validation')
        for item in test:
            question = item['question'].strip()
            if question[-1] != '?':
                question = question +'?'
            test_data.append({
                "question": item['question'],
                "target": item['best_answer']
            }) 
        
    if dataname == "popqa":
        ds = load_dataset("akariasai/PopQA", split="test")
        test = ds.shuffle(seed=42).select(range(1000))
        
        for item in test:
            question = item['question'].strip()
            if question[-1] != '?':
                question = question +'?'
            test_data.append({
                "question": item['question'],
                "target": ast.literal_eval(item['possible_answers'])[0]
            })

        train = ds.shuffle(seed=84).select(range(3000))
        test_ids = [test_item['id'] for test_item in test]
        for item in train:
            if item['id'] not in test_ids:
                question = item['question'].strip()
                if question[-1] != '?':
                    question = question +'?'
                train_data.append({
                    "question": item['question'],
                    "target": ast.literal_eval(item['possible_answers'])[0]
                })
                if len(train_data) == 2000:
                    break

    if dataname in ['triviaqa', 'sciq', 'wikiqa', 'nq']:

        dataset_path = os.path.join(data_path, dataname)

        ## load train data
        train = pd.read_csv(os.path.join(dataset_path, "train.csv"), encoding='utf-8')
        
        for i in train.index.tolist():
            train_data.append({
                "question": train['question'][i],
                "target": train['answer'][i][1:-1]
            })

        ## load test data
        test = []
        with open(os.path.join(dataset_path, "testLlama-2-7b-hftemperature_1topp_1.0_num_demos_15_answers.1.reeval.semantic_uncertainty.gpt4_correctness.jsonl"), encoding='utf-8') as f:
            for line in f:
                test.append(json.loads(line))
    
        for item in test:
            question = item['question'].strip()
            if question[-1] != '?':
                question = question +'?'
            test_data.append({
                "question": question,
                "target": item['gold_answer']
            })

    return train_data, test_data

class QDataset(Dataset):
    def __init__(self, data, ans_type, tokenizer=None):

        
        self.data = data
        self.ans_type = ans_type
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        question = self.data[idx]['question'].strip()

        prompts = [{"role": "user", "content": answer_type_dict[self.ans_type]() + question}]
        
        if self.tokenizer is not None:
            prompts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

        return prompts

        
## returning greedy-decoding answers and uses different prompt to elicit confidence
class QADataset(Dataset):

    def __init__(self, prompt_type, responses, tokenizer=None):

        self.tokenizer = tokenizer
        self.prompt_type = prompt_type
        self.questions = responses['question'].tolist()
        self.answers = responses['response'].tolist()

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):

        question = self.questions[idx].strip()

        answer = self.answers[idx]
        prompts = [{"role": "user", "content": ans0_prompt() + question},
                    {"role": "assistant", "content": str(answer)},
                    {"role": "user", "content": prompt_type_dict[self.prompt_type]()}]

        if self.tokenizer is not None:
            prompts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

        return prompts

## returning sampling answers    
class SQADataset(Dataset):
    
    def __init__(self, responses, idx, tokenizer=None):

        self.tokenizer = tokenizer
        self.questions = responses['question'].tolist()
        self.answers = responses['sampling_'+ str(idx) + '_response'].tolist()

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):

        question = self.questions[idx].strip()

        answer = self.answers[idx]
        prompts = [{"role": "user", "content": ans0_prompt() + question},
                    {"role": "assistant", "content": str(answer)},
                    {"role": "user", "content": prompt_type_dict['prob']()}]

        if self.tokenizer is not None:
            prompts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

        return prompts
 
class p_true_Dataset(Dataset):

    def __init__(self, prompt_type, responses, tokenizer=None, alter_answer='original'):

        self.tokenizer = tokenizer
        self.prompt_type = prompt_type
        self.questions = responses['question'].tolist()
        self.answers = responses['response'].tolist()
        self.targets = responses['target'].tolist()
        self.random_answers = responses['random_answer'].tolist()
        if alter_answer[:8] == 'sampling':
            self.sampling_answers = responses[alter_answer + '_response'].tolist()
        self.alter_answer = alter_answer
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):

        question = self.questions[idx].strip()
        prompt = answer_type_dict[self.prompt_type]() + question
        
        if self.alter_answer == 'original':
            answer = self.answers[idx]
        elif self.alter_answer == 'target':
            answer = self.targets[idx]
        elif self.alter_answer == 'random':
            answer = self.random_answers[idx]
        elif self.alter_answer == 'abstain':
            answer = abstain_answers[0]
        elif self.alter_answer[:8] == 'sampling':
            answer = self.sampling_answers[idx]
        else:
            print('not exsit answer type!')
        prompts = [{"role": "user", "content": f"Question:\n\n{prompt}\n\nProposed Answer: {answer}\n\nIs the proposed answer:\n\tTrue or\n\tFalse?\nThe proposed answer is: "}]
                    

        if self.tokenizer is not None:
            prompts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

        return prompts

## returning greedy-decoding answers elicited with different prompt 
class PQADataset(Dataset):

    def __init__(self, ans_type, responses, tokenizer=None, conf_type=None):

        self.tokenizer = tokenizer
        self.ans_type = ans_type
        self.questions = responses['question'].tolist()
        self.answers = responses[ans_type+'_response'].tolist()
        self.conf_type = conf_type

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):

        question = self.questions[idx].strip()

        answer = self.answers[idx]
        if self.conf_type is not None:
            prompts = [{"role": "user", "content": answer_type_dict[self.ans_type]() + question},
                    {"role": "assistant", "content": str(answer)}]
        else:
            prompts = [{"role": "user", "content": answer_type_dict[self.ans_type]() + question},
                        {"role": "assistant", "content": str(answer)},
                        {"role": "user", "content": origin_prob_prompt()}]

        if self.tokenizer is not None:
            prompts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

        if self.conf_type is not None:
            return prompts, str(answer)
        else:
            return prompts

import math

class CaliDataset(Dataset):
    def __init__(self, response, alter_answer=None):
        if alter_answer is None:
            self.confidences = response['response_prob']
        elif alter_answer == 'random':
            self.confidences = response['random_answer_prob']
        elif alter_answer == 'abstain':
            self.confidences = response['abstain_answer_0_prob']
        elif alter_answer == 'target':
            self.confidences = response['target_answer_prob']
        elif alter_answer[:8] == 'sampling':
            self.confidences = response[alter_answer + '_response_prob']
        else:
            print('not implemented!')
        self.acces = response['prom_score']
        
        
    def __len__(self):
        # this should return the size of the dataset
        return len(self.confidences)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        if math.isnan(self.confidences[idx]):
            confidence = 0
        else: 
            confidence = self.confidences[idx]
        acc = self.acces[idx]
    
        return confidence, acc

class Q_random_ADataset(Dataset):

    def __init__(self, prompt_type, responses, tokenizer=None, conf_type=None):

        self.tokenizer = tokenizer
        self.prompt_type = prompt_type
        self.questions = responses['question'].tolist()
        self.answers = responses['response'].tolist()
        self.targets = responses['target'].tolist()
        self.random_answers = responses['random_answer'].tolist()
        self.conf_type = conf_type

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):

        question = self.questions[idx].strip()

        # ### get a random answer
        # filtered_answers = [a for a in self.answers if a not in [self.answers[idx], self.targets[idx]]]
        # # random.seed(seed)
        # answer =  random.choice(filtered_answers)
        answer = self.random_answers[idx]
        if self.conf_type is not None:
            prompts = [{"role": "user", "content": ans0_prompt() + question},
                    {"role": "assistant", "content": str(answer)}]
            
        else:
            prompts = [{"role": "user", "content": ans0_prompt() + question},
                    {"role": "assistant", "content": str(answer)},
                    {"role": "user", "content": prompt_type_dict[self.prompt_type]()}]

        if self.tokenizer is not None:
            prompts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        
        if self.conf_type is not None:
            return prompts, answer
        else:
            return prompts


class AbstainDataset(Dataset):
    def __init__(self, abstain_idx, prompt_type, responses, tokenizer=None, conf_type=None):
        
        self.questions = responses['question'].tolist()
        self.prompt_type = prompt_type
        self.tokenizer = tokenizer
        self.abstain_idx = abstain_idx
        self.conf_type = conf_type

    def __len__(self):
        # this should return the size of the dataset
        return len(self.questions)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        
        question = self.questions[idx].strip()
        
        answer = abstain_answers[self.abstain_idx]

        if self.conf_type is not None:
            prompts = [{"role": "user", "content": ans0_prompt() + question},
                    {"role": "assistant", "content": str(answer)}]
            
        else:
            prompts = [{"role": "user", "content": ans0_prompt() + question},
                        {"role": "assistant", "content": str(answer)},
                        {"role": "user", "content": prompt_type_dict[self.prompt_type]()}]

        if self.tokenizer is not None:
            prompts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

        if self.conf_type is not None:
            return prompts, answer
        else:
            return prompts
    
    
class TargetDataset(Dataset):
    def __init__(self, prompt_type, responses, tokenizer=None, conf_type=None):
        
        self.questions = responses['question'].tolist()
        self.targets = responses['target'].tolist()
        self.prompt_type = prompt_type
        self.tokenizer = tokenizer
        self.conf_type = conf_type

    def __len__(self):
        # this should return the size of the dataset
        return len(self.questions)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        question = self.questions[idx].strip()
        
        answer = self.targets[idx]

        if self.conf_type is not None:
            prompts = [{"role": "user", "content": ans0_prompt() + question},
                    {"role": "assistant", "content": str(answer)}]
            
        else:
            prompts = [{"role": "user", "content": ans0_prompt() + question},
                    {"role": "assistant", "content": str(answer)},
                    {"role": "user", "content": prompt_type_dict[self.prompt_type]()}]
            
        if self.tokenizer is not None:
            prompts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

        if self.conf_type is not None:
            return prompts, answer
        else:
            return prompts

class hf_EvalDataset(Dataset):
    def __init__(self, response, apply_temp=False):
        
        self.questions = response['question'].tolist()
        self.answers = response['response'].tolist()
        self.targets = response['target'].tolist()
        self.apply_temp = apply_temp
    
    def _get_conversation_prompt(self, messages):
        """
        From filled prompt, convert it into llama-2 conversation prompt
        """
        from fastchat.conversation import get_conv_template  # optional dependency

        conv = get_conv_template("mistral")

        for message in messages:
            if message["role"] == "system":
                conv.set_system_message(message["content"])
            elif message["role"] == "user":
                conv.append_message(conv.roles[0], message["content"])

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
    
    def __len__(self):
        # this should return the size of the dataset
        return len(self.questions)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        question = self.questions[idx]
        answer = self.answers[idx]
        target = self.targets[idx]

        ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

        ABSOLUTE_PROMPT = f"""###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 1, and a score rubric representing an evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 0 and 1. You should refer to the score rubric.
        3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 0 and 1)\"
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        {ans0_prompt() + question}

        ###Response to evaluate:
        {answer}

        ###Reference Answer (Score 1):
        {target}

        ###Score Rubrics:
        Score 0: To address the instruction, the response is wrong or not semantically equivalent to the reference answer. 
        Score 1: To address the instruction, the response is correct or semantically equivalent to the reference answer. 

        ###Feedback:  """
        
        messages = [
                {"role": "system", "content": ABS_SYSTEM_PROMPT},
                {"role": "user", "content": ABSOLUTE_PROMPT},
            ]
        
        if self.apply_temp:
            prompts = self._get_conversation_prompt(messages)
        else:
            prompts = messages

        # prompts = self.tokenizer.apply_chat_template(messages, tokenize=False)

        return prompts