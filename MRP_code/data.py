from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import Dataset
import torch

class LabeledDataset(Dataset):
    def __init__(self, encoded_dataset):
        self.unlearn_input_ids = encoded_dataset["unlearn_input_ids"]
        self.unlearn_attention_mask = encoded_dataset["unlearn_attention_mask"]
        self.unlearn_labels = encoded_dataset["unlearn_labels"]
        self.retain_input_ids = encoded_dataset["retain_input_ids"]
        self.retain_attention_mask = encoded_dataset["retain_attention_mask"]
        self.retain_labels = encoded_dataset["retain_labels"]
    
    def __len__(self):
        return len(self.unlearn_input_ids)
    
    def __getitem__(self, idx):
        return {
            "unlearn_input_ids": self.unlearn_input_ids[idx],
            "unlearn_attention_mask": self.unlearn_attention_mask[idx],
            "unlearn_labels": self.unlearn_labels[idx],
            "retain_input_ids": self.retain_input_ids[idx],
            "retain_attention_mask": self.retain_attention_mask[idx],
            "retain_labels": self.retain_labels[idx]
        }

def load_ScienceQA_data(tokenizer, dataset, unlearn_subject, unlearn_topic, begin = None, end = None, seed = None, max_token = 160, batch_size = 16):

    answer_dict = {0:'A',1:'B',2:'C',3:'D'}
    
    dataset = dataset.select(range(begin,end)).shuffle(seed=seed)
    # Data preprocessing
    def ScienceQA_preprocess_function(examples):
        # 1. Build a complete dialogue text (including system/user/assistant roles)
        unlearn_input_texts = []
        unlearn_label_texts = []
        retain_input_texts = []
        retain_label_texts = []
        for q, c, a, s, t in zip(examples["question"], examples["choices"], examples["answer"], examples["subject"], examples["topic"]):

            if s in unlearn_subject and t in unlearn_topic:
            
                unlearn_input_messages = [
                    {"role": "system", "content": "Choose the correct answer's letter,just like (A), (B), (C) or (D)."},
                    {"role": "user", "content": 'Question:\n' + q + '\n' + c},
                    {"role": "assistant", "content": 'The letter of correct answer is : ('}  
                ]
                
                unlearn_full_messages = [
                    {"role": "system", "content": "Choose the correct answer's letter,just like (A), (B), (C) or (D)."},
                    {"role": "user", "content": 'Question:\n' + q + '\n' + c},
                    {"role": "assistant", "content": 'The letter of correct answer is : ('+ f"{answer_dict[a]}" }  
                ]
                # Automatically add tags such as<| im_start |>using apply_chat_template
                unlearn_full_text = tokenizer.apply_chat_template(unlearn_full_messages, tokenize=False)
                unlearn_input_text = tokenizer.apply_chat_template(unlearn_input_messages, tokenize=False)  
                unlearn_input_texts.append(unlearn_input_text)
                unlearn_label_texts.append(unlearn_full_text)  # Used for extracting labels

            if s not in unlearn_subject:

                retain_input_messages = [
                    {"role": "system", "content": "Choose the correct answer's letter,just like (A), (B), (C) or (D)."},
                    {"role": "user", "content": 'Question:\n' + q + '\n' + c},
                    {"role": "assistant", "content": 'The letter of correct answer is : ('} 
                ]
                
                retain_full_messages = [
                    {"role": "system", "content": "Choose the correct answer's letter,just like (A), (B), (C) or (D)."},
                    {"role": "user", "content": 'Question:\n' + q + '\n' + c},
                    {"role": "assistant", "content": 'The letter of correct answer is : ('+ f"{answer_dict[a]}" } 
                ]
                # Automatically add tags such as<| im_start |>using apply_chat_template
                retain_full_text = tokenizer.apply_chat_template(retain_full_messages, tokenize=False)
                retain_input_text = tokenizer.apply_chat_template(retain_input_messages, tokenize=False)  
                retain_input_texts.append(retain_input_text)
                retain_label_texts.append(retain_full_text)  # Used for extracting labels

        len_text = min(len(unlearn_input_texts),len(retain_input_texts))
        unlearn_input_texts = unlearn_input_texts[:len_text]
        unlearn_label_texts = unlearn_label_texts[:len_text]
        retain_input_texts = retain_input_texts[:len_text]
        retain_label_texts = retain_label_texts[:len_text]

        unlearn_model_inputs = tokenizer(
            unlearn_input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        unlearn_full_encoded = tokenizer(
            unlearn_label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        unlearn_labels = unlearn_full_encoded["input_ids"]
        unlearn_input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in unlearn_model_inputs["input_ids"]]
        for i, input_len in enumerate(unlearn_input_lengths):
            if input_len < max_token:
                unlearn_labels[i, :input_len] = -100            
            if input_len >= max_token:
                unlearn_labels[i,:] = -100
        unlearn_labels[unlearn_labels == tokenizer.pad_token_id] = -100
        
        
        retain_model_inputs = tokenizer(
            retain_input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        retain_full_encoded = tokenizer(
            retain_label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        retain_labels = retain_full_encoded["input_ids"]
        retain_input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in retain_model_inputs["input_ids"]]
        for i, input_len in enumerate(retain_input_lengths):
            if input_len < max_token:
                retain_labels[i, :input_len] = -100            
            if input_len >= max_token:
                retain_labels[i,:] = -100
        retain_labels[retain_labels == tokenizer.pad_token_id] = -100

        
        return {
            "unlearn_input_ids": unlearn_model_inputs["input_ids"],
            "unlearn_attention_mask": unlearn_model_inputs["attention_mask"],
            "unlearn_labels": unlearn_labels,
            "retain_input_ids": retain_model_inputs["input_ids"],
            "retain_attention_mask": retain_model_inputs["attention_mask"],
            "retain_labels": retain_labels
        }
        
    encoded_dataset = dataset.map(ScienceQA_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["unlearn_input_ids", "unlearn_attention_mask", 'unlearn_labels', "retain_input_ids", "retain_attention_mask", "retain_labels"])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader