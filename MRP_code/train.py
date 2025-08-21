import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from transformers import DataCollatorForLanguageModeling
from torch.cuda.amp import autocast, GradScaler
import math
import random
from model import *
from data import *

def init_W(ul_v,rt_v,dimension):
    
    def compute_projection_matrix(vectors):

        vectors_float32 = vectors.float()  # Convert float32 to calculate QR code
        Q_32 = torch.linalg.qr(vectors_float32.T, mode='reduced')[0]  
        Q = Q_32.to(dtype=torch.bfloat16)
        P = Q @ Q.T

        return P
    
    def pca_torch(tensor, k, center=False):
    
        if center:
            tensor = tensor - tensor.mean(dim=0)  #Centralization
            
        tensor_32 = tensor.float()
        cov = torch.matmul(tensor_32.T, tensor_32) / (tensor_32.shape[0] - 1) 
        _, S, V = torch.linalg.svd(cov)
        components = V[:k, :]
        
        return components.to(dtype=torch.bfloat16)
    
    P = compute_projection_matrix(ul_v)   
    
    return pca_torch(rt_v - rt_v@P, dimension, center=False)


def train_unlearn(model, tokenizer, lora_optimizer, layer, ul_number, learning_rate, dimension, unlearn_epoch, unlearn_topic, W_previous, retain_rate):
    
    unlearn_subject = ['natural science']

    ScienceQA_dataset = load_dataset('csv', data_files='/root/data/ScienceQA.csv')['train']

    all_unlearn_loader = load_ScienceQA_data(tokenizer, ScienceQA_dataset, unlearn_subject, unlearn_topic, begin = 0, end = ul_number, seed = 42, max_token = 160, batch_size = 5)
    validation_loader = load_ScienceQA_data(tokenizer, ScienceQA_dataset, unlearn_subject, unlearn_topic, begin = 8000, end = 10000, seed = 42, max_token = 160, batch_size = 5)    

    W_all = []
        
    # Compute the initialization of W
    
    ul_all = []
    rt_all = []
    
    for i in range(len(layer)): 
        ul_all.append([])
        rt_all.append([])
        
    for num_batch, batch in enumerate(validation_loader):

        input_ids = batch["retain_input_ids"].to(model.device)
        attention_mask = batch["retain_attention_mask"].to(model.device)
        labels = batch["retain_labels"].to(model.device)
    
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
    
        with torch.no_grad():
            for i in range(len(layer)):      
                for h in outputs.hidden_states[layer[i]]:
                    for h0 in h:  
                        rt_all[i].append(h0)
                        
        lora_optimizer.zero_grad()
        torch.cuda.empty_cache()    

    for num_batch, batch in enumerate(validation_loader):

        input_ids = batch["unlearn_input_ids"].to(model.device)
        attention_mask = batch["unlearn_attention_mask"].to(model.device)
        labels = batch["unlearn_labels"].to(model.device)
    
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
    
        with torch.no_grad():
            for i in range(len(layer)):      
                for h in outputs.hidden_states[layer[i]]:
                    for h0 in h:  
                        ul_all[i].append(h0)
                        
        lora_optimizer.zero_grad()
        torch.cuda.empty_cache()

    ul_v = [torch.stack(i, dim=0) for i in ul_all]
    rt_v = [torch.stack(i, dim=0) for i in rt_all]
    
    for i in range(len(layer)):  
        W_all.append(init_W(ul_v[i],rt_v[i],dimension))
        
    lora_optimizer.zero_grad()
    torch.cuda.empty_cache()
    
    for epoch in range(unlearn_epoch):
                
        for num_batch, batch in enumerate(all_unlearn_loader):
            
            
            print(f"Batch {num_batch + 1}/{len(all_unlearn_loader)}")

            retain_input_ids = batch["retain_input_ids"].to(model.device)
            retain_attention_mask = batch["retain_attention_mask"].to(model.device)
            retain_labels = batch["retain_labels"].to(model.device)
            unlearn_input_ids = batch["unlearn_input_ids"].to(model.device)
            unlearn_attention_mask = batch["unlearn_attention_mask"].to(model.device)
            unlearn_labels = batch["unlearn_labels"].to(model.device)
            
            perturb_hooks, W = register_x_hook(model, layer, W_all, W_previous)
            
            with autocast(dtype=torch.bfloat16):
                lora_optimizer.zero_grad()
                unlearn_outputs = model(input_ids=unlearn_input_ids, attention_mask=unlearn_attention_mask, labels=unlearn_labels)
                unlearn_loss = -unlearn_outputs.loss
                retain_outputs = model(input_ids=retain_input_ids, attention_mask=retain_attention_mask, labels=retain_labels)
                retain_loss = retain_outputs.loss
                loss = unlearn_loss + retain_loss*retain_rate
                loss.backward()  
                
            lora_optimizer.zero_grad()
            lora_optimizer.step()
            
            with torch.no_grad():
                for W in W_all:
                    W -= learning_rate * W.grad
                    W.grad.zero_()  
                    
            lora_optimizer.zero_grad()
            torch.cuda.empty_cache()
            
            for perturb_hook in perturb_hooks:
                perturb_hook.remove() 
            del perturb_hooks
    
            
        perturb_hooks, W = register_x_hook(model, layer, W_all, W_previous)
        
        torch.cuda.empty_cache()
        
        for perturb_hook in perturb_hooks:
            perturb_hook.remove()
        del perturb_hooks
        
    lora_optimizer.zero_grad()    
    torch.cuda.empty_cache()

    W_new = []

    for W in W_all:
        W_new.append(W.detach())

    lora_optimizer.zero_grad()
    torch.cuda.empty_cache()
        
    return W_new


def continual_train_unlearn(layer, ul_number, learning_rate, dimension, unlearn_epoch, order, retain_rate):
        
    model_path = "Llama-2-7b-chat-hf"
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Ensure that the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model, lora_params = load_lora_model(model_path)
    lora_optimizer = load_optimizer(lora_params, 0.0)
    
    llama2_template = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content }}{% endif %}{% endfor %}"""
    tokenizer.chat_template = llama2_template

    W_previous = []
    
    for i in range(len(layer)):
        W_previous.append(torch.zeros(1, 4096, dtype=torch.bfloat16, device='cuda'))
        
    
    for i in range(len(order)):
        unlearn_topic = order[i:i+1]

        W_trained = train_unlearn(model, tokenizer, lora_optimizer, layer, ul_number, learning_rate, dimension, unlearn_epoch, unlearn_topic, W_previous, retain_rate)

        Q_trained = []
        for i, W_t in enumerate(W_trained):
            W_previous[i] = torch.cat((W_previous[i], W_t), dim=0)
            
        torch.cuda.empty_cache()


continual_train_unlearn(layer = random.sample(range(0, 31), 2), ul_number = 5000, learning_rate = 2e-4, dimension = 2, unlearn_epoch = 2, order = random.shuffle(['chemistry','biology','physics','earth-science']), retain_rate=1.2)
