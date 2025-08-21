from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
import torch

def load_lora_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1
    )
    peft_model = get_peft_model(model, lora_config)

    lora_params = [p for n, p in peft_model.named_parameters() if "lora" in n]

    return peft_model, lora_params

def load_optimizer(lora_params, w_lr):
    optimizer_lora = AdamW(lora_params, lr=w_lr)
    return optimizer_lora

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Pre set tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Ensure padding is on the right side (at the end of the sequence)
    tokenizer.truncation_side = "right"  # Ensure that the truncation is also on the right side
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() }}{% endif %}{% endfor %}"

    return tokenizer

def register_x_hook(model, x_register_layers, W_all, W_old):

    Q_all = []
    for i, W in enumerate(W_all):

        if not isinstance(W, torch.Tensor):
            W = torch.tensor(W, dtype=torch.float32, requires_grad=True)
        elif not W.requires_grad:
            W.requires_grad_(True)

        W_float32 = torch.cat((W_old[i], W), dim=0).float()  # Convert float32 to calculate QR code
        Q_float32 = torch.linalg.qr(W_float32.T, mode='reduced')[0].T 
        Q_all.append(Q_float32.to(dtype=torch.bfloat16))  # Return to bfloat16        
    
    # Register perturb_hook to output to the specified layer 
    layers = [model.base_model.model.model.layers[x] for x in x_register_layers] 

    # Create a closure for each hook to capture the corresponding Q matrix
    hook_fns = []
    for i, Q in enumerate(Q_all):
        def make_hook_fn(Q):
            def perturb_hook_fn(module, input, output):
                output_list = list(output)
                hidden_state = output_list[0]
                
                modified_hidden_state = hidden_state - (hidden_state @ Q.t() @ Q)
                output_list[0] = modified_hidden_state
                return tuple(output_list)
            return perturb_hook_fn
        
        hook_fns.append(make_hook_fn(Q))
    
    # Register corresponding hooks for each layer
    perturb_hooks = [layer.register_forward_hook(hook_fn) 
                    for layer, hook_fn in zip(layers, hook_fns)]
    return perturb_hooks, W_all  # Return all W matrices

def set_optimizer(optimizer, requires_grad):
    for param in optimizer.param_groups[0]['params']:
        param.requires_grad = requires_grad