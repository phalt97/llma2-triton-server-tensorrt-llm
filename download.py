from transformers import LlamaTokenizer, LlamaForCausalLM

# tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", add_eos_token=True)
# model =  LlamaForCausalLM.from_pretrained("huggyllama/llama-7b")
#/root/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16

# Llama2
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", add_eos_token=True)
model =  LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
#/root/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496