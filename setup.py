from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Download the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Download the GPT-2 model
model = TFGPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

