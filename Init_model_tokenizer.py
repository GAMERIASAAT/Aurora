from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if tokenizer and model are already saved on disk
tokenizer_path = "tokenizer"
model_path = "model"
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
except:
    # If tokenizer and model are not saved on disk, download them
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # Save tokenizer and model to disk
    tokenizer.save_pretrained(tokenizer_path)
    model.save_pretrained(model_path)
