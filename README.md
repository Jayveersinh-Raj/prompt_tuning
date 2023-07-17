# Prompt tuning
This is the repo for prompt tuning a language model to improve the given prompt (vague) using Parameter efficient finetuning (PEFT) and (LoRA). 

# Dataset description
This is the dataset that contains bad prompts with corresponding improved prompts. It can be used to improve the vague prompts given by a user for a language model. The dataset is human created at our organization. 

# How to use dataset?
[🤗](https://huggingface.co/datasets/Jayveersinh-Raj/bad-improved-prompt-pairs) It can be used from this link directly from huggingface. 

# Model description
`Base Model` Bloom-7b

`type` Decoder only

# How to use the model?
[🤗](https://huggingface.co/autopilot-ai/bloom-prompt-tuner) It can be used from this link directly from huggingface, or copy the below code.

    import torch
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    peft_model_id = "autopilot-ai/bloom-prompt-tuner"
    config = PeftConfig.from_pretrained(peft_model_id)
    model_loaded = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model_loaded, peft_model_id)

    # Inference
    from IPython.display import display, Markdown

    def make_inference(prompt: str) -> None:
      """
      This is a function to make inference on a prompt tuner model to improve the passed prompt

      Parameters:
      -----------
        prompt(str): The initial prompt to improve

      Returns:
      -----------
        None
      """
  
      batch = tokenizer(f"### Bad prompt\n{prompt}\n\n### Improved/tuned prompt\n", return_tensors='pt')

      with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=200)

      display(Markdown((tokenizer.decode(output_tokens[0], skip_special_tokens=True))))
  
    # Calling the function for inference
    prompt = "write a djitkra algorithm in python please also I want it optimized you know"
    make_inference(prompt)


# Example output
![image](https://github.com/Jayveersinh-Raj/prompt_tuning/assets/69463767/bf713891-6a46-41a5-af58-7f3614ba3d20)

