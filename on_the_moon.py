# Initial installations
""" %%capture
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes """



# Import necessary libraries
from unsloth import FastLanguageModel
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import re



# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # Auto detection
    load_in_4bit=True
)

# Define EOS_TOKEN immediately after loading the tokenizer

EOS_TOKEN = tokenizer.eos_token  
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

  # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
      "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
  ] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
      model_name = "unsloth/llama-3-8b-bnb-4bit",
      max_seq_length = max_seq_length,
      dtype = dtype,
      load_in_4bit = load_in_4bit,
      # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
  )
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
# Load data and merge
files = {
    "cve.csv": "/mnt/data/cve.csv",
    "analysis_cverecommendation.csv": "/mnt/data/analysis_cverecommendation.csv",
    "system_configuration.csv": "/mnt/data/system_configuration.csv",
    "vulnerability_analysis.csv": "/mnt/data/vulnerability_analysis.csv"
}
dfs = {name: pd.read_csv(path) for name, path in files.items()}
merged_cve_vuln = pd.merge(dfs['cve.csv'], dfs['vulnerability_analysis.csv'], left_on='cve_id', right_on='id_cve', how='left')
merged_cve_vuln_sys = pd.merge(merged_cve_vuln, dfs['system_configuration.csv'], left_on='system_configuration_id', right_on='id', how='left', suffixes=('_vuln', '_sys'))
final_merge = pd.merge(
    merged_cve_vuln_sys,
    dfs['analysis_cverecommendation.csv'],
    left_on='id_y',
    right_on='vulnerability_analysis_id',
    how='left',
    suffixes=('_merged', '_rec')
)
final_data = final_merge[[
    'cve_id', 'summary', 'vulnerability_type', 'affected_components', 'impact_level',
    'description', 'system_info', 'recommendations', 'mitigation_measures'
]]
final_data.rename(columns={
    'cve_id': 'CVE ID', 'summary': 'Summary', 'vulnerability_type': 'Vulnerability Type',
    'affected_components': 'Affected Components', 'impact_level': 'Impact Level',
    'description': 'Description', 'system_info': 'System Info',
    'recommendations': 'Recommendations', 'mitigation_measures': 'Mitigation Measures'
}, inplace=True)# Preprocess and tokenize data
def normalize_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())).strip()

def formatting_prompts_func(row):
    text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Describe the CVE impact and mitigation.

### Input:
CVE ID: {row['CVE ID']} - Summary: {row['Summary']} - Vulnerability Type: {row['Vulnerability Type']} - Description: {row['Description']} - System Info: {row['System Info']} - Impact Level: {row['Impact Level']}

### Response:
{row['Recommendations']} Mitigation Measures: {row['Mitigation Measures']}
""" + EOS_TOKEN
    return normalize_text(text)

final_data['formatted_text'] = final_data.apply(formatting_prompts_func, axis=1)
final_data['input_ids'] = final_data['formatted_text'].apply(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=512)['input_ids'])
def formatting_prompts_func(row):
    """
    Formats the prompts and labels for a given row of data.

    Args:
        row (dict): A dictionary containing the row data.

    Returns:
        dict: A dictionary containing the formatted inputs and labels.

    Raises:
        None

    Example:
        row = {
            'Summary': 'This is a summary.',
            'Description': 'This is a description.',
            'Recommendations': 'These are recommendations.',
            'Mitigation Measures': 'These are mitigation measures.'
        }
        result = formatting_prompts_func(row)
        print(result)
        # Output:
        # {
        #     'input_ids': tensor([...]),
        #     'attention_mask': tensor([...]),
        #     'labels': tensor([...])
        # }
    """

    # Define the prompt structure
    prompt_text = f"Describe the CVE impact and mitigation based on the following details: {row['Summary']} {row['Description']}"
    # Generate labels from the expected output
    labels_text = f"{row['Recommendations']} Mitigation Measures: {row['Mitigation Measures']}"

    # Tokenize prompt and labels
    inputs = tokenizer(prompt_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    labels = tokenizer(labels_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")['input_ids']

    # Replace padding token id in labels with -100
    labels = [(-100 if token == tokenizer.pad_token_id else token) for token in labels.squeeze()]

    return {
        'input_ids': inputs['input_ids'].squeeze(),
        'attention_mask': inputs['attention_mask'].squeeze(),
        'labels': torch.tensor(labels)
    }

# Apply this function to your DataFrame
processed_data = [formatting_prompts_func(row) for index, row in final_data.iterrows()
] 
from torch.utils.data import random_split

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Convert list of dictionaries into a Dataset
dataset = CustomDataset(processed_data)

# Split the dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])




from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Train the model
trainer_stats = trainer.train()

model.save_pretrained("/mnt/data/alaeddine_lora_model") # Local saving
tokenizer.save_pretrained("/mnt/data/alaeddine_lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

if True:  # Adjust the condition based on your use case
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/mnt/data/alaeddine_lora_model",  # Adjust the model name as required
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

# Define the prompt format clearly
alpaca_prompt = "What is a famous tall tower in Paris?"

# Preparing the input for the model
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "What is a famous tall tower in Paris?", # instruction
            "", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")

# Generating output from the model
outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
result = tokenizer.batch_decode(outputs)

print(result)  # Print the output from the model
