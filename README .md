
# CVE Impact and Mitigation Analysis Model

This project utilizes the Unsloth library to fine-tune a language model on cybersecurity data, specifically focusing on the analysis of Common Vulnerabilities and Exposures (CVE). The goal is to predict the impact and suggest mitigation strategies for various CVE entries.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. This project also requires the following packages:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
```

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/cve-impact-analysis.git
cd cve-impact-analysis
```

### Data Preparation

Data files are expected in the following format and directory:

- `cve.csv`
- `analysis_cverecommendation.csv`
- `system_configuration.csv`
- `vulnerability_analysis.csv`

Ensure these files are located in `/mnt/data/`.

### Usage

1. **Model Setup**:
   Load the Unsloth model pre-configured for CVE data:

   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="unsloth/llama-3-8b-bnb-4bit",
       max_seq_length=2048,
       dtype=None,
       load_in_4bit=True
   )
   EOS_TOKEN = tokenizer.eos_token
   ```

2. **Data Loading**:
   Load and preprocess your data:

   ```python
   import pandas as pd
   files = {
       "cve.csv": "/mnt/data/cve.csv",
       ...
   }
   dfs = {name: pd.read_csv(path) for name, path in files.items()}
   ```

3. **Data Processing**:
   Tokenize and format the data for training:

   ```python
   def normalize_text(text):
       return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())).strip()

   final_data['input_ids'] = final_data.apply(lambda x: tokenizer(x['summary'], padding='max_length', truncation=True, max_length=512)['input_ids'])
   ```

4. **Training**:
   Configure and run the training session:

   ```python
   from transformers import TrainingArguments, Trainer
   trainer = Trainer(
       model=model,
       args=TrainingArguments(
           output_dir='./outputs',
           num_train_epochs=3,
           per_device_train_batch_size=2,
           gradient_accumulation_steps=4,
           ...
       ),
       train_dataset=train_dataset,
       eval_dataset=val_dataset
   )
   trainer.train()
   ```

### Monitoring

Monitor GPU usage and model training progress:

```python
gpu_stats = torch.cuda.get_device_properties(0)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
```

## Contributing

Contributions to enhance the model or improve the accuracy of predictions are welcome. Please fork the repository and submit a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - email@example.com
