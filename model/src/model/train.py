import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,AutoTokenizer,Trainer,
    TrainingArguments,DataCollatorForLanguageModeling,
)
from pathlib import Path
import os

DATA = Path(__file__).parent / "data.jsonl"
MODEL = "google/gemma-3-270m"

def train():
    print("setting one gpu for training")
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    print("checking gpu")
    if torch.cuda.is_available() == True:
        print("gpu found!")
    else:
        print("gpu not found")
        return

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    ds = load_dataset("json", data_files=str(DATA))["train"]
    ds = ds.map(
        lambda ex: tok(ex["text"]+tok.eos_token,truncation=True,max_length=1024),
        remove_columns=ds.column_names,
    )
    args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        #warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True, # this argument, trades speed for vram
        logging_steps = 10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok,mlm=False),
    )
    print("training started")
    trainer.train()

    print("training ended")
    trainer.save_model("out") # writes model.safetensors+config.json
    tok.save_pretrained("out") # writes tokenizer.json and those things
