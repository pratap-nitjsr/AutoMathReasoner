from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

def main():
    max_seq_length = 1024
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "llama-3-8b-instruct", 
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # We use a subset of GSM8K style data to warm start the reasoning format
    # In practice, this would load a custom generated dataset locally
    try:
        dataset = load_dataset("gsm8k", "main", split="train[:5%]")
    except Exception:
        # Fallback dummy dataset
        dataset = load_dataset("json", data_files={"train": ["dummy.json"]}, split="train")
        
    def formatting_prompts_func(examples):
        texts = []
        for q, a in zip(examples['question'], examples['answer']):
            # Assuming 'answer' has reasoning and then '#### answer'
            parts = a.split("####")
            reasoning = parts[0].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else ""
            
            text = f"Problem: {q}\nReasoning: {reasoning}\nAnswer: {final_answer}"
            texts.append(text)
        return { "text" : texts }
        
    dataset = dataset.map(formatting_prompts_func, batched = True)
    
    training_args = SFTConfig(
        output_dir="sft_outputs",
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        per_device_train_batch_size=2,
        max_steps=100,
        learning_rate=2e-5,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )
    
    print("Starting SFT Warm-Start...")
    trainer.train()

if __name__ == "__main__":
    main()
