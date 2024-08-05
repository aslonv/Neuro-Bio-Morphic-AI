from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class LanguageReasoner:
    def __init__(self, model_name="gpt2-medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_reasoning(self, context, max_length=100):
        prompt = f"Given the context: {context}\nReasoning:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        reasoning = generated_text.split("Reasoning:")[1].strip()
        return reasoning

    def extract_features(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Use the mean of the last hidden state as features
        features = outputs.hidden_states[-1].mean(dim=1)
        return features