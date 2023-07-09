from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

class RobertaInferer:
    def __init__(self, path_to_model: str):
        # torch.cuda.is_available()
        # device=torch.cuda.current_device()
        
        tokenizer = AutoTokenizer.from_pretrained(path_to_model)
        model = AutoModelForSequenceClassification.from_pretrained(path_to_model)
        self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def infer(self, text):
        return self.pipe([text], truncation=True, max_length=512)