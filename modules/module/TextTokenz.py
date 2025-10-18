class TextTokenCounter:
    def __init__(self, tokenizer):
        """
        Initialize with a tokenizer instance from the project models.
        The tokenizer is used as in the dataloaders: tokenizer(text, return_tensors="pt")['input_ids'].shape[1]
        """
        self.tokenizer = tokenizer

    def count_tokens(self, text):
        """
        Count the number of tokens in the given text using the tokenizer.
        Matches how the trainer counts tokens in texts for training.
        """
        if not text:
            return 0
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
        return inputs['input_ids'].shape[1]

# Example usage:
# from modules.model.StableDiffusionModel import StableDiffusionModel
# model = StableDiffusionModel(...)  # Load model with config
# counter = TextTokenCounter(model.tokenizer)
# tokens = counter.count_tokens("Hello world")

if __name__ == "__main__":
    import sys
    # Test with CLIP tokenizer (used in Stable Diffusion)
    try:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        counter = TextTokenCounter(tokenizer)
        text = sys.argv[1] if len(sys.argv) > 1 else "Hello world"
        print(f"Tokens in '{text}': {counter.count_tokens(text)}")
    except Exception as e:
        print(f"Cannot test: {e}")
