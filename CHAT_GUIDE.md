# 🤖 Interactive Chat with Your Model

Add this cell to your Colab notebook to chat with your trained model in real-time!

## Quick Start

```python
# Run this cell after training to chat with your model!
!python chat.py
```

## Features

- **ChatGPT-like interface** - Interactive terminal chat
- **Smart prompt suggestions** - Tailored for TinyStories
- **Auto-loads best checkpoint** - Uses `best.pt` automatically
- **Real-time generation** - See your model's creativity!

## Usage Examples

### Basic Chat
```python
# Use default settings (best checkpoint, 200 tokens)
!python chat.py
```

### Custom Settings
```python
# Longer responses
!python chat.py --max-tokens 300

# More creative (higher temperature)
!python chat.py --temperature 1.0

# More focused (lower temperature)
!python chat.py --temperature 0.5

# Use specific checkpoint
!python chat.py --checkpoint checkpoints/final.pt
```

## Prompt Suggestions

The chat interface includes built-in suggestions perfect for TinyStories:

1. "Once upon a time"
2. "One day, a little"
3. "There was a"
4. "A boy named"
5. "A girl named"
6. "In a small town"
7. "The sun was shining"
8. "It was a beautiful day"
9. "A cat and a dog"
10. "The little bird"

## Interactive Commands

While chatting, you can use these commands:

- `help` - Show prompt suggestions
- `suggestions` - Quick prompt ideas
- `settings` - View current generation settings
- `quit` or `exit` - End chat session

## Example Session

```
You: Once upon a time
🤖 Model: there was a little girl named Lily. She loved to play with her toys...

You: A boy named
🤖 Model: Tom went to the park. He saw a big tree and wanted to climb it...

You: help
💡 Suggested Prompts (TinyStories work best!):
  1. Once upon a time
  2. One day, a little
  ...
```

## Tips for Best Results

1. **Use simple prompts** - The model is trained on children's stories
2. **Start with common phrases** - "Once upon a time", "One day", etc.
3. **Keep it short** - 3-5 words work best
4. **Experiment with temperature** - Lower (0.5-0.7) for coherent, higher (0.8-1.0) for creative
5. **Watch the progress** - Use `best.pt` to see your model improve!

## Advanced: Programmatic Chat

For more control, use the chat interface programmatically:

```python
from chat import load_model, generate_response
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_model("checkpoints/best.pt", device)

# Generate a response
prompt = "Once upon a time"
response = generate_response(
    model, tokenizer, prompt, device,
    max_tokens=200,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)

print(f"Prompt: {prompt}")
print(f"Response: {response}")
```

## Troubleshooting

**No checkpoint found?**
- Make sure you've trained the model first: `!python train.py`
- Check that `checkpoints/` directory exists

**Responses don't make sense?**
- Try training longer (model needs more time)
- Use lower temperature (0.5-0.7) for more coherent text
- Use better prompts from the suggestions

**Out of memory?**
- Reduce `--max-tokens` (try 100 instead of 200)
- This shouldn't happen during inference, but if it does, restart runtime

---

**Happy chatting! 🎉**

See how your model improves as training progresses by chatting with different checkpoints!
