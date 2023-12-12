link to download llama model: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF#provided-files


DOWNLOAD THIS: <img width="633" alt="image" src="https://github.com/ngoinhaoto/SummariseResearchPaperUsingLLM/assets/68233426/7e850c54-a520-44bc-a15e-412bdc6a8ed9">


the ipynb file is for MAC ARM Chip, using windows, changes the llm init code to

```
n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

```

