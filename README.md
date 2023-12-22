# Summarise Research Using LLM Chatbot Project for the Final Project of VNUK's Artificial Intelligence Course

To use this. Clone this project into Visual Studio Code or your IDE then unzip the file and cd to the directory
```
cd SummariseResearchPaperUsingLLM
```

Create a virtual environment
```
python -m venv YourVenv
```

Activate virtual environment

For Windows Powershell
```
./YourVenv/bin/activate
```

For Mac Bash
```
source YourVenv/bin/activate
```

Install all the libraries

```
pip install -r requirements.txt
```

Use the program

```
cd /Rag
```

Add your API Key in the **.env** file (provided in a Google Doc file in the Google Drive Folder)



- Use the OpenAi Version(Recommended)
```
python process.py
```

- Use the LLaMa 2 Version(Not recommended because it takes a lot of GPU). Make sure that you downloaded the weight to the /Rag directory
```
python process_llama.py
```

link to download llama model: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF#provided-files

For the weight, download this THIS: <img width="633" alt="image" src="https://github.com/ngoinhaoto/SummariseResearchPaperUsingLLM/assets/68233426/7e850c54-a520-44bc-a15e-412bdc6a8ed9">

The ipynb file is for MAC ARM Chip which just contains the code of us messing around with the model. If you want to use it, you might have to modify the LLM init part accordingly. Please reference to [this](https://python.langchain.com/docs/integrations/llms/llamacpp)




