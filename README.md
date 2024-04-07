# NeuroLex

## Introduction
This project aims to translate human brainwaves into spoken words using advanced AI and neurotechnology. It captures brainwave data via Neurosity devices, processes the raw signals to generate text embeddings, utilizes a Language Model to interpret meaning and intent, and finally transforms this interpretation into cloned voice speech using generative voice technologies.

## Prerequisites
- Python 3.x
- Neurosity device and access to its APIs
- OpenAI API access for Language Model processing
- PlayHT or a similar service for voice cloning

## Setup Instructions

### Environment Variables
1. Create a `.env` file in the project's root directory.
2. Add your API keys to the .env file. .env.example is provided.

### Context File
- Fill out the `context.txt` with general information of where you are and what you are doing.

### Model Setup
1. Download the necessary CLIP model for embeddings and language interpretation here : [Download](https://drive.google.com/file/d/1wAk4tGsrZB3AXUO1ZRog1XElzytebRCZ/view?usp=sharing)
2. Store the model files in the `model` directory.

### Installation
Install required Python packages:

```
pip install -r requirements.txt
```

## Running

```
python main.py
```

This script initializes the system, captures brainwave data fromn Neurosity device, generates embeddings, processes the data through the Language Model to construct meaningful speech, and outputs the speech in a cloned voice.

## Output
The system will output voice speech that represents the interpreted brainwave signals. The output format and storage can be adjusted as needed.

## Contributing
Best way to contribute is to improve the embedding generation model

## License
MIT


