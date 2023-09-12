# Project-CC: A moody chatbot

This repository contains code related to the language models used for the chatbot CC.

Current folder structure:

## TODO:
 - Add documentation on Datasets: EmotionsGo etc
 - Add Folder Structure to README
 - seq2seq language modeling head being overwritten 

## Requirements:

### Libraries
List of main libraries needed for this
- datsets==2.40
- torch==1.7.0
- transformers==4.18.0
- adapter-transformers==3.0.1

### Datsets

PersonaAI

Use this command to download the PersonAI dataset:
```wget https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json```

- move the downloaded json file to ./datasets/raw/

### Folder Structure

