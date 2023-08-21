
## Info

This folder contains the code for LaGR-SEQ in cube stacking, image completion and object arrangement (corresponding to extended results shown in the appendix). The python code for each environment is found within the corresponding directories. Each directory contains:

1. Python code for LaGR-SEQ
2. Python code for LaGR without SEQ
3. Python code for DQN
4. Python code (including text descriptor) for querying LLM from server
5. LLM cache file

## Environments
The environments considered are:
1. Cube stacking
2. Image completion
3. Object arrangement (extended versions of the simulated robot experiments)

## Methods
1. LaGR-SEQ
2. LaGR without SEQ
3. DQN (Mnih et al.,"Human-level control through deep reinforcement learning") or Q learning for cube stacking

## Instructions
conda create --name lagrseq python=3.9
conda activate lagrseq
Install dependencies with pip install -r requirements.txt

Users will also need a paid account with a valid key to access OpenAI API
Set openai API key using instructions from: https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety

##Sample call:
python LaGRSEQ.py