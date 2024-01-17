<div align="center">

# Deploying LLaVA on vLLM

</div>

# Overview
In this repository, we deploy a LLaVA model using vLLM.

# Demo
[![Demo](https://img.youtube.com/vi/Ewim3fN5vJg/hqdefault.jpg)](https://www.youtube.com/embed/Ewim3fN5vJg)

## Installtion

Since this implementation is based on a [PR](https://github.com/vllm-project/vllm/pull/2153) as well as additional files located here, vLLM needs to be installed from the `vllm` subdirectory.


```bash
cd vllm
pip install .
```




## Inference
To run the model, we can use the llava server. 

```bash
python -m vllm.entrypoints.llava_server  --model llava-hf/llava-1.5-7b-hf --trust-remote-code --gpu-memory-utilization 0.90 --max-model-len 1024


python test_server.py
```

We can also run the (modified) OpenAI server to get a drop-in replacement for the openai API.

```bash
python -m vllm.entrypoints.openai.lapi_servers --model llava-hf/llava-1.5-7b-hf --trust-remote-code --gpu-memory-utilization 0.90 --max-model-len 1024

python test_openai.py
```

we can also run the Streamlit frontend

```bash
streamlit run main.py
```