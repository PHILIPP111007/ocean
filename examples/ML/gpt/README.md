Pure GPT2:

```bash
mode = inference
model id = gpt2
device = cpu
dtype = torch.float16
GPT2 config = vocab 50257 context 1024 hidden 768 heads 12 ff 3072 layers 12
parameters = 124439808
prompt tokens = 292
generated tokens = 100
elapsed seconds = 4.536279
milliseconds per token = 45.362794
tokens per second = 22.044498
```

Sublinear Attention:

```bash
mode = inference
model id = gpt2
attention = sublinear routed blocks
device = cpu
dtype = torch.float16
GPT2 config = vocab 50257 context 1024 hidden 768 heads 12 ff 3072 layers 12
attention config = local_window 100 block_size 64 top_k_blocks 5 refresh_interval 50 mandatory_recent_blocks 2 exploration_blocks 1
parameters = 124439808
prompt tokens = 292
generated tokens = 100
elapsed seconds = 14.150095
milliseconds per token = 141.500948
tokens per second = 7.067090
```
