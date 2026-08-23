# Benchmarks

## Backend

```python
app = FastAPI()

@app.get("/get_json")
async def get_json(request: Request):
    data = {
        "name": "Ocean",
        "version": 1,
        "enabled": True,
        "values": [10, 20]
    }
    return JSONResponse(content=data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
```

FastAPI:

```bash
hyperfine "curl http://127.0.0.1:8080/get_json" --min-runs 100
Benchmark 1: curl http://127.0.0.1:8080/get_json
  Time (mean ± σ):      10.0 ms ±   1.0 ms    [User: 4.0 ms, System: 4.9 ms]
  Range (min … max):     7.6 ms …  12.6 ms    230 runs
```

Ocean:

```bash
hyperfine "curl http://127.0.0.1:8080/get_json" --min-runs 100
Benchmark 1: curl http://127.0.0.1:8080/get_json
  Time (mean ± σ):       8.3 ms ±   1.0 ms    [User: 3.8 ms, System: 4.3 ms]
  Range (min … max):     6.2 ms …  11.4 ms    283 runs
```

## Matmul

Numpy:

```bash
hyperfine "python matmul.py" -w 20
Benchmark 1: python matmul.py
  Time (mean ± σ):     113.7 ms ±  20.9 ms    [User: 2096.9 ms, System: 15.6 ms]
  Range (min … max):    87.7 ms … 157.7 ms    29 runs
```

Ocean:

```bash
hyperfine "./matmul" -w 5
Benchmark 1: ./matmul
  Time (mean ± σ):     965.6 ms ±  84.7 ms    [User: 964.5 ms, System: 0.8 ms]
  Range (min … max):   864.7 ms … 1058.2 ms    10 runs
```

Ocean + -O3 flag

```bash
hyperfine "./matmul" -w 5
Benchmark 1: ./matmul
  Time (mean ± σ):      77.4 ms ±   4.1 ms    [User: 76.4 ms, System: 0.9 ms]
  Range (min … max):    71.9 ms …  86.1 ms    39 runs
```

## LLM GPT2

```bash
ocean build examples/ML/gpt2_native_ternary_inference.oc \
  --cflag=-lOpenCL \
  --cflag=-I"/usr/include/CL/" \
  --cflag=-L"/usr/lib/x86_64-linux-gnu/libOpenCL.so" \
  --cflag=-O3
```

```bash
(.venv) phil@phil-TUF-Gaming-F16-FX608JMI:~/GitHub/ocean$ ./examples/ML/gpt2_native_ternary_inference 
mode = inference
model device = gpu
backend device = NVIDIA GeForce RTX 5060 Laptop GPU
GPT2 config = vocab 50257, context 1024, hidden 768, heads 12, ff 3072, layers 12
prompt tokens = 16
generated tokens = 100
elapsed seconds = 0.845570
milliseconds per token = 8.455695
tokens per second = 118.263486
first generated token = 0.000000
[ok] Ocean GPT2 inference benchmark
```
