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
  Time (mean ± σ):      1.395 s ±  0.103 s    [User: 1.394 s, System: 0.001 s]
  Range (min … max):    1.317 s …  1.673 s    10 runs
```

Ocean + -O3 flag

```bash
hyperfine "./matmul" -w 20
Benchmark 1: ./matmul
  Time (mean ± σ):     102.9 ms ±  20.2 ms    [User: 102.2 ms, System: 0.7 ms]
  Range (min … max):    69.3 ms … 142.8 ms    33 runs
```
