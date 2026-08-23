# Ocean 🌊

> **Python-like syntax. Native C11. ML, backend servers, systems code — in one language.**

<img src="images/ocean.jpg" alt="Ocean project illustration" width="100%" height="500px" />

Ocean is an experimental compiled language that keeps Python-like syntax while
lowering programs to native C11.

The project is designed around two first-class directions:

```text
Ocean
├── Backend / systems
│   ├── HTTP servers
│   ├── TCP sockets
│   ├── JSON
│   ├── files
│   ├── routing / middleware
│   └── native C/POSIX interop
│
└── ML / HPC
    ├── Tensor
    ├── autograd
    ├── Transformer
    ├── TinyGPT
    ├── CPU / GPU
    └── OpenMP
```

The idea is simple:

> write code that feels close to Python, but keep native compilation,
> explicit ownership, predictable runtime behavior, and direct access to C.

---

## ✨ Why Ocean?

Ocean aims to combine:

- **Python-like syntax**
- **native C11 output**
- **static validation**
- **automatic ownership management**
- **ARC + borrowing**
- **C/POSIX FFI**
- **Tensor + autograd**
- **Transformer / GPT building blocks**
- **CPU + GPU device API**
- **OpenMP**
- **HTTP/backend runtime**

A small Ocean program still looks familiar:

```ocean
def square(x: float32) -> float32:
    return x * x


def main() -> int:
    var value: float32 = square(4.0)

    print(value)

    return 0
```

But the compiler pipeline is:

```text
Ocean source
    ↓
Parser
    ↓
Typed IR
    ↓
Validator
    ↓
CCodeGenerator
    ↓
C11
    ↓
native executable
```

---

# 🧠 ML in Ocean

Ocean already has a small eager ML stack:

```text
Tensor
Parameter
Module

Linear
ReLU
LayerNorm

MultiHeadAttention
TransformerBlock

Embedding
CrossEntropyLoss

SGD
AdamW

TinyGPT
```

---

## Train a neural network

A minimal training loop looks intentionally familiar:

```python
import <std/tensor/tensor.oc>
import <std/ml/nn.oc>
import <std/ml/optim.oc>


def main() -> int:
    var x: Tensor[float32] = Tensor.from_list(
        [[0.0], [1.0], [2.0], [3.0]],
        "cpu"
    )

    var y: Tensor[float32] = Tensor.from_list(
        [[1.0], [3.0], [5.0], [7.0]],
        "cpu"
    )

    var model: Linear = Linear(1, 1)

    var optimizer: AdamW = AdamW(
        model.parameters(),
        0.05,
        0.9,
        0.999,
        0.00000001,
        0.01
    )

    var loss_fn: MSELoss = MSELoss()

    var step: int = 0

    while step < 200:
        optimizer.zero_grad()

        var prediction: Tensor[float32] = model.forward(x)
        var loss: Tensor[float32] = loss_fn.forward(prediction, y)

        loss.backward()
        optimizer.step()

        step = step + 1

    return 0
```

Under the hood this goes through Ocean's own Tensor runtime and dynamic autograd
engine.

No Python runtime is required by the generated binary.

---

## TinyGPT

Ocean can already train a small decoder-only Transformer end-to-end.

The current TinyGPT stack is:

```text
token embedding
+
position embedding
    ↓
TransformerBlock
    ↓
TransformerBlock
    ↓
LayerNorm
    ↓
lm_head
    ↓
CrossEntropyLoss
```

A real verified toy training run produced:

```text
initial loss = 2.079442
final loss   = 0.057157

predicted next token = 7

token embedding grad    = 1
position embedding grad = 1
lm head grad            = 1
```

That means the full path works:

```text
Embedding
→ attention
→ residuals
→ LayerNorm
→ FFN
→ logits
→ CrossEntropy
→ backward
→ optimizer
```

## GPT-2 inference benchmark

`examples/ML/gpt2_native_ternary_inference.oc` measures generation of new
tokens for the canonical GPT-2 small profile `(50257, 1024, 768, 12, 3072,
12)`. The benchmark runs only `eval()`/inference, disables autograd graph
construction, performs one warmup token, and prints elapsed seconds,
milliseconds per token, and tokens per second:

```bash
ocean run examples/ML/gpt2_native_ternary_inference.oc \
    --cflags "-I${CONDA_PREFIX}/include -L/usr/local/cuda/targets/x86_64-linux/lib -lOpenCL -DOCEAN_TENSOR_ENABLE_OPENCL"
```

The benchmark now uses a per-layer KV-cache: the prompt is prefetched once and
each generated token computes only the new query/key/value row. The cache uses
GPU-resident `[B, H, max_seq, head_dim]` tensors and native OpenCL kernels for
cache-row writes and prefix reads.

---

## Build a Transformer block

```python
var block: TransformerBlock = TransformerBlock(
    128,   # d_model
    8,     # heads
    512    # feed-forward size
)

var hidden: Tensor[float32] = block.forward(
    input,
    causal_mask
)
```

The attention path supports Transformer-shaped tensors such as:

```text
[B, H, T, D]
```

with batched matmul, transpose, permute, broadcasting, softmax, masking and
autograd.

---

# ⚡ GPU

Ocean exposes a backend-neutral device API:

```python
var model: TinyGPT = TinyGPT(...)

model.to("gpu")

var tokens_gpu: Tensor[int64] = tokens.to("gpu")
var mask_gpu: Tensor[float32] = mask.to("gpu")

var logits: Tensor[float32] = model.forward(
    tokens_gpu,
    positions_gpu,
    mask_gpu
)
```

And training keeps the same API:

```python
model.to("gpu")

var optimizer: AdamW = AdamW(
    model.parameters(),
    0.001,
    0.9,
    0.999,
    0.00000001,
    0.01
)

loss.backward()
optimizer.step()
```

The current GPU backend is based on OpenCL.

The public API intentionally stays:

```text
"cpu"
"gpu"
```

instead of exposing OpenCL/CUDA-specific objects to application code.

Some Tensor operations are already GPU-native, while others still use
correctness-first CPU fallback internally. A CUDA backend can later live behind
the same `"gpu"` API.

OpenCL:

```bash
ocean build examples/ML/gpt2_native_ternary_inference.oc \
    --cflag=-lOpenCL \
    --cflag=-I"/usr/include/CL/" \
    --cflag=-L"/usr/lib/x86_64-linux-gnu/libOpenCL.so" \
    --cflag=-O3
```

CUDA:

```bash
ocean build examples/ML/gpt2_native_ternary_inference.oc \
    --compiler nvcc \
    --cflag=-O3 \
    --cflag=-DOCEAN_TENSOR_ENABLE_CUDA
```

---

# 🌐 Backend development in Ocean

Ocean is not only an ML language.

The standard library is also developing a native backend stack:

```text
std/net
├── TCP sockets
├── HTTP
├── Request / Response
├── App
├── Router
├── middleware
├── worker pool
└── keep-alive
```

The goal is to write backend services with Python-like ergonomics and compile
them into native executables.

---

## A small API server

The intended API looks like this:

```python
import <std/net/web.oc>
import <std/json/json.oc>


def health(request: Request) -> Response:
    var body: Json = Json.object()
    var status: Json = Json.str("ok")

    body.set("status", status)

    return Response.json_value(body)


def hello(request: Request) -> Response:
    var name: str = request.query_param(
        "name",
        "Ocean"
    )

    return Response.text(
        "Hello, " + name + "!"
    )


def main() -> int:
    var app: App = App.create()

    app.get("/health", health)
    app.get("/hello", hello)

    app.workers(8)
    app.queue_size(256)
    app.keep_alive(5000)

    app.run("0.0.0.0", 8080)

    return 0
```

The runtime is built on C11/POSIX primitives rather than a Python event loop.

---

## Routers

Large applications can be organized by route groups:

```python
var app: App = App.create()

var api: Router = Router.create("/api/v1")

api.get("/users/{id}", get_user)
api.post("/users", create_user)
api.patch("/users/{id}", update_user)
api.delete("/users/{id}", delete_user)

app.include(api)
```

Which produces routes like:

```text
GET     /api/v1/users/{id}
POST    /api/v1/users
PATCH   /api/v1/users/{id}
DELETE  /api/v1/users/{id}
```

---

## Middleware

The intended middleware API is deliberately familiar:

```python
def request_log(
    request: Request,
    call_next: Next
) -> Response:

    print("before request")

    var response: Response = call_next.call(request)

    print("after request")

    return response


app.middleware(request_log)
```

The same mechanism can support:

```text
CORS
authentication
request IDs
access logging
timing
rate limiting
error recovery
```

---

## Native worker pool

The backend runtime is designed around a bounded worker pool:

```text
accept thread
    ↓
bounded connection queue
    ↓
worker #1
worker #2
...
worker #N
```

This gives predictable concurrency and memory usage without spawning an
unbounded thread per request.

---

# 🧬 One language, two worlds

A particularly interesting Ocean use case is combining backend and ML in the
same native program.

For example:

```text
HTTP request
    ↓
JSON parsing
    ↓
Tensor preprocessing
    ↓
TinyGPT / model inference
    ↓
JSON response
```

Conceptually:

```python
def predict(request: Request) -> Response:
    var payload: Json = request.json()

    var tokens: Tensor[int64] = encode(payload)
    var tokens_gpu: Tensor[int64] = tokens.to("gpu")

    var logits: Tensor[float32] = model.forward(
        tokens_gpu,
        positions,
        causal_mask
    )

    var result: Json = decode_logits(logits)

    return Response.json_value(result)
```

That is one of the core directions of Ocean:

> **native backend + native ML without crossing a Python/C boundary.**

---

# 🧵 Parallel numerical code

Ocean also supports an ownership-safe subset of OpenMP:

```python
#pragma omp parallel for collapse(2) schedule(static)
for i in range(rows):
    for j in range(columns):
        output[i, j] = left[i, j] + right[i, j]
```

The compiler automatically adds OpenMP compiler flags when required.

---

# 🧠 Ownership without writing Rust

Ocean separates memory behavior by type:

| Type | Runtime model |
|---|---|
| `int`, `float`, `bool` | plain values |
| `array[T]` | unique-owned buffer |
| `list[T]`, `dict[K,V]`, classes | ARC |
| `Tensor[T]` | managed Tensor object |
| `&T` | immutable borrow |
| `&mut T` | exclusive mutable borrow |
| raw pointers / C calls | explicit unsafe boundary |

A mutable borrow looks like:

```ocean
def scale(
    values: &mut array[float32],
    factor: float32
) -> None:

    for i in range(len(values)):
        values[i] = values[i] * factor

    return None
```

Borrowing does not imply a heap allocation or a reference-count increment.

---

# 🔌 C / POSIX interop

Ocean can call native C APIs through an explicit unsafe boundary:

```python
cimport <math.h>


def main() -> int:
    unsafe:
        var value: float64 = @sqrt(16.0)

    print(value)

    return 0
```

This keeps low-level integration possible without exposing raw C semantics
through ordinary safe code.

---

# 📦 NumPy weights

Ocean Tensor can load standard NumPy `.npy` files:

```python
var weights: Tensor[float32] = Tensor.load_npy(
    "weights.npy",
    "cpu"
)

weights.save_npy(
    "weights_copy.npy"
)
```

This makes it possible to move model weights between Python tooling and Ocean
without inventing a custom binary format.

---

# 🚀 Quick start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

Build an Ocean program:

```bash
python main.py build
```

Or, after editable installation:

```bash
ocean --help
```

---

## Package layout

A minimal project can use:

```toml
[package]
name = "my_app"
version = "0.1.0"
entry = "src/main.oc"
source = "src"
build = "build"

[build]
compiler = "gcc"
cflags = ["-std=c11"]

[build.profiles.release]
cflags = ["-O3"]
```

Then:

```bash
ocean check
ocean build
ocean run
```

---

# 🧪 Current status

Ocean is an active prototype.

The current verified CPU/ML regression baseline is:

```text
138 passed
```

There is also a GPU integration test. With the micromamba `base` OpenCL setup,
both GPU integration tests run against the local NVIDIA OpenCL runtime.

Important distinction:

```text
GPU test passed  -> GPU execution verified
GPU test skipped -> environment not verified
```

For the micromamba base environment:

```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate base
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
python -m pytest tests/test_gpu_training_v01_ocean.py tests/test_gpu_hotpaths_v01_runtime.py -q
```

Current working milestones include:

```text
Typed IR
ownership / ARC
borrowing
Tensor
ND broadcasting
batched matmul
autograd
LayerNorm
MultiHeadAttention
TransformerBlock
Embedding
CrossEntropyLoss
SGD
AdamW
TinyGPT training
OpenMP
File IO
NumPy .npy
std/net backend foundation
```

---

# 🗺️ Roadmap

Near-term priorities:

### ML / GPU

```text
GPU-native Transformer kernels
TinyGPT.generate()
KV cache
RoPE
Tensor.arange
GPU-native AdamW
CUDA backend
mixed precision
quantization
```

### Backend

```text
Router stabilization
nested routers
middleware
keep-alive
graceful shutdown
CORS
cookies
uploads
streaming
WebSocket
OpenAPI
database ecosystem
```

### Language / compiler

```text
stronger borrow/data-flow analysis
better diagnostics
traits/interfaces
Result / Option
zero-copy Tensor views
memory pools
LSP / tooling
```

---

# 📚 Documentation

For architecture and implementation details:

- [`docs/Handoff.md`](docs/Handoff.md) — current technical state and roadmap
- [`AGENTS.md`](AGENTS.md) — repository rules for coding agents
- [`docs/MEMORY_MODEL.md`](docs/MEMORY_MODEL.md) — ownership and memory model
- [`std/tensor/README.md`](std/tensor/README.md) — Tensor API
- [`std/tensor/OpenCL.md`](std/tensor/OpenCL.md) — GPU backend design
- [`std/net/README.md`](std/net/README.md) — networking/backend stack

---

# 🌊 Philosophy

Ocean is exploring a space between:

```text
Python
    simplicity

C / C++
    native runtime control

Rust
    ownership discipline

PyTorch
    ML ergonomics

FastAPI-like frameworks
    backend ergonomics
```

The goal is not to clone any one of them.

The goal is to make code like this feel natural:

```python
model.to("gpu")

app.post("/predict", predict)

app.run("0.0.0.0", 8080)
```

while still ending up with a native executable.

---

## License

See the repository license for details.
