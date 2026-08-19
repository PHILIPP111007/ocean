# AGENTS.md — Ocean repository instructions

This file defines the working rules for coding agents operating in the Ocean
repository.

For broad architectural context, read:

```text
docs/Handoff.md
```

before making non-trivial compiler, runtime, ML, GPU, ownership, or web/backend
changes.

The main development branch is:

```text
dev
```

---

## 1. Project identity

Ocean is a Python-like statically compiled language that lowers Ocean source
through a typed compiler pipeline into C11.

The project has two equally important application verticals:

```text
Ocean
├── Backend / systems
│   ├── TCP / HTTP
│   ├── JSON
│   ├── files
│   ├── pthread workers
│   ├── routing
│   └── middleware
│
└── ML / HPC
    ├── Tensor
    ├── autograd
    ├── Transformer
    ├── TinyGPT
    ├── CPU / GPU
    └── OpenMP
```

Do not evolve one vertical by deleting, bypassing, or forgetting the other.

---

## 2. Compiler pipeline

The canonical pipeline is:

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

Key files:

```text
src/parser.py
src/typed_ir.py
src/debug.py

src/codegen/
    orchestrator.py
    expressions.py
    calls.py
    oop.py
    ownership.py
    tensor_codegen.py
    list_codegen.py
    ...

src/modules/
    imports.py
    constants.py

main.py
```

Prefer extending the Typed IR path.

Do not introduce a new ad-hoc legacy AST path unless there is no viable typed
representation and the limitation is explicitly documented.

---

## 3. Repository layout

Important directories:

```text
src/            compiler/frontend/backend
std/            Ocean standard library + C runtimes
tests/          regression and integration tests
examples/       executable Ocean examples
docs/           architecture and handoff documentation
```

ML runtime:

```text
std/tensor/tensor.oc
std/tensor/tensor_runtime.h
std/tensor/tensor_runtime.c
std/tensor/autograd_runtime.h
std/tensor/autograd_runtime.c

std/ml/ml.oc
std/ml/nn.oc
std/ml/optim.oc
```

Backend/server runtime:

```text
std/net/socket.oc
std/net/http.oc
std/net/web.oc
std/net/net_runtime.h
std/net/net_runtime.c
std/net/web_runtime.h
std/net/web_runtime.c
std/json/
std/io/
```

---

## 4. Coding style

Use:

```text
4-space indentation
snake_case for Python functions/variables
PascalCase for Ocean/Python classes
ocean_ prefix for generated/runtime Ocean C symbols
```

Keep compiler logic in the existing focused `src/codegen/` modules.

Do not rebuild a monolithic generator.

Preserve standard C/POSIX names for FFI:

```text
malloc
free
memcpy
sqrt
pthread_create
...
```

---

## 5. Build and tests

Run the full suite:

```bash
python -m pytest
```

or:

```bash
pytest --verbose
```

Package build:

```bash
python main.py build
```

Package workflows:

```bash
python main.py init
python main.py check
python main.py build
python main.py run
python main.py test
python main.py clean
```

Use C11.

For runtime work, prefer strict diagnostics:

```bash
-std=c11
-Wall
-Wextra
-Wpedantic
-Werror
```

For memory bugs:

```bash
-fsanitize=address,undefined
-O0
-g3
-fno-omit-frame-pointer
```

---

## 6. Current verified baseline

As of 2026-08-19, the latest reported full regression result after the GPU
device-semantics work is:

```text
137 passed, 1 skipped
```

The skipped test is:

```text
tests/test_gpu_training_v01_ocean.py
```

It was skipped because OpenCL was not fully usable in the local environment.

Therefore:

> CPU compiler/runtime/ML regressions are green.
>
> The new GPU training path is implemented but must not be called fully verified
> until the GPU integration test actually passes rather than skips.

Do not convert `skipped` into "GPU works".

---

## 7. Memory model

Conceptual memory kinds:

```text
VALUE
    int
    float
    bool
    struct

OWNED
    array[T]

SHARED / ARC
    list[T]
    dict[K, V]
    tuple[T]
    class
    Tensor[T]

BORROWED
    &T
    &mut T

RAW / FFI
    *T
    void*
    C handles
```

ARC uses:

```c
typedef struct ocean_object_header {
    size_t refcount;
    void (*destroy)(void*);
} ocean_object_header;
```

and:

```c
ocean_retain(...)
ocean_release(...)
```

ARC is currently non-atomic.

Do not silently make normal ARC atomic without a deliberate concurrency design.

---

## 8. Container ownership rules

For reference elements:

```text
append reference       -> retain
insert reference       -> retain
replace reference      -> retain(new), release(old)

remove                  -> release
clear                   -> release all
destroy                 -> release all

pop                     -> transfer ownership to caller
```

`del` is explicit early release, not required manual memory management.

Scope cleanup must remain automatic.

---

## 9. Borrowing

Safe borrows:

```ocean
var x: &T = value
var y: &mut T = value
```

Raw C address-of remains separate:

```ocean
var p: *int = &x
```

The current checker is lexical, not full Rust NLL.

Do not claim stronger guarantees than implemented.

---

## 10. Inheritance

Current safe model:

```text
single inheritance
```

Do not re-enable multiple inheritance through unsafe C casts.

A future traits/interfaces design is preferred.

---

## 11. C imports and external symbols

Important implementation detail:

```ocean
cimport <std/tensor/autograd_runtime.h>
```

does not automatically mean the frontend parses every C prototype.

The Validator maintains external C function knowledge in:

```text
src/debug.py
```

When adding a new runtime C function, verify all of:

```text
1. declaration in .h
2. implementation in .c
3. cimport in .oc
4. Validator external symbol registration
5. frontend return type handling if used as an expression
```

This exact issue occurred when adding:

```text
ocean_autograd_adamw_create
ocean_autograd_adamw_begin_step
ocean_autograd_adamw_step
```

---

## 12. Tensor public API

The public tensor type is:

```ocean
Tensor[T]
```

Do not restore a separate lowercase tensor API.

Important operations include:

```text
Tensor.zeros
Tensor.from_list
Tensor.load_npy

save_npy
to
copy

matmul
add/sub/mul/div
scalar arithmetic

reshape
transpose
permute

sum_dim
mean_dim

exp
log
sqrt
pow
softmax
layer_norm
masked_fill

shape
ndim
size
device
item

requires_grad_
backward
grad
zero_grad
```

---

## 13. Tensor identity invariant

Tensor runtime uses a monotonic identity/generation value.

Do not use raw pointer address as Tensor identity.

Allocator pointer reuse previously caused stale autograd metadata bugs.

Autograd lookup must continue to account for Tensor identity/generation.

---

## 14. ND Tensor

ND Tensor supports Transformer-style workloads:

```text
arbitrary-rank shapes
broadcasting
ND reshape
ND slicing
batched/broadcasted matmul
arbitrary-dim transpose
permute
sum_dim
mean_dim
.npy arbitrary rank
autograd
```

Batched matmul contract:

```text
A [..., M, K]
B [..., K, N]
```

Backward:

```text
dA = dY @ B.transpose(-2, -1)
dB = A.transpose(-2, -1) @ dY
```

then normalize/sum gradients to the original metadata shape.

Do not remove the higher-rank `sum_to_meta` fix that left-pads missing leading
dimensions with singleton axes.

---

## 15. Autograd lifetime invariant

This is a critical rule.

`ocean_autograd_meta*` describes graph topology, but:

```text
meta->tensor
```

is not a lifetime guarantee for the underlying Tensor storage.

A real heap-use-after-free was found in LayerNorm backward when an intermediate
Tensor wrapper was released before backward.

If backward needs actual forward Tensor values, the graph node must own them:

```c
node->saved_left = ocean_tensor_copy(tensor);
node->saved_right = ocean_tensor_copy(other);
```

and backward must read the saved handles.

Do not read:

```c
node->left->tensor
node->right->tensor
```

for forward values unless lifetime is proven.

---

## 16. Do not delete autograd metadata on Tensor release

Do not add a generic "release Tensor => delete metadata" hook.

Graph nodes currently hold raw:

```c
ocean_autograd_meta*
```

pointers.

Deleting metadata at wrapper destruction can create dangling graph topology.

Fix needed data lifetime through:

```text
saved_left
saved_right
immutable metadata copied to node
```

not by eager metadata destruction.

---

## 17. ML stack currently working

The current minimal ML framework includes:

```text
Tensor
Parameter
Module

Linear
ReLU
MSELoss

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

## 18. TinyGPT baseline

Current TinyGPT v0.1:

```text
token embedding
+
position embedding
    ↓
TransformerBlock × 2
    ↓
LayerNorm
    ↓
lm_head
```

Verified toy result:

```text
initial loss = 2.079442
final loss   = 0.057157
predicted next token = 7

token embedding grad    = 1
position embedding grad = 1
lm head grad            = 1
```

This is the end-to-end training baseline.

Do not weaken TinyGPT tests merely to make a regression pass.

---

## 19. AdamW invariant

AdamW v0.1 includes:

```text
m
v
bias correction
epsilon
decoupled weight decay
```

Each optimizer instance has its own:

```text
state_id
step
per-Parameter state
```

Two optimizers must not accidentally share moment buffers.

Keep the numerical two-step reference test.

---

## 20. Parameter and Module device semantics

Current API:

```ocean
parameter.to("gpu")
model.to("gpu")
```

`Parameter.to(device)` must:

```text
move/copy Tensor to target device
restore requires_grad=True
preserve Parameter semantics
```

`Module.to(device)` must operate on the concrete model's `parameters()`.

---

## 21. Inherited `Module.to` specialization

The ordinary inherited-method stub is insufficient for `Module.to`.

A parent implementation that statically calls:

```text
Module_parameters()
```

would lose the concrete override such as:

```text
Linear.parameters()
TinyGPT.parameters()
```

Therefore codegen currently specializes inherited `Module.to` to call the
concrete class's `parameters()`.

Do not remove this specialization until the language has correct dynamic
dispatch for this pattern.

---

## 22. GPU backend status

The current GPU backend is OpenCL.

Public API must remain backend-neutral:

```ocean
tensor.to("gpu")
model.to("gpu")
```

User code should not see:

```text
cl_mem
cl_context
cl_command_queue
```

Native OpenCL paths exist for part of the Tensor runtime, including important
float32/int32 operations such as matmul and basic elementwise/scalar ops.

Some operations still use correctness-first CPU fallback:

```text
GPU
 ↓
CPU
 ↓
operation
 ↓
GPU
```

Do not describe such operations as GPU-native.

---

## 23. GPU optimizer semantics

Current GPU-aware SGD/AdamW correctness path can temporarily copy GPU weights
and gradients to CPU, update there, then copy the updated values into the
original GPU Tensor handle.

This is functional architecture, not final performance architecture.

Future work:

```text
GPU-native SGD kernel
GPU-native AdamW kernel
GPU m/v state
```

---

## 24. `ocean_tensor_copy_into`

`ocean_tensor_copy_into(destination, source)` exists to update data inside an
existing Tensor handle without replacing its identity.

This is important for leaf Parameter state.

Do not casually replace optimizer updates with:

```text
parameter.data = parameter.data.to(...)
```

inside every step.

Preserve Tensor identity where autograd/optimizer state expects it.

---

## 25. OpenCL verification

The GPU integration test is:

```bash
python -m pytest tests/test_gpu_training_v01_ocean.py -q -s
```

Useful environment checks:

```bash
pkg-config --exists OpenCL
pkg-config --cflags OpenCL
pkg-config --libs OpenCL
clinfo -l

nvidia-smi
nvcc --version
```

If OpenCL is not available, do not silently run device `"gpu"` as CPU.

Fail explicitly or skip an environment-dependent test.

---

## 26. CUDA

CUDA is a reasonable future GPU backend for NVIDIA/H100 systems.

Do not replace the public API with CUDA-specific calls.

Preferred architecture:

```text
device="gpu"
    ↓
preferred runtime backend
    ├── OpenCL
    └── CUDA
```

Backend selection is an implementation/build concern.

---

## 27. ML tests to run after Tensor/autograd changes

At minimum, if the corresponding files exist:

```bash
python -m pytest tests/test_layernorm_v01_ocean.py -q
python -m pytest tests/test_transformer_block_v01_ocean.py -q
python -m pytest tests/test_tiny_gpt_v01_ocean.py -q
python -m pytest tests/test_tiny_gpt_adamw_v01_ocean.py -q
python -m pytest tests/test_adamw_v01_runtime.py -q
python -m pytest tests/test_adamw_v01_ocean.py -q
python -m pytest
```

After OpenCL is configured:

```bash
python -m pytest tests/test_gpu_training_v01_ocean.py -q -s
```

---

## 28. Numerical testing requirements

New ML primitives should not be accepted based only on compilation.

Prefer:

```text
forward reference
backward reference
finite-difference gradient check
shape tests
broadcast tests
lifetime test
full regression
```

When debugging a crash after backward, use ASan instead of guessing.

---

## 29. Known frontend/compiler quirks

Be aware of these previously observed issues.

### Multiline class method signatures

Some multiline method signatures historically resolved calls but failed to
emit implementations.

For critical ML methods, prefer a one-line signature unless a regression test
proves the multiline path is safe.

### Constructor lowering

Constructor lowering is more restrictive than ordinary method bodies.

Do not assume every expression valid in `forward()` is valid in `__init__()`.

### Chained attribute calls in `print`

A pattern such as:

```ocean
print(projection.weight.has_grad())
```

has previously confused the Validator.

A local alias is safer when needed.

### `Tensor.item()` reassignment

A fresh typed declaration is safer than assigning an `item()` result into an
existing variable if type inference regresses.

### 1D Tensor assignment

Historically:

```ocean
positions[0] = 0
```

could lower incorrectly as C indexing on the Tensor wrapper.

TinyGPT uses a `[1, T]` positions Tensor with:

```ocean
positions[0, i]
```

until the 1D assignment path is explicitly covered by regression.

### bool vs int

Python frontend implementation must check `bool` before generic `int`, because:

```python
isinstance(True, int)
```

is true.

---

## 30. Backend/server development is first-class

Ocean is intended to support native backend servers.

Do not remove or de-prioritize:

```text
std/net
TCP sockets
HTTP runtime
Request/Response
Router
middleware
worker pool
keep-alive
JSON integration
```

The server/backend stack is a first-class project goal.

---

## 31. `std/net` architecture

Relevant files:

```text
std/net/socket.oc
std/net/http.oc
std/net/web.oc
std/net/net_runtime.h
std/net/net_runtime.c
std/net/web_runtime.h
std/net/web_runtime.c
```

Layering:

```text
TCP sockets
    ↓
HTTP client/runtime
    ↓
typed web server API
```

Application code should use Ocean objects, not raw socket/pthread layouts.

---

## 32. Typed web API

Target/public objects:

```text
Request
Response
App
Router
Next
```

Typical handler:

```ocean
def handler(request: Request) -> Response:
    ...
```

Request API includes concepts such as:

```text
method
path
query
body
json
remote
header
query_param
path_param
```

Response API includes:

```text
text
json
json_value
html
empty
redirect
add_header
```

Use `std/json` rather than manual JSON string concatenation.

---

## 33. Web ABI naming

Ocean classes lower with the `ocean_` prefix.

For example:

```text
Request  -> ocean_Request
Response -> ocean_Response
```

Do not declare callback ABI against imaginary:

```c
struct Request
struct Response
```

Use generated Ocean class names consistently.

---

## 34. Raw C call attribute workaround

Historically, raw C calls with direct attribute expressions such as:

```ocean
@ocean_web_get(self.handle, ...)
```

could emit incorrect C member syntax.

The robust stdlib pattern is:

```text
Ocean wrapper method
    ↓
local C-typed handle
    ↓
raw @C call
```

Prefer extracting `self.handle` through a normal Ocean method/local variable
before crossing the raw C boundary until generic lowering is fixed.

---

## 35. HTTP worker pool

The preferred server architecture is bounded fixed worker threads:

```text
accept thread
    ↓
bounded connection queue
    ↓
worker pool
```

Prefer this over unbounded thread-per-request.

If web runtime uses pthreads, build integration must include:

```text
-pthread
```

---

## 36. Keep-alive

The backend direction includes HTTP/1.1 sequential keep-alive.

Conceptual controls:

```text
keep_alive timeout
max_keep_alive_requests
```

Correct handling must include:

```text
Connection
Content-Length
Keep-Alive
```

HTTP/2 and HTTP/3 are not current prerequisites.

---

## 37. Middleware

Target middleware shape:

```ocean
def middleware(
    request: Request,
    call_next: Next
) -> Response:
    ...
```

The chain must preserve before/after ordering around route handlers.

Use middleware for:

```text
CORS
logging
request-id
auth
timing
recovery
rate limiting
```

---

## 38. Router

Target API includes:

```text
route
get
post
put
patch
delete
options
head
any
include
```

Router must depend on public web runtime ABI, not private `App` struct fields.

Do not couple Router to internal route-table layout.

Nested routers are part of the intended direction.

---

## 39. Web thread safety

ARC is non-atomic.

Request-local managed objects should remain thread-confined to a worker:

```text
Request
Response
Json
str
list
temporary classes
```

Do not assume shared mutable Ocean objects are thread-safe.

Future shared application state needs:

```text
Shared[T]
atomic ARC
Send/Sync-like rules
locks
thread-safe containers
```

---

## 40. `std/net` regression expectations

Important server tests include:

```text
GET / -> 200
404
405
HEAD fallback

path params
query params

POST JSON
JSON response

custom headers

middleware ordering

multiple concurrent connections

keep-alive:
    sequential requests
    Connection: close
    request limit
    idle timeout

worker queue saturation

Router prefix
Router path params
GET/POST/PUT/PATCH/DELETE
```

Compile web runtime strictly:

```bash
gcc \
    -std=c11 \
    -pthread \
    -Wall \
    -Wextra \
    -Wpedantic \
    -Werror \
    -I. \
    -c std/net/web_runtime.c
```

Use ASan/UBSan for server smoke tests.

---

## 41. Do not do these things

Do not:

- convert Tensor back into `list[list[T]]`;
- delete autograd metadata whenever a Tensor wrapper is released;
- use raw pointer address as Tensor identity;
- disable bounds checks through `NDEBUG`;
- add silent CPU fallback for selecting device `"gpu"`;
- call a skipped GPU test a successful GPU validation;
- claim CPU-fallback Transformer ops are GPU-native;
- remove the special concrete `Module.to` path without replacing its semantics;
- re-enable unsafe multiple inheritance;
- add handwritten generic SIMD that assumes one element size/layout;
- couple Router to private web runtime structs;
- expose raw OpenCL/socket/pthread handles in ordinary user code;
- weaken numerical/gradient tests to make regressions pass.

---

## 42. Current priorities

Near-term priorities:

```text
P0
    make OpenCL environment usable
    get GPU integration test to actually pass

P1
    GPU-native Transformer hot paths
    softmax
    LayerNorm
    batched matmul
    Embedding
    reductions
    optimizer kernels

P2
    autoregressive TinyGPT.generate()

P3
    KV cache

P4
    Tensor.arange
    automatic position ids
    RoPE

P5
    optional CUDA backend behind the same "gpu" API

P6
    continue std/net:
        Router stabilization
        middleware
        keep-alive
        graceful shutdown
        CORS
        cookies/uploads
        WebSocket/OpenAPI later
```

Backend/server and ML/HPC are both first-class.

---

## 43. Before submitting a change

For compiler/runtime changes:

```text
1. identify the affected invariant
2. add/update a focused regression
3. compile generated C
4. run the binary
5. use strict warnings
6. use ASan/UBSan for lifetime-sensitive code
7. run the full pytest suite
8. update docs/Handoff.md when architecture/status changed
```

For ML changes:

```text
numerical correctness
gradient correctness
finite differences where practical
full regression
```

For GPU changes:

```text
verify the test actually ran on GPU
do not treat skip as pass
compare CPU/GPU outputs
```

For web/backend changes:

```text
strict C compile
socket-level integration test
concurrency test when relevant
ASan/UBSan
```

---

## 44. Commit and PR guidance

Use concise imperative commit summaries.

A PR should state:

```text
what behavior changed
which architecture/invariant was affected
which tests were run
whether generated C changed
whether ownership/lifetime changed
whether GPU or server runtime behavior changed
```

Do not claim tests were run if they were not run.

---

## 45. Final rule

When uncertain about current architectural intent, read:

```text
docs/Handoff.md
```

and preserve these core invariants:

```text
Typed IR as compiler contract
deterministic ownership
safe autograd lifetime
stable Tensor identity
backend-neutral device API
real numerical tests
backend/server as a first-class use case
ML/HPC as a first-class use case
```
