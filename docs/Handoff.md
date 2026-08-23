# Handoff — Phils Language / Ocean

> Обновлено: **2026-08-23**
>
> Этот документ — актуальный технический handoff по текущему состоянию Ocean.
> Он намеренно отделяет **что уже доказано тестами** от **что реализовано, но ещё
> не подтверждено в целевом окружении**, и от **что остаётся roadmap**.
>
> Основная ветка разработки: `main`.

---

# 1. Что такое Ocean

Ocean — компилируемый язык с Python-подобным синтаксисом и C11 backend.

Цели проекта:

- Python-like syntax;
- компиляция в C11;
- предсказуемая ownership-модель;
- ARC для shared/reference объектов;
- borrowing через `&T` / `&mut T`;
- хороший C ABI / FFI;
- HPC/OpenMP;
- Tensor runtime;
- autograd;
- ML/Transformer/GPT;
- CPU и GPU backends.

Целевая ниша:

```text
Python-like syntax
+ static semantics
+ ownership / borrows
+ direct C ABI
+ C11 portability
+ Tensor / ML runtime
```

Ocean не должен превращаться в динамический Python runtime.

---

# 2. Краткий статус проекта

На 2026-08-19 подтверждены:

```text
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

Работают:

- базовые типы;
- list/dict/tuple/class;
- ARC;
- lexical borrow checking;
- single inheritance;
- C imports / FFI;
- OpenMP;
- File/BinaryFile;
- TCP sockets / HTTP client foundation;
- HTTP/Web backend foundation (`std/net`);
- typed `Request` / `Response` / `App` wrappers;
- middleware / Router / worker-pool architecture;
- NumPy `.npy`;
- Tensor CPU backend;
- OpenCL Tensor backend foundation;
- ND Tensor;
- broadcasting;
- batched matmul;
- reshape/transpose/permute;
- reductions;
- softmax;
- LayerNorm;
- dynamic autograd;
- Parameter;
- Module;
- Linear;
- ReLU;
- MSELoss;
- SGD;
- AdamW;
- Embedding;
- CrossEntropyLoss;
- GPT-2-style ternary decoder model;
- GPT-2 tanh-approximation GELU with autograd;
- MultiHeadAttention;
- TransformerBlock;
- TinyGPT;
- next-token training end-to-end.

Последняя общая проверка после GPT-2/GELU изменений:

```text
142 passed, 2 skipped, 1 failed
```

Единственный failure — `tests/test_net_std.py::test_std_net_http`: текущий
sandbox запрещает серверному smoke-тесту `bind/listen` и возвращает
`Operation not permitted`. Это ограничение окружения, а не регрессия Tensor,
autograd или ML.

Пропущен именно GPU integration test, потому что в текущем окружении
OpenCL development/runtime environment пока не полностью доступен.

Важно:

> CPU/ML/autograd regressions зелёные.
>
> Реальное выполнение нового `model.to("gpu")` на H100 пока НЕ считается
> подтверждённым, пока `tests/test_gpu_training_v01_ocean.py` не перестанет
> быть `skipped` и не завершится `passed`.

---

# 3. Главная архитектура compiler pipeline

Основной путь:

```text
Ocean source
    ↓
Parser
    ↓
TypedModule / Typed IR
    ↓
Validator
    ↓
CCodeGenerator
    ↓
generated C11
    ↓
gcc/clang
    ↓
binary
```

Ключевые файлы:

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

Основное правило:

> Новые backend-фичи должны по возможности идти через Typed IR, а не создавать
> отдельный legacy AST path.

---

# 4. Codegen architecture

Старый `CCodeGenerator` был монолитным. Сейчас backend разбит на mixin/modules:

```text
src/codegen/
├── core.py
├── scope.py
├── types.py
├── orchestrator.py
├── statements.py
├── calls.py
├── indexing.py
├── io.py
├── expressions.py
├── list_codegen.py
├── tuple_codegen.py
├── dict_codegen.py
├── helpers.py
├── imports.py
├── oop.py
├── ownership.py
└── tensor_codegen.py
```

Не следует возвращать giant-switch / giant-class архитектуру.

---

# 5. Namespace generated C

Внутренний runtime использует префикс:

```text
ocean_
```

Например:

```c
ocean_object_header
ocean_retain
ocean_release

ocean_tensor_handle_t
ocean_tensor_matmul
ocean_autograd_backward
```

Стандартный C/POSIX ABI не переименовывается:

```c
malloc
free
memcpy
sqrt
pthread_create
```

---

# 6. Memory model

Текущая концептуальная модель:

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

## 6.1 ARC

Reference-типы используют:

```c
typedef struct ocean_object_header {
    size_t refcount;
    void (*destroy)(void*);
} ocean_object_header;
```

И:

```c
ocean_retain(obj);
ocean_release(obj);
```

ARC сейчас non-atomic.

Не следует делать обычный refcount atomic без отдельного concurrency design.

## 6.2 Container ownership

Для reference elements:

```text
append reference       -> retain
insert reference       -> retain
replace reference      -> retain(new), release(old)

remove                  -> release
clear                   -> release all
destroy                 -> release all

pop                     -> transfer ownership to caller
```

## 6.3 `del`

`del` — explicit early release, а не обязательная manual memory management.

Scope cleanup должен происходить автоматически.

---

# 7. Borrowing

Safe borrowing:

```ocean
var x: &Tensor[float32] = tensor
var y: &mut Tensor[float32] = tensor
```

RAW pointer semantics отдельно:

```ocean
var p: *int = &x
```

Это не одно и то же.

Текущий borrow checker lexical, не Rust-style NLL.

Не следует обещать interprocedural/NLL guarantees, которых пока нет.

---

# 8. Classes и inheritance

Классы — ARC objects.

Поддерживаемая модель:

```text
single inheritance
```

Multiple inheritance пока должна считаться запрещённой/неподдерживаемой,
потому что корректный multi-base ABI ещё не реализован.

Долгосрочный вариант:

```text
traits / interfaces
```

вместо небезопасного C-cast based MI.

---

# 9. C imports и Validator

Очень важная текущая особенность:

```ocean
cimport <std/tensor/autograd_runtime.h>
```

**не означает**, что frontend автоматически парсит C header и получает signatures.

Validator держит whitelist известных external C functions в:

```text
src/debug.py
```

Поэтому при добавлении новой runtime C API функции нужно проверить одновременно:

1. declaration в `.h`;
2. implementation в `.c`;
3. `cimport` в `.oc`;
4. регистрацию symbol name в Validator;
5. корректный return type в frontend, если результат используется как expression.

Это уже проявилось при добавлении AdamW:

```text
ocean_autograd_adamw_create
ocean_autograd_adamw_begin_step
ocean_autograd_adamw_step
```

Header был корректным, но Validator сначала считал функции необъявленными.

---

# 10. Tensor runtime

Основные файлы:

```text
std/tensor/tensor.oc
std/tensor/tensor_runtime.h
std/tensor/tensor_runtime.c

std/tensor/autograd_runtime.h
std/tensor/autograd_runtime.c
```

Публичный тип:

```ocean
Tensor[T]
```

Старый отдельный lowercase tensor path не должен возвращаться.

---

# 11. Tensor storage

Runtime handle хранит:

```text
identity
dtype
device
shape
strides
ndim
size
CPU/GPU storage
```

Используется monotonic Tensor identity/generation.

Это важно для autograd metadata.

Нельзя снова сопоставлять graph metadata только по raw pointer address:
allocator может переиспользовать адрес освобождённого Tensor.

---

# 12. Tensor API

Основной публичный API включает:

```text
Tensor.zeros(...)
Tensor.from_list(...)
Tensor.load_npy(...)

tensor.save_npy(...)

tensor.to(device)
tensor.copy()

tensor.ternary_quantize()

tensor.matmul(...)
tensor.add(...)
tensor.sub(...)
tensor.mul(...)
tensor.div(...)

tensor.add_scalar(...)
tensor.sub_scalar(...)
tensor.mul_scalar(...)
tensor.div_scalar(...)

tensor.reshape(...)
tensor.transpose(...)
tensor.permute(...)

tensor.sum_dim(...)
tensor.mean_dim(...)

tensor.exp()
tensor.log()
tensor.sqrt()
tensor.pow(...)

tensor.softmax(...)
tensor.layer_norm(...)

tensor.masked_fill(...)

tensor.shape(...)
tensor.ndim()
tensor.size()
tensor.device()

tensor.item()

tensor.requires_grad_(...)
tensor.backward()
tensor.grad()
tensor.zero_grad()
```

---

# 13. ND Tensor milestone

Поддерживается arbitrary-rank Tensor path для Transformer workloads.

Реализованы:

```text
zeros_nd
from_cpu_strided
reshape ND
typed ND get/set
ndim/shape/size
broadcasting
slice arbitrary axis
.npy arbitrary rank
batched/broadcasted matmul
transpose arbitrary dims
permute
sum_dim
mean_dim
autograd
```

Batched matmul contract:

```text
A [..., M, K]
B [..., K, N]

leading dimensions broadcast
```

Backward:

```text
dA = dY @ B.transpose(-2, -1)
dB = A.transpose(-2, -1) @ dY
```

после чего gradient приводится к исходной shape через sum-to-meta.

---

# 14. Известный bug: higher-rank gradient reduction

При TinyGPT был найден случай, где gradient имел форму:

```text
[T, C]
```

а target meta:

```text
[1, T, C]
```

Старый `sum_to_meta` падал:

```text
autograd cannot reduce gradient to a higher rank
```

Исправление:

- недостающие leading dimensions разрешается left-pad'ить единицами;
- затем выполняется обычный reduction/broadcast normalization.

Этот behaviour не следует откатывать.

---

# 15. Autograd design

Autograd динамический, eager-style.

Tensor metadata хранит:

```text
requires_grad
leaf
grad
shape metadata
graph node
tensor identity
```

Graph node хранит:

```text
left/right parent metadata
saved_left
saved_right
operation data
```

---

# 16. Критический lifetime invariant autograd

Очень важное правило:

> `ocean_autograd_meta*` — topology metadata, но `meta->tensor` НЕ является
> гарантией lifetime Tensor storage.

При TinyGPT был найден реальный:

```text
heap-use-after-free
```

в LayerNorm backward.

Причина:

- forward intermediate Tensor wrapper уничтожался;
- runtime Tensor handle освобождался;
- graph metadata оставался;
- backward читал `node->left->tensor`;
- получался UAF.

Правильный pattern:

```c
node->saved_left = ocean_tensor_copy(tensor);
```

если backward нужны реальные forward values.

И затем backward читает:

```c
node->saved_left
```

а node destructor освобождает saved Tensor.

НЕПРАВИЛЬНО:

```c
node->left->tensor
```

если backward читает данные Tensor.

---

# 17. Не удалять autograd metadata на Tensor release

Ранее рассматривалась идея удалять autograd metadata при release Tensor.

Этого делать нельзя в текущей архитектуре.

Graph nodes содержат raw pointers:

```c
ocean_autograd_meta*
```

и удаление metadata при wrapper release создаёт dangling graph topology.

Правило:

> Исправлять lifetime нужных Tensor values через `saved_left/saved_right`,
> а не через агрессивное удаление metadata.

---

# 18. Math/autograd v0.3

Реализованы:

```text
exp
log
sqrt
pow
softmax(dim)
layer_norm
```

с backward.

Softmax numerically stable.

LayerNorm primitive используется в affine `LayerNorm` module.

---

# 19. Causal attention

Causal masking сейчас строится через composition:

```text
keep = 1 - mask
result = input * keep + mask * value
```

API:

```ocean
tensor.masked_fill(mask, value)
```

Для attention mask используется большое отрицательное значение:

```text
-1e9
```

Проверялось:

```text
future attention weights ≈ 0
row sums ≈ 1
Q/K/V backward
finite differences
```

---

# 20. `permute`

Реализован:

```ocean
tensor.permute([0, 2, 1, 3])
```

Autograd хранит permutation и в backward использует inverse permutation.

Не создавать отдельный opcode на каждую permutation.

---

# 21. ML standard library

Основные файлы:

```text
std/ml/ml.oc
std/ml/nn.oc
std/ml/optim.oc
```

Текущий минимальный framework:

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

GPT2Config
GPT2Embedding
TernaryLinear
GPT2Attention
GPT2MLP
GPT2Block
GPT2Ternary

SGD
AdamW

TinyGPT
```

---

# 22. Parameter

Текущая логика:

```ocean
class Parameter:
    data: Tensor[float32]
```

`Parameter` включает:

```text
tensor()
grad()
has_grad()
zero_grad()
step()          # SGD
adamw_step(...)
to(device)
```

После `Parameter.to(device)` новый Tensor снова получает:

```text
requires_grad = True
leaf semantics
```

---

# 23. Module

Базовый API:

```text
train()
eval()
is_training()
parameters()
to(device)
```

`Module.to(device)` переносит все `Parameter`, возвращаемые `parameters()`.

Это важно: текущая система ещё не имеет полноценной recursive reflection по
полям класса, поэтому корректность зависит от `parameters()` конкретного Module.

---

# 24. Важная особенность inherited `Module.to`

Обычный inherited method stub не подходит для `Module.to`.

Почему:

если inherited stub вызывает:

```c
Module_to(...)
```

то внутри него статически будет:

```c
Module_parameters(...)
```

и override:

```text
TinyGPT.parameters()
Linear.parameters()
...
```

потеряется.

Поэтому codegen имеет специальный path:

> Для inherited `Module.to` генерируется concrete-class implementation,
> который вызывает `ConcreteClass_parameters(self)`.

Не удалять эту специализацию, пока в языке нет настоящего dynamic dispatch
для такого вызова.

---

# 25. Linear

`Linear`:

```text
weight [out_features, in_features]
bias   [1, out_features]
```

Forward:

```text
weight_t = weight.transpose()
output = input.matmul(weight_t)
result = output + bias
```

Для горячей output-проекции пример
`examples/ML/medium_gpt_native_ternary_train_gpu.oc` использует отдельный
`MatmulLinear`: он хранит полноточный weight в форме
`[in_features, out_features]` и не выполняет transpose на каждом forward.

Работает с batched input благодаря ND matmul + broadcasting.

---

# 26. LayerNorm module

Affine LayerNorm:

```text
normalized = input.layer_norm(-1, eps)
scaled = normalized * gamma
result = scaled + beta
```

Параметры:

```text
gamma [1, d_model]
beta  [1, d_model]
```

Инициализация gamma через обычный Tensor op внутри constructor ранее была
проблематичной, поэтому используется:

```text
Tensor.zeros(...)
ocean_tensor_fill(..., 1.0)
```

---

# 27. MultiHeadAttention

Текущий flow:

```text
input [B, T, C]

q_proj/k_proj/v_proj
    ↓
[B, T, C]

reshape
    ↓
[B, T, H, D]

permute
    ↓
[B, H, T, D]

scores = Q @ K^T
scaled = scores / sqrt(D)
masked = causal masked_fill
weights = softmax(-1)
context = weights @ V

permute back
reshape
out_proj
```

---

# 28. TransformerBlock

Текущий block:

```text
Pre-LN

x
 ↓
LayerNorm
 ↓
MultiHeadAttention
 ↓
residual add
 ↓
LayerNorm
 ↓
Linear d_model → d_ff
 ↓
ReLU
 ↓
Linear d_ff → d_model
 ↓
residual add
```

Dropout пока не является частью текущего минимального block.

---

# 29. GPT primitives

## Embedding

Input:

```text
indices [B, T] int64
weight  [V, C] float32
```

Output:

```text
[B, T, C]
```

Backward:

```text
scatter-add into weight.grad
```

Repeated token IDs должны аккумулировать gradients.

Token IDs non-differentiable.

## CrossEntropyLoss

Поддерживает:

```text
logits  [..., V]
target  [...]
```

Mean reduction.

Forward использует stable log-sum-exp.

Backward:

```text
softmax(logits) - one_hot(target)
```

с нормировкой на число элементов.

---

## GELU

`Tensor.gelu()` реализует GPT-2 tanh approximation:

```text
0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x^3)))
```

Forward и backward используются в GPT-2 GPU-модели. Для contiguous float32
Tensor обе операции выполняются native OpenCL kernels; неподдержанные dtype и
формы сохраняют correctness-first fallback semantics.

---

# 30. TinyGPT v0.1

Рабочая архитектура:

```text
token_embedding
+
position_embedding
    ↓
TransformerBlock × 2
    ↓
LayerNorm
    ↓
Linear lm_head
    ↓
logits
```

Проверенная конфигурация:

```text
vocab_size      = 8
context_length  = 7
d_model         = 16
n_heads         = 4
d_ff            = 64
n_layers        = 2
```

Toy task:

```text
input : 0 1 2 3 4 5 6
target: 1 2 3 4 5 6 7
```

Подтверждённый результат:

```text
initial loss = 2.079442
final loss   = 0.057157
predicted next token = 7

token embedding grad    = 1
position embedding grad = 1
lm head grad            = 1

[ok] Ocean TinyGPT v0.1
```

Это доказывает end-to-end:

```text
Embedding
→ attention
→ LayerNorm
→ residual
→ FFN
→ lm_head
→ CrossEntropy
→ backward
→ optimizer
```

## GPT-2-style ternary model

`examples/ML/gpt2_native_ternary.oc` добавляет полноценное training core
decoder-only архитектуры GPT-2:

```text
token embedding + learned position embedding
    ↓
pre-LN causal self-attention
    ↓
residual
    ↓
pre-LN MLP with GPT-2 GELU
    ↓
residual
    ↓
final LayerNorm + tied token-embedding LM head
```

Linear weights используют weight-only ternarization со straight-through
estimator: forward видит `{-scale, 0, +scale}`, а AdamW обновляет
полноточный master Parameter. Конфигурация smoke-теста компактная, но классы
поддерживают до 12 decoder blocks — глубину GPT-2 small.

Training остаётся QAT/STE-режимом: master weights и optimizer state остаются
`float32`. Для inference уже добавлен packed deployment path: 16 ternary
значений кодируются в один `int32` word (`00 = 0`, `01 = +1`, `10 = -1`), а
общий `scale` хранится отдельно. OpenCL packed matmul декодирует коды на лету,
включая fused bias path для `TernaryLinear`; tied LM-head также использует
transposed packed embedding weights. Q/K/V attention projections объединены в
`packed_qkv_inference_into`: один OpenCL workgroup загружает input tile один
раз и сразу пишет Q/K/V в три заранее выделенных `[... , d_model]` Tensor’а.
Inference path больше не создаёт промежуточный `[... , 3 * d_model]` buffer и
не запускает три post-projection slice kernels.

Канонический профиль доступен через Ocean-фабрику:

```ocean
var config: GPT2Config = gpt2_small_config()
```

Он задаёт:

```text
vocab_size = 50257
max_seq_len = 1024
d_model = 768
n_heads = 12
d_ff = 3072
n_layers = 12
```

`GPT2Embedding` использует GPT-2 initialization range `0.02`; обычный
`std.ml.Embedding` сохраняет старый default `0.1` для обратной совместимости.

Это архитектурно точная GPT-2-подобная модель, но пока не ABI-совместимый
загрузчик pretrained GPT-2: нет tokenizer/checkpoint loader, dropout и fused
`c_attn` layout. Inference path уже поддерживает per-layer KV-cache.

Для inference-only замера добавлен отдельный пример
`examples/ML/gpt2_native_ternary_inference.oc`. Он использует канонический
GPT-2 small профиль, делает один warmup token, отключает autograd через
`Tensor.set_grad_enabled(False)` и печатает elapsed time, milliseconds/token и
tokens/second. Prompt prefill заполняет GPU-resident `K/V` cache каждого
decoder block, после чего новый token обрабатывается только одним шагом
attention.

Для inference-only deployment добавлен отдельный lifecycle:

```text
model.prepare_inference()
    ↓
packed ternary weights готовы
    ↓
model.freeze_for_inference()
    ↓
освободить FP32 master weights TernaryLinear
```

`Parameter.release_data()` освобождает Tensor handle, а сам `Parameter` остаётся
частью model object. После freeze линейные слои используют только packed-веса;
повторный вызов freeze идемпотентен. Token/position embeddings намеренно
остаются FP32, поскольку они нужны для embedding lookup. Frozen model является
read-only inference object и не должен передаваться в optimizer/training path.

`Tensor.grad_enabled()` / `Tensor.set_grad_enabled(...)` — глобальный runtime
переключатель построения autograd graph. Он не заменяет `Module.eval()`: для
inference нужно вызывать оба API, если модель содержит train/eval-зависимые
операции.

OpenCL runtime больше не подменяет `CL_DEVICE_TYPE_GPU` на
`CL_DEVICE_TYPE_DEFAULT`: если ICD не предоставляет настоящий GPU, выбор
`device="gpu"` завершается явной ошибкой. Это не позволяет CPU OpenCL
реализации выдавать себя за GPU в benchmark.

---

# 31. TinyGPT parameters

На текущем этапе `TinyGPT.parameters()` содержит explicit/manual list.

Это намеренно.

Recursive parameter discovery по nested Module fields пока недостаточно
надёжен, поэтому нельзя тихо заменить explicit list на reflection-like
автоматику без отдельного тестового milestone.

---

# 32. Optimizers

## SGD

Рабочий базовый optimizer.

Используется для regression baseline.

## AdamW v0.1

Реализован:

```text
first moment m
second moment v
bias correction
epsilon
decoupled weight decay
```

Формула:

```text
m_t = beta1*m_(t-1) + (1-beta1)*g
v_t = beta2*v_(t-1) + (1-beta2)*g²

m_hat = m_t / (1-beta1^t)
v_hat = v_t / (1-beta2^t)

p = p - lr * weight_decay * p
      - lr * m_hat / (sqrt(v_hat) + eps)
```

Optimizer state имеет собственный:

```text
state_id
step
per-Parameter m/v
```

Два разных AdamW instance не должны разделять moment state.

Есть numerical two-step reference test.

TinyGPT + AdamW также проходит regression.

---

# 33. GPU architecture

Текущий GPU backend — **OpenCL**.

Public API не должен экспонировать:

```text
cl_mem
cl_context
cl_command_queue
```

Пользователь видит только:

```ocean
tensor.to("gpu")
model.to("gpu")
```

---

# 34. Tensor `.to(device)`

У Tensor давно есть:

```ocean
tensor.to("cpu")
tensor.to("gpu")
```

Runtime:

```text
CPU → GPU
GPU → CPU
GPU → GPU copy
CPU → CPU copy
```

GPU storage представлен opaque handle.

---

# 35. Новый `Parameter.to("gpu")`

Добавлен device transfer Parameter:

```ocean
parameter.to("gpu")
```

Семантика:

```text
old parameter Tensor
    ↓
Tensor.to("gpu")
    ↓
new Tensor
    ↓
requires_grad_(True)
    ↓
Parameter.data = moved Tensor
```

Цель:

- веса реально живут на GPU;
- optimizer получает новый leaf Tensor;
- дальнейший forward использует GPU Tensor.

---

# 36. Новый `Module.to("gpu")`

API:

```ocean
var model: TinyGPT = TinyGPT(...)
model.to("gpu")
```

Переносит все параметры:

```text
for parameter in model.parameters():
    parameter.to(device)
```

Для concrete inherited modules используется специальный codegen path,
описанный выше.

---

# 37. GPU kernels: что уже есть

OpenCL backend уже имеет native kernels / backend paths для части операций:

```text
matmul
binary arithmetic
scalar arithmetic
ternary quantization
fill
softmax по последней оси для float32
LayerNorm по последней оси для float32
softmax backward по последней оси для float32
LayerNorm backward по последней оси для float32
GELU forward/backward для float32
sum_dim/mean_dim по последней оси для float32
SGD update для GPU float32
AdamW update и GPU m/v buffers для float32
Embedding forward для GPU float32 weights + GPU int64 indices
Embedding backward с atomic accumulation для повторяющихся token IDs
CrossEntropy forward/backward для GPU float32 logits + GPU int64 targets
Batched matmul для GPU float32 с batch broadcasting
Transposed batched matmul для GPU autograd backward
GPU `permute`/`transpose_dims` для arbitrary-rank Tensor с stride-aware gather
```

В первую очередь оптимизированы:

```text
float32
int32
```

Matmul использует tiled kernel:

```text
8 × 8 workgroup
local memory tiles
bounds checks
```

Для autoregressive decode добавлен специализированный OpenCL `matvec` path:

```text
[1, K] × [K, N]
[1, 1, K] × [K, N]
    ↓
128-thread workgroup
local tile входного вектора
без фиктивных строк обычного 8 × 8 matmul
```

Это уменьшает лишнюю работу при `batch=1, sequence=1`, который является
основным режимом GPT-2 token-by-token inference. Остальные формы продолжают
использовать обычный tiled или batched matmul kernel.

Inference-only `TernaryLinear` дополнительно использует fused
`linear_inference`: для формы `[1, 1, K] × [K, N]` bias добавляется внутри того
же matvec kernel. Training/autograd path сохраняет обычную последовательность
`matmul` + `add`.

Для GPT-2 inference поверх этого добавлен fused packed QKV path. Вызовы
`q_proj`, `k_proj`, `v_proj` в `forward`, prompt prefill и cached decode
сводятся к одному `ocean_tensor_packed_qkv_split` kernel; training path
по-прежнему использует независимые проекции и autograd.

Для single-token GPU decode cached attention дополнительно использует
`ocean_tensor_packed_qkv_attention_decode`: один workgroup на attention head
считает packed Q/K/V, записывает текущую строку KV-cache, выполняет `QKᵀ`,
causal softmax и weighted-V reduction. Это убирает отдельные launches для
reshape/permute, cache write/slice, score matmul, softmax и context matmul.

---

# 38. GPU fallback semantics

Не все Tensor operations сейчас GPU-native.

Для неподдержанных path runtime может делать:

```text
GPU Tensor
   ↓
copy to CPU
   ↓
CPU implementation
   ↓
copy result to GPU
```

Это корректно функционально, но может быть очень медленно.

Особенно это касается части:

```text
broadcast-heavy operations
some ND paths and unsupported batched dtypes
autograd paths для неподдержанных осей и dtype
```

Поэтому:

> `device == "gpu"` ещё не означает, что весь Transformer forward является
> GPU-native без host round-trips.

---

# 39. GPU optimizer v0.2

SGD и AdamW были расширены для GPU-resident Parameter.

Для float32 GPU Parameters текущий native path:

```text
GPU weight + GPU grad
   ↓
OpenCL SGD/AdamW update kernel
   ↓
weight обновляется in-place
```

AdamW state (`m/v`) теперь хранится как opaque Tensor на том же device и
обновляется тем же kernel. CPU Parameters сохраняют отдельный прямой CPU path.

Для неподдержанных dtype/device runtime сохраняет корректную явную семантику;
GPU-native path сейчас ограничен contiguous float32.

Следующие этапы:

```text
GPU-native batched matmul
GPU-native optimizer kernels для других numeric dtypes
```

---

# 40. `ocean_tensor_copy_into`

Для device-aware optimizer update добавлен runtime primitive:

```c
ocean_tensor_copy_into(destination, source)
```

Он копирует данные в существующий Tensor handle без смены identity.

Это важно для Parameter:

- leaf metadata остаётся привязано к тому же Tensor handle;
- optimizer не заменяет Parameter object;
- GPU weight storage обновляется in-place semantically.

Не заменять это на:

```text
parameter = tensor.to(...)
```

в optimizer step, иначе можно разрушить autograd leaf identity/state.

---

# 41. GPU status на 2026-08-21

Добавлен integration test:

```text
tests/test_gpu_training_v01_ocean.py
tests/test_gpu_hotpaths_v01_runtime.py
```

Он должен проверить:

```text
Linear model.to("gpu")
input.to("gpu")
target.to("gpu")
forward on gpu
backward
AdamW step

Hotpath test additionally checks CPU/GPU equivalence for softmax, LayerNorm,
their last-axis backward gradients, reductions, SGD, and AdamW moments.
loss decreases
output.device() == "gpu"
parameter.device() == "gpu"
```

После установки OpenCL-пакетов в micromamba `base` оба GPU-теста проходят:

```text
2 passed
```

Рабочая настройка окружения:

```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate base
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
python -m pytest tests/test_gpu_training_v01_ocean.py tests/test_gpu_hotpaths_v01_runtime.py -q
```

В `base` установлены:

```text
opencl-headers
ocl-icd
```

Не следует добавлять `$CONDA_PREFIX/lib` в `LD_LIBRARY_PATH` в этом окружении:
NVIDIA OpenCL loader из системного CUDA runtime работает стабильнее conda
loader; conda используется для headers и `pkg-config` metadata.

Проверить:

```bash
ls -l /usr/include/CL/cl.h
ls -l /usr/local/include/CL/cl.h

ldconfig -p | grep -i opencl

clinfo -l

pkg-config --exists OpenCL
pkg-config --cflags OpenCL
pkg-config --libs OpenCL
```

На Debian/Ubuntu типичные packages:

```bash
sudo apt install ocl-icd-opencl-dev opencl-headers clinfo
```

После настройки обязательно:

```bash
python -m pytest tests/test_gpu_training_v01_ocean.py -q -s
```

GPU milestone считается подтверждённым только при:

```text
1 passed
```

а не `skipped`.

---

# 42. OpenCL vs CUDA

На текущем этапе не нужно немедленно выбрасывать OpenCL backend.

Правильный порядок:

1. довести окружение OpenCL;
2. проверить реальный GPU integration test;
3. измерить performance;
4. только после этого решать, нужен ли CUDA backend как отдельный backend.

Для H100 CUDA backend в долгосрочной перспективе логичен, но public API должен
остаться backend-neutral:

```ocean
model.to("gpu")
```

а не:

```ocean
model.to("cuda")
```

если мы хотим оставить абстракцию portable.

Внутри runtime позже можно выбирать:

```text
GPU backend
    ├── CUDA
    └── OpenCL
```

---

# 43. OpenCL build integration

`main.py::compile_c()` обнаруживает standard runtimes по generated includes.

Если Tensor runtime требует OpenCL и доступен:

```bash
pkg-config --exists OpenCL
```

compile path автоматически добавляет:

```text
OpenCL cflags
OpenCL link flags
-DOCEAN_TENSOR_ENABLE_OPENCL
```

Если OpenCL backend отсутствует, `to("gpu")` должен падать явно,
а не молча выполнять весь код на CPU.

Silent fallback устройства запрещён.

---

# 44. `.npy`

Поддерживается:

```text
Tensor.load_npy(path, device)
Tensor.save_npy(path)
```

Reader/writer не зависит от NumPy runtime.

Поддерживаются:

```text
bool
int8/int16/int32/int64
uint8/uint16/uint32/uint64
float16/float32/float64
```

Поддерживаются `.npy` v1/v2/v3.

Fortran-order и object/string dtypes пока не основной path.

GPU Tensor при save может копироваться на CPU.

---

# 45. OpenMP

Compiler поддерживает OpenMP pragmas и автоматически добавляет:

```text
-fopenmp
```

если generated C содержит OpenMP.

Не использовать OpenMP для кода с managed objects без validator guarantees.

---

# 46. Frontend/compiler quirks, которые уже встречались

Эти проблемы реальны и важны при следующих ML milestones.

## 46.1 Multiline class method signatures

Некоторые multiline signatures ранее:

- корректно резолвились на call site;
- но method implementation не попадал в `class_method_scopes`;
- generated C содержал call без implementation.

Практический workaround:

> Пока держать критические class method signatures в одну строку,
> если parser/codegen regression не доказал обратное.

## 46.2 Method calls in constructors

Constructor lowering более ограничен, чем обычный method body.

Не предполагать, что любой expression, работающий в `forward`, безопасно
работает внутри `__init__`.

## 46.3 `len(self.parameters)`

Ранее мог генерироваться undefined `builtin_len`.

Надёжный pattern:

```ocean
var parameters: list[Parameter] = self.parameters
len(parameters)
```

если regression снова проявится.

## 46.4 `Tensor.item()` + reassignment

Ранее:

```ocean
final_loss = final_loss_tensor.item()
```

мог типизироваться как `unknown`.

Надёжный pattern:

```ocean
var final_loss_value: float64 = final_loss_tensor.item()
```

с fresh typed declaration.

## 46.5 Chained attribute methods в `print`

Проблемный pattern:

```ocean
print(projection.weight.has_grad())
```

Validator мог интерпретировать промежуточный:

```text
projection.weight
```

как standalone variable.

Надёжнее:

```ocean
var weight: Parameter = projection.weight
print(weight.has_grad())
```

## 46.6 1D Tensor assignment

Исторически:

```ocean
positions[0] = 0
```

мог lower'иться как обычный C indexing:

```c
positions[0] = 0;
```

где `positions` — Ocean Tensor wrapper.

Для TinyGPT использовался workaround:

```text
positions [1, T]
positions[0, i]
```

Пока отдельный regression не докажет исправление 1D set path, помнить про это.

---

# 47. Bool/int lowering

Python implementation frontend должен отличать:

```python
bool
```

от:

```python
int
```

Поскольку:

```python
isinstance(True, int) == True
```

нельзя проверять integer literal до bool.

Надёжное правило:

```python
isinstance(value, bool)
```

должно идти раньше integer handling.

---

# 48. Generated runtime / demand driven

Runtime helpers должны генерироваться/линковаться только когда нужны.

Не возвращаться к ситуации, где:

```ocean
sqrt(...)
```

тянет list/string/ARC/Tensor helpers.

---

# 49. Testing philosophy

Для Ocean недостаточно только проверить generated C text.

Нужны уровни:

```text
1. parser / Typed IR
2. Validator
3. generated C
4. strict GCC/Clang compile
5. binary execution
6. numerical reference
7. finite differences для autograd
8. regression suite
```

Для ML операций желательно:

```text
forward reference
backward reference
finite differences
shape tests
broadcast tests
lifetime tests
```

---

# 50. Текущая regression база

Последний фактически полученный результат после inference-memory lifecycle:

```text
143 passed, 2 skipped, 1 failed
```

`skipped`:

```text
GPU training/inference integration
GPU Tensor hotpaths integration
```

CPU TinyGPT, AdamW, GPT-2 smoke, inference benchmark compilation и остальные
ML regression tests проходят. Реальное выполнение GPT-2 inference benchmark на
GPU ещё не подтверждено, поскольку OpenCL platform отсутствует в текущем
окружении. Один сетевой failure вызван ограничением sandbox на `bind/listen`.

При изменениях Tensor/autograd всегда отдельно прогонять:

```bash
python -m pytest tests/test_layernorm_v01_ocean.py -q
python -m pytest tests/test_transformer_block_v01_ocean.py -q
python -m pytest tests/test_tiny_gpt_v01_ocean.py -q
python -m pytest tests/test_tiny_gpt_adamw_v01_ocean.py -q
python -m pytest tests/test_adamw_v01_runtime.py -q
python -m pytest tests/test_adamw_v01_ocean.py -q
python -m pytest
```

После настройки OpenCL:

```bash
python -m pytest tests/test_gpu_training_v01_ocean.py -q -s
```

---

# 51. ASan обязателен для autograd lifetime bugs

Если появляется SIGSEGV после:

```text
forward
backward
```

не гадать.

Собирать:

```bash
-fsanitize=address
-O0
-g3
-fno-omit-frame-pointer
```

ASan уже позволил найти LayerNorm use-after-free.

Особенно подозрительны:

```text
node->left->tensor
node->right->tensor
```

в backward implementations.

---

# 52. Что НЕ делать

Не следует:

- возвращать Tensor как `list[list[T]]`;
- удалять autograd metadata при каждом Tensor release;
- отключать bounds checks через `NDEBUG`;
- делать silent CPU fallback для `device="gpu"` на уровне device selection;
- считать GPU feature готовой только потому, что тест `skipped`;
- reintroduce raw pointer identity как autograd identity;
- делать generic SIMD, предполагающий фиксированный element size;
- использовать multiple inheritance через unsafe C casts;
- добавлять отдельный ad-hoc AST/codegen path для каждой Tensor операции;
- скрывать CPU round-trip под словами "GPU-native".

---

# 53. Ближайший roadmap

## P0 — подтвердить GPU training

1. Установить/настроить OpenCL headers/runtime.
2. Убедиться, что:

```bash
pkg-config --exists OpenCL
clinfo -l
```

видят backend/device.
3. Добиться:

```text
tests/test_gpu_training_v01_ocean.py → passed
```

4. Добавить GPU test для:
   - SGD;
   - AdamW;
   - Linear inference;
   - Linear training.

## P1 — GPU-native Transformer path

Убрать host round-trip из:

```text
broadcast binary
softmax
LayerNorm
GELU forward/backward
reductions
```

После этого:

```text
TinyGPT CPU vs GPU
```

benchmark.

Обязательно сравнивать:

```text
loss
logits
gradients
predicted token
runtime
memory
```

## P2 — autoregressive inference

Реализовать:

```text
TinyGPT.generate()
greedy decoding
last-token logits
max_new_tokens
```

Эта базовая схема сохранена для регрессионного сравнения; production-like
GPT-2 inference path использует KV-cache ниже.

## P3 — KV cache

Реализовано в `GPT2Ternary.generate_greedy_kv()`:

```text
GPU-resident K/V cache per decoder block
prompt prefill
single-token decode
native cache-row write kernel
native cache-prefix read kernel
```

Осталось проверить на реальном GPU:

```text
identical tokens
close logits
speed/token
scaling with context length
```

## P4 — positional encoding

Добавить:

```text
Tensor.arange
automatic position ids
RoPE
```

После этого learned positional embedding можно оставить как supported option.

## P5 — CUDA backend

Если OpenCL на H100 функционально работает, но performance ограничивает проект:

```text
Tensor backend interface
    ├── CPU
    ├── OpenCL
    └── CUDA
```

Public API оставить:

```ocean
.to("gpu")
```

Backend selection должен быть runtime/build-level detail.

## P6 — optimizer performance

Перевести:

```text
SGD
AdamW
m/v state
```

на GPU buffers и kernels.

## P7 — views / memory

Сделать:

```text
zero-copy reshape
zero-copy transpose/permute where valid
offset
strides
storage ownership
memory pool
arena
```

---

# 54. ML roadmap после GPU

После GPU-native TinyGPT:

```text
Dropout
RoPE
RMSNorm
KV cache
FlashAttention-like kernel
weight tying
checkpoint load/save
tokenizer
dataset API
mixed precision
quantization
```

---

# 55. Quantization roadmap

Уже поддерживаются compact numeric dtypes, включая `int8`.

Но это ещё не полноценная quantization API.

Нужно отдельно:

```text
scale
zero_point
per-tensor
per-channel
weight-only
int8
int4
int2
```

Не смешивать packed storage и обычный dtype API без metadata.

---

# 56. Backend-neutral device model

Долгосрочная желаемая модель:

```ocean
var model = TinyGPT(...)
model.to("gpu")

var x = x.to("gpu")
var y = model.forward(x)
```

А внутри:

```text
device="gpu"
    ↓
preferred GPU backend
    ↓
CUDA or OpenCL
```

Если explicit backend control понадобится, лучше отдельная configuration layer,
а не разрушение основного PyTorch-like API.

---

# 57. Команды для разработки

Полный regression:

```bash
python -m pytest
```

Конкретный ML test:

```bash
python -m pytest tests/test_tiny_gpt_v01_ocean.py -q
```

GPU:

```bash
python -m pytest tests/test_gpu_training_v01_ocean.py -q -s
```

Проверка OpenCL:

```bash
pkg-config --exists OpenCL
pkg-config --cflags OpenCL
pkg-config --libs OpenCL
clinfo -l
```

Проверка NVIDIA:

```bash
nvidia-smi
nvcc --version
```

Graphify:

```bash
./.venv/bin/graphify update .
```

---

# 58. На что смотреть при следующем падении TinyGPT

Если ошибка validation:

```text
function not declared
```

проверить C-function whitelist Validator.

Если GCC:

```text
incompatible types
```

смотреть generated C и Tensor intrinsic lowering.

Если runtime:

```text
shape/rank mismatch
```

проверять broadcasting/sum_to_meta/matmul.

Если SIGSEGV:

```text
ASan immediately
```

Если backward UAF:

```text
ищем node->left->tensor / node->right->tensor
```

и решаем через:

```text
saved_left / saved_right
```

если operation нуждается в forward values.

---

# 59. Текущая точка продолжения

Самый правильный следующий шаг после этого handoff:

```text
1. починить OpenCL окружение
2. добиться реального pass GPU integration test
3. добавить TinyGPT GPU inference test
4. добавить TinyGPT GPU training test
5. профилировать CPU fallback
6. переносить горячие Transformer operations в GPU-native kernels
```

До выполнения пункта 2 не считать GPU training полностью завершённым.

---

# 60. Ключевые инварианты проекта

Если нужно запомнить только несколько вещей:

1. **Typed IR — основной compiler contract.**
2. **ARC и ownership должны оставаться deterministic.**
3. **Autograd metadata != Tensor storage lifetime.**
4. **Backward, которому нужны forward values, сохраняет собственный Tensor copy.**
5. **Tensor identity не равен raw pointer address.**
6. **Module.to(device) обязан использовать concrete model parameters().**
7. **`gpu` не означает GPU-native для каждого op, пока есть host fallback.**
8. **Skipped GPU test не является подтверждением GPU execution.**
9. **Каждый новый ML primitive должен иметь numerical/backward test.**
10. **После каждого крупного ML/runtime изменения нужен полный regression suite.**

---

# 61. Итог

На текущем этапе Ocean уже умеет не только компилировать обычный код в C,
но и обучать небольшой GPT-подобный Transformer end-to-end:

```text
Ocean source
→ Tensor
→ Embedding
→ Transformer
→ CrossEntropy
→ autograd
→ AdamW
→ trained TinyGPT
```

CPU path подтверждён тестами.

GPU device abstraction и `.to("gpu")` уже заведены в Tensor/Parameter/Module,
но реальный GPU training test ещё должен быть подтверждён после настройки
OpenCL development/runtime environment.

Следующий крупный рубеж:

> **полностью подтверждённый TinyGPT training/inference на GPU, а затем
> GPU-native Transformer kernels и оптимизация KV-cache generation.**

---

# 62. Backend/server development — `std/net`

Ocean предназначен не только для ML/HPC. На нём также проектируется и уже
частично реализован **обычный серверный backend stack** поверх C11/POSIX.

Это важная часть проекта и её нельзя считать второстепенной или исторической.

Цель:

```text
Ocean source
    ↓
typed Request / Response / Router / App
    ↓
std/net + std/json
    ↓
C11/POSIX sockets + pthread worker pool
    ↓
native HTTP backend executable
```

То есть на Ocean должен быть возможен код уровня:

```ocean
def get_user(request: Request) -> Response:
    var root: Json = Json.object()
    var name: Json = Json.str("Ocean")

    root.set("name", name)

    return Response.json_value(root)


def main() -> int:
    var app: App = App.create()

    app.get("/users/{id}", get_user)
    app.workers(8)
    app.queue_size(256)
    app.keep_alive(5000)

    app.run("0.0.0.0", 8080)

    return 0
```

Не обязательно, чтобы конкретно все показанные convenience methods уже были
стабильны в tracked sources; это целевой API направления.

## 62.1 Структура `std/net`

Текущая/целевая структура:

```text
std/net/
├── socket.oc
├── http.oc
├── web.oc
├── net_runtime.h
├── net_runtime.c
├── web_runtime.h
├── web_runtime.c
└── README.md
```

Слои:

```text
socket.oc
    ↓
TCP sockets

http.oc
    ↓
HTTP client

web.oc
    ↓
HTTP server / application API
```

Пользовательский Ocean-код не должен работать напрямую с `socket fd`,
`struct sockaddr`, `pthread_t` или private C layouts.

---

# 63. Typed backend API

Web layer должен скрывать raw handles за обычными Ocean class objects:

```text
Request
Response
App
Router
Next
```

Внутри:

```text
Request  -> ocean_web_request_t
Response -> ocean_web_response_t
App      -> ocean_web_app_t
Router   -> private router runtime
Next     -> ocean_web_next_t
```

Пользовательский handler:

```ocean
def handler(request: Request) -> Response:
    ...
```

а не:

```c
ocean_web_response_t handler(ocean_web_request_t request)
```

Raw ABI — implementation detail stdlib.

---

# 64. `Request`

Основной API request:

```text
Request.method(request)
Request.path(request)
Request.query(request)
Request.body(request)
Request.json(request)
Request.remote(request)

Request.header(request, name, default)

Request.query_param(request, name, default)
Request.path_param(request, name, default)
```

Следующий typed слой:

```text
Request.path_int(...)
Request.query_int(...)
Request.query_bool(...)
```

В перспективе:

```text
Request.state / request context
cookies
multipart/form-data
uploads
```

---

# 65. `Response`

Основной API:

```text
Response.text(...)
Response.text_status(...)

Response.json(...)
Response.json_status(...)

Response.json_value(Json)
Response.json_value_status(status, Json)

Response.html(...)

Response.empty(status)
Response.redirect(...)

Response.add_header(...)
```

Важно интегрировать backend с `std/json`, а не собирать JSON через string
concatenation.

Пример:

```ocean
def health(request: Request) -> Response:
    var root: Json = Json.object()
    var status: Json = Json.str("ok")

    root.set("status", status)

    return Response.json_value(root)
```

---

# 66. ABI naming rule для web backend

Ocean classes lower'ятся с `ocean_` prefix.

Например:

```text
class Request
    ↓
ocean_Request
ocean_create_Request(...)
ocean_Request_raw_handle(...)
```

и:

```text
class Response
    ↓
ocean_Response
ocean_create_Response(...)
ocean_Response_take_handle(...)
```

Поэтому callback ABI нельзя объявлять через несуществующие:

```c
struct Request;
struct Response;
```

Правильная идея:

```c
typedef struct ocean_Request ocean_Request;
typedef struct ocean_Response ocean_Response;

typedef ocean_Response *(*ocean_web_handler_t)(
    ocean_Request *request
);
```

Generated type names должны совпадать с реальным Ocean class lowering.

---

# 67. Важная FFI-грабля для backend wrappers

Проблемный pattern:

```ocean
@ocean_web_get(self.handle, path, handler)
```

или:

```ocean
@ocean_web_next_call(self.handle, request_handle)
```

Attribute access внутри raw C-call arguments ранее мог lower'иться буквально:

```c
self.handle
```

вместо:

```c
self->handle
```

Надёжный stdlib pattern:

```ocean
def raw_handle(self) -> ocean_web_app_t:
    return self.handle
```

затем:

```ocean
unsafe:
    var app_handle: ocean_web_app_t = self.raw_handle()
    @ocean_web_get(app_handle, path, handler)
```

То есть:

```text
Ocean method
    ↓
local C-typed handle
    ↓
raw C call
```

Долгосрочно это надо исправить в generic compiler lowering C-call arguments,
а не только обходить внутри `std/net`.

---

# 68. Worker pool

Целевой HTTP server использует bounded fixed-size worker pool:

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

API:

```ocean
var app: App = App.create()

app.workers(8)
app.queue_size(256)
```

Это предпочтительнее thread-per-request:

```text
thread-per-request
    ↓
unbounded pthread count
    ↓
unpredictable memory / scheduling

fixed worker pool
    ↓
bounded pthread count
    ↓
predictable resources
```

Worker владеет accepted connection во время обработки request/keep-alive
sequence.

Если web runtime использует pthreads, CLI build path должен автоматически
добавлять:

```text
-pthread
```

---

# 69. HTTP/1.1 keep-alive

Целевой API:

```ocean
app.keep_alive(5000)
app.max_keep_alive_requests(100)
```

Один TCP connection:

```text
connect
    ↓
GET /a
    ↓
GET /b
    ↓
POST /c
    ↓
close
```

Runtime должен учитывать:

```http
Connection: keep-alive
Connection: close
```

и корректно выставлять:

```text
Content-Length
Connection
Keep-Alive
```

Текущий целевой scope:

```text
HTTP/1.1 sequential keep-alive
```

Пока не приоритет:

```text
HTTP pipelining
HTTP/2
HTTP/3
```

При connection-oriented worker pool keep-alive connection закрепляется за
worker до close/timeout, поэтому слишком большой idle timeout может ухудшать
concurrency.

---

# 70. Middleware

Целевой API:

```ocean
def request_log(
    request: Request,
    call_next: Next
) -> Response:

    print("before")

    var response: Response = call_next.call(request)

    print("after")

    return response


app.middleware(request_log)
```

Chain:

```text
request
    ↓
middleware #1 before
    ↓
middleware #2 before
    ↓
route handler
    ↓
middleware #2 after
    ↓
middleware #1 after
    ↓
response
```

На этом слое должны строиться:

```text
CORS
access logging
request id
authentication
authorization
timing
recovery
rate limiting
structured errors
```

`Next` — Ocean wrapper над runtime callback state.

Raw `ocean_web_next_t` не должен появляться в обычном application code.

---

# 71. Router

Целевой Python-like API:

```ocean
var app: App = App.create()

var api: Router = Router.create("/api/v1")

api.get("/users/{id}", get_user)
api.post("/users", create_user)
api.put("/users/{id}", replace_user)
api.patch("/users/{id}", update_user)
api.delete("/users/{id}", delete_user)

app.include(api)
```

Результат:

```text
GET     /api/v1/users/{id}
POST    /api/v1/users
PUT     /api/v1/users/{id}
PATCH   /api/v1/users/{id}
DELETE  /api/v1/users/{id}
```

Поддерживаемый/целевой route API:

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
```

Router не должен зависеть от private layout:

```c
app->routes
```

или от private internal route type.

Правильная архитектура:

```text
Router
    owns prefix
    owns private route descriptions
        ↓
App.include(router)
        ↓
public ocean_web_route(...)
```

То есть Router зависит от **public web runtime ABI**, а не от internal structs.

---

# 72. Nested routers

Следующий DX-layer:

```ocean
var api: Router = Router.create("/api")
var users: Router = Router.create("/users")

users.get("/{id}", get_user)

api.include(users)
app.include(api)
```

Итог:

```text
/api/users/{id}
```

Это важно для реального backend проекта, где routes делятся по модулям.

---

# 73. Thread safety backend-кода

ARC сейчас non-atomic.

Поэтому базовое правило web runtime:

> Managed objects одного request должны быть thread-confined одному worker.

Безопасный типичный lifetime:

```text
worker
    ↓
Request
Response
Json
str
list
temporary classes
    ↓
destroy before / at request completion
```

Нельзя считать безопасным общий mutable:

```text
list
dict
Json
class instance
Tensor
```

если несколько workers используют его одновременно без synchronization.

Для полноценного shared application state нужны:

```text
Shared[T]
atomic ARC
Send/Sync-like rules
mutex/rwlock
thread-safe containers
```

До этого глобальное mutable state должно использовать raw/native synchronized
runtime primitives или быть архитектурно изолировано.

---

# 74. Что уже делает Ocean пригодным для backend

Даже до полного FastAPI-like DX у языка уже есть необходимые базовые слои:

```text
native C11 compilation
POSIX integration
TCP sockets
HTTP runtime
JSON
File/BinaryFile
classes
lists/dicts/strings
pthreads
worker-pool design
typed request/response wrappers
routing foundation
middleware design
```

То есть backend/server направление — **первоклассная цель Ocean** наряду с
ML/HPC.

---

# 75. Regression suite для `std/net`

Минимум, который должен быть закреплён тестами:

```text
GET / -> 200

unknown route -> 404
known path + wrong method -> 405
HEAD fallback to GET

path params
query params

POST JSON body
Response.json_value(Json)

custom response headers

middleware before/after ordering
middleware response mutation

multiple concurrent connections

keep-alive:
    two sequential requests on one socket
    Connection: close
    max requests
    idle timeout

worker queue saturation

Router prefix
Router path params after include
Router GET
Router POST
Router PUT
Router PATCH
Router DELETE
```

C runtime отдельно:

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

Для server runtime обязательны также:

```text
ASan
UBSan
concurrent smoke tests
socket disconnect tests
malformed HTTP tests
```

---

# 76. Backend roadmap

После стабилизации worker pool / keep-alive / middleware / Router:

1. nested routers;
2. typed path/query parsing;
3. request state/context;
4. CORS middleware;
5. structured HTTP errors;
6. graceful shutdown `SIGINT` / `SIGTERM`;
7. cookies;
8. multipart/form-data;
9. uploads;
10. `FileResponse` / `sendfile()`;
11. streaming responses;
12. TLS/HTTPS отдельным runtime layer;
13. WebSocket;
14. OpenAPI/schema generation;
15. request/response model validation;
16. database client ecosystem;
17. connection pools;
18. observability / metrics / tracing.

`async/await` пока не обязателен.

Для текущей ownership-модели fixed worker pool проще, предсказуемее и уже
достаточно полезен для реального backend-кода.

---

# 77. Две равноправные области Ocean

Развитие языка сейчас имеет две большие прикладные вертикали:

```text
Ocean
├── Backend / systems
│   ├── HTTP
│   ├── TCP
│   ├── JSON
│   ├── files
│   ├── pthread workers
│   └── routing/middleware
│
└── ML / HPC
    ├── Tensor
    ├── autograd
    ├── Transformer
    ├── TinyGPT
    ├── CPU/GPU
    └── OpenMP
```

Нельзя развивать ML часть ценой удаления или забвения backend/server части.

Обе области опираются на одни и те же ключевые свойства языка:

```text
native compilation
ownership
predictable memory
C ABI
classes
containers
static validation
```
