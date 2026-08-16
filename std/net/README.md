# `std/net` — networking и HTTP/Web для Ocean

Модуль `std/net` предоставляет сетевой стек Ocean:

- TCP-сокеты;
- HTTP/1.1 client;
- HTTP/Web server;
- маршрутизацию по HTTP method и URL;
- path/query parameters;
- работу с request headers и body;
- типизированные `Request` и `Response`;
- интеграцию с `std/json`.

Структура модуля:

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

---

## Импорт

### TCP sockets

```python
import <std/net/socket.oc>
```

### HTTP client

```python
import <std/net/http.oc>
```

### HTTP/Web server

```python
import <std/net/web.oc>
```

`web.oc` сам использует `std/json`, поэтому в web-приложении можно работать с `Json`.

При необходимости можно импортировать JSON напрямую:

```python
import <std/json/json.oc>
import <std/net/web.oc>
```

---

# 1. TCP sockets

Класс `Socket` предоставляет низкоуровневый TCP API.

## TCP client

```python
import <std/net/socket.oc>


def main() -> int:
    var socket: Socket = Socket.connect(
        "127.0.0.1",
        8080,
        5000
    )

    socket.send("Hello from Ocean!")

    var response: str = socket.recv(4096)

    print(response)

    socket.close()

    return 0
```

Из-за текущих ограничений parser рекомендуется писать вызовы с аргументами в одну строку:

```python
var socket: Socket = Socket.connect("127.0.0.1", 8080, 5000)
```

---

## TCP server

```python
import <std/net/socket.oc>


def main() -> int:
    var server: Socket = Socket.tcp()

    server.bind("127.0.0.1", 8080, True)
    server.listen(128)

    print("Listening on 127.0.0.1:8080")

    while True:
        var client: Socket = server.accept()

        var request: str = client.recv(4096)

        print(request)

        client.send("Hello from Ocean!")
        client.close()

    return 0
```

---

## `Socket` API

### Создание сокета

```python
var socket: Socket = Socket.tcp()
```

### Подключение

```python
var socket: Socket = Socket.connect(
    "example.com",
    80,
    5000
)
```

Аргументы:

```text
host        hostname или IP
port        TCP port
timeout_ms  timeout в миллисекундах
```

### Bind

```python
server.bind(
    "0.0.0.0",
    8080,
    True
)
```

Третий аргумент включает `SO_REUSEADDR`.

### Listen

```python
server.listen(128)
```

### Accept

```python
var client: Socket = server.accept()
```

### Send

```python
var sent: int = client.send("hello")
```

### Receive

```python
var data: str = client.recv(8192)
```

### Timeout

```python
socket.set_timeout(5000)
```

### Проверка состояния

```python
var opened: bool = socket.is_open()
```

### Адрес клиента

```python
var peer: str = socket.peer_address()
```

### Локальный адрес

```python
var local: str = socket.local_address()
```

### Закрытие

```python
socket.close()
```

---

# 2. HTTP client

`HTTP` предоставляет простой HTTP/1.1 client.

```python
import <std/net/http.oc>
```

---

## GET

```python
def main() -> int:
    var response: HttpResponse = HTTP.get(
        "http://127.0.0.1:8080/",
        5000
    )

    print(response.status())
    print(response.body())

    return 0
```

---

## GET с headers

```python
var headers: str = "Authorization: Bearer token\r\n"

var response: HttpResponse = HTTP.get_with_headers(
    "http://127.0.0.1:8080/api",
    headers,
    5000
)
```

---

## POST

```python
var response: HttpResponse = HTTP.post(
    "http://127.0.0.1:8080/users",
    "{\"name\":\"Ocean\"}",
    "application/json",
    5000
)
```

---

## PUT

```python
var response: HttpResponse = HTTP.put(
    "http://127.0.0.1:8080/users/1",
    "{\"name\":\"Ocean\"}",
    "application/json",
    5000
)
```

---

## DELETE

```python
var response: HttpResponse = HTTP.delete(
    "http://127.0.0.1:8080/users/1",
    5000
)
```

---

## Произвольный HTTP method

```python
var response: HttpResponse = HTTP.request(
    "PATCH",
    "http://127.0.0.1:8080/users/1",
    "Content-Type: application/json\r\n",
    "{\"enabled\":true}",
    5000
)
```

---

## `HttpResponse`

### HTTP status

```python
var status: int = response.status()
```

### Успешный status

```python
var ok: bool = response.ok()
```

### Status text

```python
var text: str = response.status_text()
```

### Headers

```python
var headers: str = response.headers()
```

### Body

```python
var body: str = response.body()
```

---

# 3. Web server

`std/net/web.oc` — server-side HTTP framework поверх networking runtime.

Основные типы:

```text
App
Request
Response
```

Типичный handler:

```python
def index(request: Request) -> Response:
    return Response.text("Hello from Ocean!")
```

---

# 4. Минимальный HTTP server

```python
import <std/net/web.oc>


def index(request: Request) -> Response:
    return Response.text("Hello from Ocean!")


def main() -> int:
    var app: App = App.create()

    app.get("/", index)

    app.serve("127.0.0.1", 8080)

    return 0
```

Запуск:

```bash
ocean run ./server.oc
```

Проверка:

```bash
curl http://127.0.0.1:8080/
```

Ответ:

```text
Hello from Ocean!
```

---

# 5. Routing

`App` поддерживает основные HTTP methods:

```python
app.get(path, handler)
app.post(path, handler)
app.put(path, handler)
app.patch(path, handler)
app.delete(path, handler)
app.options(path, handler)
app.head(path, handler)
app.any(path, handler)
```

Также доступна регистрация произвольного HTTP method:

```python
app.route("PROPFIND", "/storage/{name}", handler)
```

---

## GET

```python
def index(request: Request) -> Response:
    return Response.text("index")


app.get("/", index)
```

---

## POST

```python
def create_user(request: Request) -> Response:
    return Response.text_status(201, "created")


app.post("/users", create_user)
```

---

## PUT

```python
def replace_user(request: Request) -> Response:
    return Response.text("updated")


app.put("/users/{id}", replace_user)
```

---

## PATCH

```python
def patch_user(request: Request) -> Response:
    return Response.text("patched")


app.patch("/users/{id}", patch_user)
```

---

## DELETE

```python
def delete_user(request: Request) -> Response:
    return Response.empty(204)


app.delete("/users/{id}", delete_user)
```

---

# 6. Path parameters

Роуты поддерживают параметры вида:

```text
/users/{id}
```

Пример:

```python
def get_user(request: Request) -> Response:
    var user_id: str = Request.path_param(
        request,
        "id",
        ""
    )

    return Response.text(user_id)


app.get("/users/{id}", get_user)
```

Запрос:

```bash
curl http://127.0.0.1:8080/users/42
```

Ответ:

```text
42
```

Можно использовать несколько path parameters:

```text
/projects/{project}/users/{user}
```

---

# 7. Query parameters

Для URL:

```text
/search?q=ocean&page=2
```

используется:

```python
def search(request: Request) -> Response:
    var query: str = Request.query_param(
        request,
        "q",
        ""
    )

    var page: str = Request.query_param(
        request,
        "page",
        "1"
    )

    return Response.text(query)
```

Значения query parameters проходят URL decoding.

---

# 8. Request

`Request` — типизированное представление входящего HTTP request.

Handler всегда может использовать сигнатуру:

```python
def handler(request: Request) -> Response:
```

---

## Method

```python
var method: str = Request.method(request)
```

Например:

```text
GET
POST
PUT
PATCH
DELETE
```

---

## Path

```python
var path: str = Request.path(request)
```

---

## Raw query string

```python
var query: str = Request.query(request)
```

---

## Body

```python
var body: str = Request.body(request)
```

---

## Client address

```python
var remote: str = Request.remote(request)
```

---

## Header

```python
var authorization: str = Request.header(
    request,
    "Authorization",
    ""
)
```

Третий аргумент — значение по умолчанию.

---

## Query parameter

```python
var page: str = Request.query_param(
    request,
    "page",
    "1"
)
```

---

## Path parameter

```python
var user_id: str = Request.path_param(
    request,
    "id",
    ""
)
```

---

# 9. JSON request body

`Request` интегрирован с `std/json`.

Например клиент отправляет:

```json
{
    "name": "Alice",
    "enabled": true
}
```

Handler:

```python
def create_user(request: Request) -> Response:
    var body: Json = Request.json(request)

    var name_json: Json = body.get("name")
    var name: str = name_json.as_str()

    return Response.text(name)
```

`Request.json()` эквивалентен:

```python
var body: str = Request.body(request)
var json: Json = Json.parse(body)
```

---

# 10. Response

Все handlers должны возвращать:

```python
Response
```

Например:

```python
def index(request: Request) -> Response:
    return Response.text("Hello")
```

---

## Text response

```python
return Response.text("Hello")
```

HTTP status:

```text
200 OK
```

Content-Type:

```text
text/plain; charset=utf-8
```

---

## Text response с status

```python
return Response.text_status(
    201,
    "created"
)
```

---

## HTML response

```python
return Response.html(
    "<h1>Hello from Ocean</h1>"
)
```

---

## HTML с status

```python
return Response.html_status(
    404,
    "<h1>Not found</h1>"
)
```

---

## Empty response

```python
return Response.empty(204)
```

---

## Redirect

```python
return Response.redirect("/login")
```

По умолчанию:

```text
302 Found
```

Можно указать status:

```python
return Response.redirect_status(
    307,
    "/new-location"
)
```

---

# 11. JSON response

Есть два варианта.

## Готовая JSON-строка

```python
return Response.json(
    "{\"ok\":true}"
)
```

или:

```python
return Response.json_status(
    201,
    "{\"created\":true}"
)
```

Этот вариант следует использовать только когда JSON уже сериализован.

---

## `Json` из `std/json`

Рекомендуемый вариант:

```python
def get_json(request: Request) -> Response:
    var root: Json = Json.object()

    var name: Json = Json.str("Ocean")
    var version: Json = Json.int(1)
    var enabled: Json = Json.bool(True)

    root.set("name", name)
    root.set("version", version)
    root.set("enabled", enabled)

    return Response.json_value(root)
```

Ответ:

```json
{
    "name": "Ocean",
    "version": 1,
    "enabled": true
}
```

`Response.json_value()` сериализует `Json` через `Json.stringify()`.

---

## JSON response с status

```python
return Response.json_value_status(
    201,
    root
)
```

---

# 12. Response headers

К response можно добавить пользовательский header:

```python
def index(request: Request) -> Response:
    var response: Response = Response.text("hello")

    Response.add_header(
        response,
        "X-Request-ID",
        "abc123"
    )

    return response
```

---

# 13. Полный пример REST API

```python
import <std/json/json.oc>
import <std/net/web.oc>


def index(request: Request) -> Response:
    return Response.text("Hello from Ocean!")


def hello(request: Request) -> Response:
    var name: str = Request.path_param(request, "name", "world")
    return Response.text("Hello, " + name + "!")


def search(request: Request) -> Response:
    var query: str = Request.query_param(request, "q", "")

    var root: Json = Json.object()
    var value: Json = Json.str(query)

    root.set("query", value)

    return Response.json_value(root)


def create_user(request: Request) -> Response:
    var body: Json = Request.json(request)

    var response: Json = Json.object()
    response.set("user", body)

    return Response.json_value_status(201, response)


def update_user(request: Request) -> Response:
    var user_id: str = Request.path_param(request, "id", "")
    var body: Json = Request.json(request)

    var root: Json = Json.object()
    var id_value: Json = Json.str(user_id)

    root.set("id", id_value)
    root.set("body", body)

    return Response.json_value(root)


def patch_user(request: Request) -> Response:
    var user_id: str = Request.path_param(request, "id", "")

    return Response.text(
        "patched user " + user_id
    )


def delete_user(request: Request) -> Response:
    return Response.empty(204)


def get_json(request: Request) -> Response:
    var root: Json = Json.object()

    var name: Json = Json.str("Ocean")
    var version: Json = Json.int(1)
    var enabled: Json = Json.bool(True)

    var values: Json = Json.arr()

    var first: Json = Json.int(10)
    var second: Json = Json.int(20)

    values.append(first)
    values.append(second)

    root.set("name", name)
    root.set("version", version)
    root.set("enabled", enabled)
    root.set("values", values)

    return Response.json_value(root)


def main() -> int:
    var app: App = App.create()

    app.set_server_header("Ocean/0.1")

    app.get("/", index)
    app.get("/hello/{name}", hello)
    app.get("/search", search)

    app.post("/users", create_user)
    app.put("/users/{id}", update_user)
    app.patch("/users/{id}", patch_user)
    app.delete("/users/{id}", delete_user)

    app.get("/get_json", get_json)

    app.serve("127.0.0.1", 8080)

    return 0
```

---

# 14. Проверка REST API через `curl`

## GET

```bash
curl http://127.0.0.1:8080/
```

---

## Path parameter

```bash
curl http://127.0.0.1:8080/hello/Alice
```

---

## Query parameter

```bash
curl 'http://127.0.0.1:8080/search?q=Ocean'
```

---

## POST JSON

```bash
curl \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"name":"Alice"}' \
    http://127.0.0.1:8080/users
```

---

## PUT

```bash
curl \
    -X PUT \
    -H 'Content-Type: application/json' \
    -d '{"name":"Bob"}' \
    http://127.0.0.1:8080/users/42
```

---

## PATCH

```bash
curl \
    -X PATCH \
    http://127.0.0.1:8080/users/42
```

---

## DELETE

```bash
curl \
    -i \
    -X DELETE \
    http://127.0.0.1:8080/users/42
```

---

# 15. Автоматическое HTTP framing

Пользователь не должен вручную формировать:

```text
HTTP/1.1 200 OK
Content-Type: ...
Content-Length: ...
Connection: close
```

`Response` и web runtime автоматически формируют HTTP response.

Например:

```python
return Response.json_value(root)
```

превращается в ответ примерно:

```http
HTTP/1.1 200 OK
Server: Ocean/0.1
Content-Type: application/json; charset=utf-8
Content-Length: ...
Connection: close

{"name":"Ocean"}
```

---

# 16. Автоматические ошибки router

Если URL не зарегистрирован:

```text
404 Not Found
```

Если URL существует, но HTTP method не поддерживается:

```text
405 Method Not Allowed
```

Для `HEAD` server может использовать соответствующий `GET` route, но не отправляет body.

---

# 17. Настройка App

## Server header

```python
app.set_server_header("Ocean/0.1")
```

---

## Maximum request body

```python
app.set_max_body_bytes(
    10 * 1024 * 1024
)
```

Текущий runtime имеет ограничение размера request body для защиты от неограниченного чтения входящих данных.

---

## Serve

Локально:

```python
app.serve(
    "127.0.0.1",
    8080
)
```

На всех интерфейсах:

```python
app.serve(
    "0.0.0.0",
    8080
)
```

---

# 18. Архитектура

```text
Ocean application
        │
        ▼
     std/net/web.oc
        │
        ├── App
        ├── Request
        └── Response
        │
        ▼
   web_runtime.c
        │
        ├── HTTP parser
        ├── router
        ├── path params
        ├── query params
        └── response writer
        │
        ▼
      POSIX sockets
```

`Request` и `Response` являются обычными типами Ocean.

Пользовательский handler:

```python
def handler(request: Request) -> Response:
```

не должен работать непосредственно с:

```text
ocean_web_request_t
ocean_web_response_t
```

Эти opaque C handles являются внутренней частью stdlib/FFI.

---

# 19. Ownership

`Request` содержит borrowed handle входящего HTTP request.

Raw request существует только во время выполнения handler.

Не следует сохранять `Request` глобально или использовать его после завершения handler.

`Response` владеет внутренним `ocean_web_response_t` до передачи ответа web runtime.

При возврате `Response` из handler runtime забирает внутренний response handle и отправляет HTTP response.

Для пользователя это означает обычную модель:

```python
def handler(request: Request) -> Response:
    return Response.text("ok")
```

Без ручного освобождения request/response.

---

# 20. `Json` ownership

`Json` является managed объектом stdlib.

Например:

```python
var root: Json = Json.object()
var child: Json = Json.str("value")

root.set("child", child)

return Response.json_value(root)
```

`Response.json_value()` сериализует JSON до передачи его HTTP runtime.

---

# 21. Текущие ограничения

Текущая версия `std/net` ориентирована на простой HTTP/1.1 backend.

На данный момент следует учитывать следующие ограничения:

- server работает синхронно;
- один request обрабатывается за раз;
- HTTP keep-alive пока не является основной моделью server runtime;
- server закрывает соединение после response;
- HTTPS/TLS server пока отсутствует;
- HTTP client v1 поддерживает plain `http://`;
- binary body с `NUL` не является полноценным `bytes` API;
- request body ориентирован на `Content-Length`;
- chunked request body пока не является частью публичного API;
- middleware пока отсутствует;
- automatic schema validation пока отсутствует;
- dependency injection пока отсутствует;
- OpenAPI generation пока отсутствует;
- WebSocket пока отсутствует;
- streaming response пока отсутствует;
- multipart/form-data пока отсутствует;
- production worker/thread pool пока отсутствует.

---

# 22. Рекомендуемая структура backend проекта

```text
app/
├── main.oc
├── routes/
│   ├── users.oc
│   └── health.oc
├── services/
├── models/
└── repositories/
```

Пример `main.oc`:

```python
import <std/net/web.oc>


def health(request: Request) -> Response:
    return Response.json(
        "{\"status\":\"ok\"}"
    )


def main() -> int:
    var app: App = App.create()

    app.get("/health", health)

    app.serve(
        "0.0.0.0",
        8080
    )

    return 0
```

---

# 23. План развития `std/net`

Следующие логичные этапы развития server runtime:

```text
1. thread pool / worker pool
2. HTTP keep-alive
3. graceful shutdown
4. middleware
5. CORS
6. structured errors
7. JSON/schema validation
8. multipart/form-data
9. file responses
10. streaming responses
11. TLS/HTTPS
12. WebSocket
13. OpenAPI
14. route groups
15. request context
```

Цель API — сохранить простую модель:

```python
def get_user(request: Request) -> Response:
    ...
```

и при этом постепенно расширять runtime без изменения обычного backend-кода пользователя.
