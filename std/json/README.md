# Ocean JSON v1

Files to add to the repository:

- `std/json/json.oc`
- `std/json/json_runtime.h`
- `std/json/json_runtime.c`
- `tests/test_json_io.py`
- `examples/json.oc`

Compiler change: add `ocean_json_handle_t` cleanup to the generated class destructor in
`src/codegen/oop.py`:

```python
elif field_type == "ocean_json_handle_t":
    self.add_line(f"ocean_json_release({access});")
```

The public `Json` class is ARC-managed. Its opaque runtime handle owns the JSON tree.
`get()/at()/value_at()` return deep clones, and `set()/append()/set_at()` clone the incoming
value. This makes nested Json values independent of root/container lifetime in v1.
