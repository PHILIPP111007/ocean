# `File` and `BinaryFile`

Import the standard file API with:

```text
import <std/io/file.oc>
```

`open(path, mode)` returns a managed `File`:

```text
var output: File = open("notes.txt", "w")
output.writelines(["first\n", "second\n"])
output.close()

var input: File = open("notes.txt", "r")
var lines: list[str] = input.readlines()
input.close()
```

`File` provides `read()`, `readline()`, `readlines()`, `write(value)`,
`writelines(values)`, `flush()`, `eof()`, and `close()`. Modes are passed to
the C stream backend, for example `"r"`, `"w"`, and `"a"`.

`open_binary(path, mode)` returns a managed `BinaryFile` with
`read_byte()`, `read_bytes(count)`, `write_byte(value)`, `write_bytes(values)`,
`flush()`, `eof()`, and `close()`. Bytes are represented as `int` in safe
Ocean code.

Both objects close their underlying stream when explicitly closed or when
their owning Ocean object is released. File handles are opaque; raw `FILE *`
values do not cross into safe Ocean code.
