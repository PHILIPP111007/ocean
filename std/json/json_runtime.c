#include "std/json/json_runtime.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OCEAN_JSON_MAX_DEPTH 512u

struct ocean_json_value {
    ocean_json_kind_t kind;
    union {
        bool boolean;
        char *number_text;
        char *string;
        struct {
            struct ocean_json_value **items;
            size_t size;
            size_t capacity;
        } array;
        struct {
            char **keys;
            struct ocean_json_value **values;
            size_t size;
            size_t capacity;
        } object;
    } as;
};

struct ocean_json_parser {
    const char *start;
    const char *current;
};

struct ocean_json_buffer {
    char *data;
    size_t size;
    size_t capacity;
};

static _Noreturn void ocean_json_fail(const char *message) {
    fprintf(stderr, "Ocean JSON error: %s\n", message);
    exit(EXIT_FAILURE);
}

static _Noreturn void ocean_json_parse_fail(
    const struct ocean_json_parser *parser,
    const char *message
) {
    size_t offset = (size_t)(parser->current - parser->start);
    fprintf(stderr, "Ocean JSON parse error at byte %zu: %s\n", offset, message);
    exit(EXIT_FAILURE);
}

static void *ocean_json_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr && size != 0) ocean_json_fail("out of memory");
    return ptr;
}

static void *ocean_json_calloc(size_t count, size_t size) {
    void *ptr = calloc(count, size);
    if (!ptr && count != 0 && size != 0) ocean_json_fail("out of memory");
    return ptr;
}

static void *ocean_json_realloc(void *ptr, size_t size) {
    void *result = realloc(ptr, size);
    if (!result && size != 0) ocean_json_fail("out of memory");
    return result;
}

static char *ocean_json_strdup(const char *value) {
    if (!value) ocean_json_fail("null string");
    size_t length = strlen(value);
    char *copy = (char *)ocean_json_malloc(length + 1);
    memcpy(copy, value, length + 1);
    return copy;
}

static struct ocean_json_value *ocean_json_alloc(ocean_json_kind_t kind) {
    struct ocean_json_value *value =
        (struct ocean_json_value *)ocean_json_calloc(1, sizeof(*value));
    value->kind = kind;
    return value;
}

static void ocean_json_require(
    ocean_json_handle_t value,
    ocean_json_kind_t expected,
    const char *operation
) {
    if (!value) ocean_json_fail("operation on released Json value");
    if (value->kind != expected) {
        char message[160];
        snprintf(
            message,
            sizeof(message),
            "%s requires JSON kind %d, got %d",
            operation,
            (int)expected,
            (int)value->kind
        );
        ocean_json_fail(message);
    }
}

static void ocean_json_release_value(struct ocean_json_value *value) {
    if (!value) return;
    switch (value->kind) {
        case OCEAN_JSON_NUMBER:
            free(value->as.number_text);
            break;
        case OCEAN_JSON_STRING:
            free(value->as.string);
            break;
        case OCEAN_JSON_ARRAY:
            for (size_t i = 0; i < value->as.array.size; ++i) {
                ocean_json_release_value(value->as.array.items[i]);
            }
            free(value->as.array.items);
            break;
        case OCEAN_JSON_OBJECT:
            for (size_t i = 0; i < value->as.object.size; ++i) {
                free(value->as.object.keys[i]);
                ocean_json_release_value(value->as.object.values[i]);
            }
            free(value->as.object.keys);
            free(value->as.object.values);
            break;
        case OCEAN_JSON_NULL:
        case OCEAN_JSON_BOOL:
            break;
    }
    free(value);
}

void ocean_json_release(ocean_json_handle_t value) {
    ocean_json_release_value(value);
}

static struct ocean_json_value *ocean_json_clone_value(
    const struct ocean_json_value *value
);

static void ocean_json_array_reserve(
    struct ocean_json_value *array,
    size_t required
) {
    if (required <= array->as.array.capacity) return;
    size_t capacity = array->as.array.capacity ? array->as.array.capacity : 4;
    while (capacity < required) {
        if (capacity > (size_t)-1 / 2) ocean_json_fail("array is too large");
        capacity *= 2;
    }
    array->as.array.items = (struct ocean_json_value **)ocean_json_realloc(
        array->as.array.items,
        capacity * sizeof(*array->as.array.items)
    );
    array->as.array.capacity = capacity;
}

static void ocean_json_object_reserve(
    struct ocean_json_value *object,
    size_t required
) {
    if (required <= object->as.object.capacity) return;
    size_t capacity = object->as.object.capacity ? object->as.object.capacity : 4;
    while (capacity < required) {
        if (capacity > (size_t)-1 / 2) ocean_json_fail("object is too large");
        capacity *= 2;
    }
    object->as.object.keys = (char **)ocean_json_realloc(
        object->as.object.keys,
        capacity * sizeof(*object->as.object.keys)
    );
    object->as.object.values = (struct ocean_json_value **)ocean_json_realloc(
        object->as.object.values,
        capacity * sizeof(*object->as.object.values)
    );
    object->as.object.capacity = capacity;
}

static size_t ocean_json_object_find(
    const struct ocean_json_value *object,
    const char *key
) {
    for (size_t i = 0; i < object->as.object.size; ++i) {
        if (strcmp(object->as.object.keys[i], key) == 0) return i;
    }
    return (size_t)-1;
}

static void ocean_json_object_put_owned(
    struct ocean_json_value *object,
    char *key,
    struct ocean_json_value *value
) {
    size_t existing = ocean_json_object_find(object, key);
    if (existing != (size_t)-1) {
        free(key);
        ocean_json_release_value(object->as.object.values[existing]);
        object->as.object.values[existing] = value;
        return;
    }
    ocean_json_object_reserve(object, object->as.object.size + 1);
    size_t index = object->as.object.size++;
    object->as.object.keys[index] = key;
    object->as.object.values[index] = value;
}

static struct ocean_json_value *ocean_json_clone_value(
    const struct ocean_json_value *value
) {
    if (!value) return NULL;
    struct ocean_json_value *copy = ocean_json_alloc(value->kind);
    switch (value->kind) {
        case OCEAN_JSON_NULL:
            break;
        case OCEAN_JSON_BOOL:
            copy->as.boolean = value->as.boolean;
            break;
        case OCEAN_JSON_NUMBER:
            copy->as.number_text = ocean_json_strdup(value->as.number_text);
            break;
        case OCEAN_JSON_STRING:
            copy->as.string = ocean_json_strdup(value->as.string);
            break;
        case OCEAN_JSON_ARRAY:
            ocean_json_array_reserve(copy, value->as.array.size);
            for (size_t i = 0; i < value->as.array.size; ++i) {
                copy->as.array.items[i] =
                    ocean_json_clone_value(value->as.array.items[i]);
            }
            copy->as.array.size = value->as.array.size;
            break;
        case OCEAN_JSON_OBJECT:
            ocean_json_object_reserve(copy, value->as.object.size);
            for (size_t i = 0; i < value->as.object.size; ++i) {
                copy->as.object.keys[i] =
                    ocean_json_strdup(value->as.object.keys[i]);
                copy->as.object.values[i] =
                    ocean_json_clone_value(value->as.object.values[i]);
            }
            copy->as.object.size = value->as.object.size;
            break;
    }
    return copy;
}

/* ---------- String builder ---------- */

static void ocean_json_buffer_reserve(
    struct ocean_json_buffer *buffer,
    size_t extra
) {
    if (extra > (size_t)-1 - buffer->size - 1) {
        ocean_json_fail("serialized JSON is too large");
    }
    size_t required = buffer->size + extra + 1;
    if (required <= buffer->capacity) return;
    size_t capacity = buffer->capacity ? buffer->capacity : 128;
    while (capacity < required) {
        if (capacity > (size_t)-1 / 2) {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    buffer->data = (char *)ocean_json_realloc(buffer->data, capacity);
    buffer->capacity = capacity;
}

static void ocean_json_buffer_append_char(
    struct ocean_json_buffer *buffer,
    char value
) {
    ocean_json_buffer_reserve(buffer, 1);
    buffer->data[buffer->size++] = value;
    buffer->data[buffer->size] = '\0';
}

static void ocean_json_buffer_append_bytes(
    struct ocean_json_buffer *buffer,
    const char *data,
    size_t length
) {
    ocean_json_buffer_reserve(buffer, length);
    memcpy(buffer->data + buffer->size, data, length);
    buffer->size += length;
    buffer->data[buffer->size] = '\0';
}

static void ocean_json_buffer_append_cstr(
    struct ocean_json_buffer *buffer,
    const char *value
) {
    ocean_json_buffer_append_bytes(buffer, value, strlen(value));
}

static void ocean_json_append_utf8(
    struct ocean_json_buffer *buffer,
    uint32_t codepoint
) {
    if (codepoint <= 0x7F) {
        ocean_json_buffer_append_char(buffer, (char)codepoint);
    } else if (codepoint <= 0x7FF) {
        ocean_json_buffer_append_char(buffer, (char)(0xC0u | (codepoint >> 6)));
        ocean_json_buffer_append_char(buffer, (char)(0x80u | (codepoint & 0x3Fu)));
    } else if (codepoint <= 0xFFFF) {
        ocean_json_buffer_append_char(buffer, (char)(0xE0u | (codepoint >> 12)));
        ocean_json_buffer_append_char(
            buffer,
            (char)(0x80u | ((codepoint >> 6) & 0x3Fu))
        );
        ocean_json_buffer_append_char(buffer, (char)(0x80u | (codepoint & 0x3Fu)));
    } else if (codepoint <= 0x10FFFF) {
        ocean_json_buffer_append_char(buffer, (char)(0xF0u | (codepoint >> 18)));
        ocean_json_buffer_append_char(
            buffer,
            (char)(0x80u | ((codepoint >> 12) & 0x3Fu))
        );
        ocean_json_buffer_append_char(
            buffer,
            (char)(0x80u | ((codepoint >> 6) & 0x3Fu))
        );
        ocean_json_buffer_append_char(buffer, (char)(0x80u | (codepoint & 0x3Fu)));
    } else {
        ocean_json_fail("invalid Unicode codepoint");
    }
}

/* ---------- Parser ---------- */

static void ocean_json_skip_ws(struct ocean_json_parser *parser) {
    while (
        *parser->current == ' ' || *parser->current == '\t' ||
        *parser->current == '\n' || *parser->current == '\r'
    ) {
        parser->current++;
    }
}

static int ocean_json_hex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static uint32_t ocean_json_parse_hex4(struct ocean_json_parser *parser) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        int digit = ocean_json_hex(*parser->current);
        if (digit < 0) ocean_json_parse_fail(parser, "invalid \\u escape");
        result = (result << 4) | (uint32_t)digit;
        parser->current++;
    }
    return result;
}

static char *ocean_json_parse_string_raw(struct ocean_json_parser *parser) {
    if (*parser->current != '"') {
        ocean_json_parse_fail(parser, "expected string");
    }
    parser->current++;
    struct ocean_json_buffer buffer = {0};

    while (*parser->current && *parser->current != '"') {
        unsigned char ch = (unsigned char)*parser->current++;
        if (ch < 0x20u) {
            free(buffer.data);
            ocean_json_parse_fail(parser, "control character in string");
        }
        if (ch != '\\') {
            ocean_json_buffer_append_char(&buffer, (char)ch);
            continue;
        }

        char escape = *parser->current++;
        switch (escape) {
            case '"': ocean_json_buffer_append_char(&buffer, '"'); break;
            case '\\': ocean_json_buffer_append_char(&buffer, '\\'); break;
            case '/': ocean_json_buffer_append_char(&buffer, '/'); break;
            case 'b': ocean_json_buffer_append_char(&buffer, '\b'); break;
            case 'f': ocean_json_buffer_append_char(&buffer, '\f'); break;
            case 'n': ocean_json_buffer_append_char(&buffer, '\n'); break;
            case 'r': ocean_json_buffer_append_char(&buffer, '\r'); break;
            case 't': ocean_json_buffer_append_char(&buffer, '\t'); break;
            case 'u': {
                uint32_t codepoint = ocean_json_parse_hex4(parser);
                if (codepoint >= 0xD800u && codepoint <= 0xDBFFu) {
                    if (parser->current[0] != '\\' || parser->current[1] != 'u') {
                        free(buffer.data);
                        ocean_json_parse_fail(parser, "missing low surrogate");
                    }
                    parser->current += 2;
                    uint32_t low = ocean_json_parse_hex4(parser);
                    if (low < 0xDC00u || low > 0xDFFFu) {
                        free(buffer.data);
                        ocean_json_parse_fail(parser, "invalid low surrogate");
                    }
                    codepoint = 0x10000u +
                        ((codepoint - 0xD800u) << 10) + (low - 0xDC00u);
                } else if (codepoint >= 0xDC00u && codepoint <= 0xDFFFu) {
                    free(buffer.data);
                    ocean_json_parse_fail(parser, "isolated low surrogate");
                }
                ocean_json_append_utf8(&buffer, codepoint);
                break;
            }
            default:
                free(buffer.data);
                ocean_json_parse_fail(parser, "invalid string escape");
        }
    }

    if (*parser->current != '"') {
        free(buffer.data);
        ocean_json_parse_fail(parser, "unterminated string");
    }
    parser->current++;

    if (!buffer.data) return ocean_json_strdup("");
    return buffer.data;
}

static struct ocean_json_value *ocean_json_parse_value(
    struct ocean_json_parser *parser,
    unsigned depth
);

static struct ocean_json_value *ocean_json_parse_array(
    struct ocean_json_parser *parser,
    unsigned depth
) {
    struct ocean_json_value *array = ocean_json_alloc(OCEAN_JSON_ARRAY);
    parser->current++;
    ocean_json_skip_ws(parser);
    if (*parser->current == ']') {
        parser->current++;
        return array;
    }

    for (;;) {
        struct ocean_json_value *item = ocean_json_parse_value(parser, depth + 1);
        ocean_json_array_reserve(array, array->as.array.size + 1);
        array->as.array.items[array->as.array.size++] = item;
        ocean_json_skip_ws(parser);
        if (*parser->current == ']') {
            parser->current++;
            return array;
        }
        if (*parser->current != ',') {
            ocean_json_release_value(array);
            ocean_json_parse_fail(parser, "expected ',' or ']' in array");
        }
        parser->current++;
        ocean_json_skip_ws(parser);
    }
}

static struct ocean_json_value *ocean_json_parse_object(
    struct ocean_json_parser *parser,
    unsigned depth
) {
    struct ocean_json_value *object = ocean_json_alloc(OCEAN_JSON_OBJECT);
    parser->current++;
    ocean_json_skip_ws(parser);
    if (*parser->current == '}') {
        parser->current++;
        return object;
    }

    for (;;) {
        if (*parser->current != '"') {
            ocean_json_release_value(object);
            ocean_json_parse_fail(parser, "object key must be a string");
        }
        char *key = ocean_json_parse_string_raw(parser);
        ocean_json_skip_ws(parser);
        if (*parser->current != ':') {
            free(key);
            ocean_json_release_value(object);
            ocean_json_parse_fail(parser, "expected ':' after object key");
        }
        parser->current++;
        ocean_json_skip_ws(parser);
        struct ocean_json_value *item = ocean_json_parse_value(parser, depth + 1);
        ocean_json_object_put_owned(object, key, item);
        ocean_json_skip_ws(parser);
        if (*parser->current == '}') {
            parser->current++;
            return object;
        }
        if (*parser->current != ',') {
            ocean_json_release_value(object);
            ocean_json_parse_fail(parser, "expected ',' or '}' in object");
        }
        parser->current++;
        ocean_json_skip_ws(parser);
    }
}

static struct ocean_json_value *ocean_json_parse_number(
    struct ocean_json_parser *parser
) {
    const char *start = parser->current;
    if (*parser->current == '-') parser->current++;

    if (*parser->current == '0') {
        parser->current++;
        if (*parser->current >= '0' && *parser->current <= '9') {
            ocean_json_parse_fail(parser, "leading zero in number");
        }
    } else if (*parser->current >= '1' && *parser->current <= '9') {
        do {
            parser->current++;
        } while (*parser->current >= '0' && *parser->current <= '9');
    } else {
        ocean_json_parse_fail(parser, "invalid number");
    }

    if (*parser->current == '.') {
        parser->current++;
        if (*parser->current < '0' || *parser->current > '9') {
            ocean_json_parse_fail(parser, "fraction requires digits");
        }
        do {
            parser->current++;
        } while (*parser->current >= '0' && *parser->current <= '9');
    }

    if (*parser->current == 'e' || *parser->current == 'E') {
        parser->current++;
        if (*parser->current == '+' || *parser->current == '-') parser->current++;
        if (*parser->current < '0' || *parser->current > '9') {
            ocean_json_parse_fail(parser, "exponent requires digits");
        }
        do {
            parser->current++;
        } while (*parser->current >= '0' && *parser->current <= '9');
    }

    size_t length = (size_t)(parser->current - start);
    char *text = (char *)ocean_json_malloc(length + 1);
    memcpy(text, start, length);
    text[length] = '\0';

    errno = 0;
    char *end = NULL;
    double number = strtod(text, &end);
    if (errno == ERANGE || !end || *end != '\0' || !isfinite(number)) {
        free(text);
        ocean_json_parse_fail(parser, "number is outside supported range");
    }

    struct ocean_json_value *value = ocean_json_alloc(OCEAN_JSON_NUMBER);
    value->as.number_text = text;
    return value;
}

static bool ocean_json_match(struct ocean_json_parser *parser, const char *word) {
    size_t length = strlen(word);
    if (strncmp(parser->current, word, length) != 0) return false;
    parser->current += length;
    return true;
}

static struct ocean_json_value *ocean_json_parse_value(
    struct ocean_json_parser *parser,
    unsigned depth
) {
    if (depth > OCEAN_JSON_MAX_DEPTH) {
        ocean_json_parse_fail(parser, "nesting depth limit exceeded");
    }
    ocean_json_skip_ws(parser);
    char ch = *parser->current;
    if (ch == '{') return ocean_json_parse_object(parser, depth);
    if (ch == '[') return ocean_json_parse_array(parser, depth);
    if (ch == '"') {
        struct ocean_json_value *value = ocean_json_alloc(OCEAN_JSON_STRING);
        value->as.string = ocean_json_parse_string_raw(parser);
        return value;
    }
    if (ch == '-' || (ch >= '0' && ch <= '9')) {
        return ocean_json_parse_number(parser);
    }
    if (ocean_json_match(parser, "true")) {
        struct ocean_json_value *value = ocean_json_alloc(OCEAN_JSON_BOOL);
        value->as.boolean = true;
        return value;
    }
    if (ocean_json_match(parser, "false")) {
        struct ocean_json_value *value = ocean_json_alloc(OCEAN_JSON_BOOL);
        value->as.boolean = false;
        return value;
    }
    if (ocean_json_match(parser, "null")) {
        return ocean_json_alloc(OCEAN_JSON_NULL);
    }
    ocean_json_parse_fail(parser, "expected JSON value");
}

ocean_json_handle_t ocean_json_parse(const char *text) {
    if (!text) ocean_json_fail("cannot parse a null string");
    struct ocean_json_parser parser = {text, text};
    ocean_json_skip_ws(&parser);
    struct ocean_json_value *value = ocean_json_parse_value(&parser, 0);
    ocean_json_skip_ws(&parser);
    if (*parser.current != '\0') {
        ocean_json_release_value(value);
        ocean_json_parse_fail(&parser, "trailing characters after JSON value");
    }
    return value;
}

/* ---------- Constructors ---------- */

ocean_json_handle_t ocean_json_new_null(void) {
    return ocean_json_alloc(OCEAN_JSON_NULL);
}

ocean_json_handle_t ocean_json_new_bool(bool value) {
    struct ocean_json_value *result = ocean_json_alloc(OCEAN_JSON_BOOL);
    result->as.boolean = value;
    return result;
}

ocean_json_handle_t ocean_json_new_int(int64_t value) {
    char text[64];
    snprintf(text, sizeof(text), "%" PRId64, value);
    struct ocean_json_value *result = ocean_json_alloc(OCEAN_JSON_NUMBER);
    result->as.number_text = ocean_json_strdup(text);
    return result;
}

ocean_json_handle_t ocean_json_new_number(double value) {
    if (!isfinite(value)) ocean_json_fail("JSON number must be finite");
    char text[64];
    snprintf(text, sizeof(text), "%.17g", value);
    struct ocean_json_value *result = ocean_json_alloc(OCEAN_JSON_NUMBER);
    result->as.number_text = ocean_json_strdup(text);
    return result;
}

ocean_json_handle_t ocean_json_new_string(const char *value) {
    struct ocean_json_value *result = ocean_json_alloc(OCEAN_JSON_STRING);
    result->as.string = ocean_json_strdup(value);
    return result;
}

ocean_json_handle_t ocean_json_new_array(void) {
    return ocean_json_alloc(OCEAN_JSON_ARRAY);
}

ocean_json_handle_t ocean_json_new_object(void) {
    return ocean_json_alloc(OCEAN_JSON_OBJECT);
}

/* ---------- Inspection / scalar access ---------- */

int ocean_json_kind(ocean_json_handle_t value) {
    if (!value) ocean_json_fail("kind() on released Json value");
    return (int)value->kind;
}

bool ocean_json_is_null(ocean_json_handle_t value) {
    return value && value->kind == OCEAN_JSON_NULL;
}

bool ocean_json_is_bool(ocean_json_handle_t value) {
    return value && value->kind == OCEAN_JSON_BOOL;
}

bool ocean_json_is_number(ocean_json_handle_t value) {
    return value && value->kind == OCEAN_JSON_NUMBER;
}

bool ocean_json_is_string(ocean_json_handle_t value) {
    return value && value->kind == OCEAN_JSON_STRING;
}

bool ocean_json_is_array(ocean_json_handle_t value) {
    return value && value->kind == OCEAN_JSON_ARRAY;
}

bool ocean_json_is_object(ocean_json_handle_t value) {
    return value && value->kind == OCEAN_JSON_OBJECT;
}

size_t ocean_json_size(ocean_json_handle_t value) {
    if (!value) ocean_json_fail("size() on released Json value");
    if (value->kind == OCEAN_JSON_ARRAY) return value->as.array.size;
    if (value->kind == OCEAN_JSON_OBJECT) return value->as.object.size;
    ocean_json_fail("size() requires a JSON array or object");
}

bool ocean_json_as_bool(ocean_json_handle_t value) {
    ocean_json_require(value, OCEAN_JSON_BOOL, "as_bool()");
    return value->as.boolean;
}

int64_t ocean_json_as_int(ocean_json_handle_t value) {
    ocean_json_require(value, OCEAN_JSON_NUMBER, "as_int()");
    const char *text = value->as.number_text;
    if (strchr(text, '.') || strchr(text, 'e') || strchr(text, 'E')) {
        ocean_json_fail("as_int() requires an integer JSON number");
    }
    errno = 0;
    char *end = NULL;
    intmax_t parsed = strtoimax(text, &end, 10);
    if (errno == ERANGE || !end || *end != '\0' ||
        parsed < INT64_MIN || parsed > INT64_MAX) {
        ocean_json_fail("JSON integer does not fit int64");
    }
    return (int64_t)parsed;
}

double ocean_json_as_float(ocean_json_handle_t value) {
    ocean_json_require(value, OCEAN_JSON_NUMBER, "as_float()");
    errno = 0;
    char *end = NULL;
    double parsed = strtod(value->as.number_text, &end);
    if (errno == ERANGE || !end || *end != '\0' || !isfinite(parsed)) {
        ocean_json_fail("JSON number cannot be represented as float64");
    }
    return parsed;
}

char *ocean_json_as_string_copy(ocean_json_handle_t value) {
    ocean_json_require(value, OCEAN_JSON_STRING, "as_str()");
    return ocean_json_strdup(value->as.string);
}

/* ---------- Object operations ---------- */

bool ocean_json_object_has(ocean_json_handle_t object, const char *key) {
    ocean_json_require(object, OCEAN_JSON_OBJECT, "has()");
    if (!key) ocean_json_fail("object key cannot be null");
    return ocean_json_object_find(object, key) != (size_t)-1;
}

ocean_json_handle_t ocean_json_object_get(
    ocean_json_handle_t object,
    const char *key
) {
    ocean_json_require(object, OCEAN_JSON_OBJECT, "get()");
    if (!key) ocean_json_fail("object key cannot be null");
    size_t index = ocean_json_object_find(object, key);
    if (index == (size_t)-1) return ocean_json_new_null();
    return ocean_json_clone_value(object->as.object.values[index]);
}

void ocean_json_object_set(
    ocean_json_handle_t object,
    const char *key,
    ocean_json_handle_t value
) {
    ocean_json_require(object, OCEAN_JSON_OBJECT, "set()");
    if (!key) ocean_json_fail("object key cannot be null");
    if (!value) ocean_json_fail("cannot store a released Json value");
    ocean_json_object_put_owned(
        object,
        ocean_json_strdup(key),
        ocean_json_clone_value(value)
    );
}

bool ocean_json_object_remove(ocean_json_handle_t object, const char *key) {
    ocean_json_require(object, OCEAN_JSON_OBJECT, "remove()");
    if (!key) ocean_json_fail("object key cannot be null");
    size_t index = ocean_json_object_find(object, key);
    if (index == (size_t)-1) return false;
    free(object->as.object.keys[index]);
    ocean_json_release_value(object->as.object.values[index]);
    for (size_t i = index + 1; i < object->as.object.size; ++i) {
        object->as.object.keys[i - 1] = object->as.object.keys[i];
        object->as.object.values[i - 1] = object->as.object.values[i];
    }
    object->as.object.size--;
    return true;
}

char *ocean_json_object_key_at(ocean_json_handle_t object, size_t index) {
    ocean_json_require(object, OCEAN_JSON_OBJECT, "key_at()");
    if (index >= object->as.object.size) ocean_json_fail("object index out of bounds");
    return ocean_json_strdup(object->as.object.keys[index]);
}

ocean_json_handle_t ocean_json_object_value_at(
    ocean_json_handle_t object,
    size_t index
) {
    ocean_json_require(object, OCEAN_JSON_OBJECT, "value_at()");
    if (index >= object->as.object.size) ocean_json_fail("object index out of bounds");
    return ocean_json_clone_value(object->as.object.values[index]);
}

/* ---------- Array operations ---------- */

ocean_json_handle_t ocean_json_array_get(
    ocean_json_handle_t array,
    size_t index
) {
    ocean_json_require(array, OCEAN_JSON_ARRAY, "at()");
    if (index >= array->as.array.size) ocean_json_fail("array index out of bounds");
    return ocean_json_clone_value(array->as.array.items[index]);
}

void ocean_json_array_set(
    ocean_json_handle_t array,
    size_t index,
    ocean_json_handle_t value
) {
    ocean_json_require(array, OCEAN_JSON_ARRAY, "set_at()");
    if (index >= array->as.array.size) ocean_json_fail("array index out of bounds");
    if (!value) ocean_json_fail("cannot store a released Json value");
    struct ocean_json_value *copy = ocean_json_clone_value(value);
    ocean_json_release_value(array->as.array.items[index]);
    array->as.array.items[index] = copy;
}

void ocean_json_array_append(
    ocean_json_handle_t array,
    ocean_json_handle_t value
) {
    ocean_json_require(array, OCEAN_JSON_ARRAY, "append()");
    if (!value) ocean_json_fail("cannot append a released Json value");
    ocean_json_array_reserve(array, array->as.array.size + 1);
    array->as.array.items[array->as.array.size++] = ocean_json_clone_value(value);
}

/* ---------- Serializer ---------- */

static void ocean_json_serialize_string(
    struct ocean_json_buffer *buffer,
    const char *value
) {
    static const char hex[] = "0123456789abcdef";
    ocean_json_buffer_append_char(buffer, '"');
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        unsigned char ch = *p;
        switch (ch) {
            case '"': ocean_json_buffer_append_cstr(buffer, "\\\""); break;
            case '\\': ocean_json_buffer_append_cstr(buffer, "\\\\"); break;
            case '\b': ocean_json_buffer_append_cstr(buffer, "\\b"); break;
            case '\f': ocean_json_buffer_append_cstr(buffer, "\\f"); break;
            case '\n': ocean_json_buffer_append_cstr(buffer, "\\n"); break;
            case '\r': ocean_json_buffer_append_cstr(buffer, "\\r"); break;
            case '\t': ocean_json_buffer_append_cstr(buffer, "\\t"); break;
            default:
                if (ch < 0x20u) {
                    char escaped[7] = {'\\', 'u', '0', '0', hex[ch >> 4], hex[ch & 0x0F], '\0'};
                    ocean_json_buffer_append_cstr(buffer, escaped);
                } else {
                    ocean_json_buffer_append_char(buffer, (char)ch);
                }
                break;
        }
    }
    ocean_json_buffer_append_char(buffer, '"');
}

static void ocean_json_serialize_indent(
    struct ocean_json_buffer *buffer,
    int indent,
    unsigned depth
) {
    if (indent <= 0) return;
    ocean_json_buffer_append_char(buffer, '\n');
    size_t spaces = (size_t)indent * depth;
    for (size_t i = 0; i < spaces; ++i) ocean_json_buffer_append_char(buffer, ' ');
}

static void ocean_json_serialize_value(
    struct ocean_json_buffer *buffer,
    const struct ocean_json_value *value,
    int indent,
    unsigned depth
) {
    if (depth > OCEAN_JSON_MAX_DEPTH) ocean_json_fail("serialization depth limit exceeded");
    switch (value->kind) {
        case OCEAN_JSON_NULL:
            ocean_json_buffer_append_cstr(buffer, "null");
            break;
        case OCEAN_JSON_BOOL:
            ocean_json_buffer_append_cstr(buffer, value->as.boolean ? "true" : "false");
            break;
        case OCEAN_JSON_NUMBER:
            ocean_json_buffer_append_cstr(buffer, value->as.number_text);
            break;
        case OCEAN_JSON_STRING:
            ocean_json_serialize_string(buffer, value->as.string);
            break;
        case OCEAN_JSON_ARRAY:
            ocean_json_buffer_append_char(buffer, '[');
            for (size_t i = 0; i < value->as.array.size; ++i) {
                if (i) ocean_json_buffer_append_char(buffer, ',');
                if (indent > 0) ocean_json_serialize_indent(buffer, indent, depth + 1);
                ocean_json_serialize_value(buffer, value->as.array.items[i], indent, depth + 1);
            }
            if (indent > 0 && value->as.array.size) {
                ocean_json_serialize_indent(buffer, indent, depth);
            }
            ocean_json_buffer_append_char(buffer, ']');
            break;
        case OCEAN_JSON_OBJECT:
            ocean_json_buffer_append_char(buffer, '{');
            for (size_t i = 0; i < value->as.object.size; ++i) {
                if (i) ocean_json_buffer_append_char(buffer, ',');
                if (indent > 0) ocean_json_serialize_indent(buffer, indent, depth + 1);
                ocean_json_serialize_string(buffer, value->as.object.keys[i]);
                ocean_json_buffer_append_char(buffer, ':');
                if (indent > 0) ocean_json_buffer_append_char(buffer, ' ');
                ocean_json_serialize_value(buffer, value->as.object.values[i], indent, depth + 1);
            }
            if (indent > 0 && value->as.object.size) {
                ocean_json_serialize_indent(buffer, indent, depth);
            }
            ocean_json_buffer_append_char(buffer, '}');
            break;
    }
}

char *ocean_json_stringify(ocean_json_handle_t value, int indent) {
    if (!value) ocean_json_fail("cannot stringify a released Json value");
    if (indent < 0 || indent > 32) ocean_json_fail("indent must be in range 0..32");
    struct ocean_json_buffer buffer = {0};
    ocean_json_serialize_value(&buffer, value, indent, 0);
    if (!buffer.data) return ocean_json_strdup("");
    return buffer.data;
}
