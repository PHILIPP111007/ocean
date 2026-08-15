#ifndef OCEAN_STD_JSON_RUNTIME_H
#define OCEAN_STD_JSON_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct ocean_json_value *ocean_json_handle_t;

typedef enum ocean_json_kind {
    OCEAN_JSON_NULL = 0,
    OCEAN_JSON_BOOL = 1,
    OCEAN_JSON_NUMBER = 2,
    OCEAN_JSON_STRING = 3,
    OCEAN_JSON_ARRAY = 4,
    OCEAN_JSON_OBJECT = 5
} ocean_json_kind_t;

/* Parsing / serialization */
ocean_json_handle_t ocean_json_parse(const char *text);
char *ocean_json_stringify(ocean_json_handle_t value, int indent);
void ocean_json_release(ocean_json_handle_t value);

/* Constructors */
ocean_json_handle_t ocean_json_new_null(void);
ocean_json_handle_t ocean_json_new_bool(bool value);
ocean_json_handle_t ocean_json_new_int(int64_t value);
ocean_json_handle_t ocean_json_new_number(double value);
ocean_json_handle_t ocean_json_new_string(const char *value);
ocean_json_handle_t ocean_json_new_array(void);
ocean_json_handle_t ocean_json_new_object(void);

/* Type inspection */
int ocean_json_kind(ocean_json_handle_t value);
bool ocean_json_is_null(ocean_json_handle_t value);
bool ocean_json_is_bool(ocean_json_handle_t value);
bool ocean_json_is_number(ocean_json_handle_t value);
bool ocean_json_is_string(ocean_json_handle_t value);
bool ocean_json_is_array(ocean_json_handle_t value);
bool ocean_json_is_object(ocean_json_handle_t value);
size_t ocean_json_size(ocean_json_handle_t value);

/* Scalar conversion. Strings are returned as newly allocated Ocean strings. */
bool ocean_json_as_bool(ocean_json_handle_t value);
int64_t ocean_json_as_int(ocean_json_handle_t value);
double ocean_json_as_float(ocean_json_handle_t value);
char *ocean_json_as_string_copy(ocean_json_handle_t value);

/* Object operations. Returned values are independent deep clones. */
bool ocean_json_object_has(ocean_json_handle_t object, const char *key);
ocean_json_handle_t ocean_json_object_get(ocean_json_handle_t object, const char *key);
void ocean_json_object_set(
    ocean_json_handle_t object,
    const char *key,
    ocean_json_handle_t value
);
bool ocean_json_object_remove(ocean_json_handle_t object, const char *key);
char *ocean_json_object_key_at(ocean_json_handle_t object, size_t index);
ocean_json_handle_t ocean_json_object_value_at(
    ocean_json_handle_t object,
    size_t index
);

/* Array operations. Returned values are independent deep clones. */
ocean_json_handle_t ocean_json_array_get(ocean_json_handle_t array, size_t index);
void ocean_json_array_set(
    ocean_json_handle_t array,
    size_t index,
    ocean_json_handle_t value
);
void ocean_json_array_append(ocean_json_handle_t array, ocean_json_handle_t value);

#endif
