#include "std/io/file_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ocean_file_handle {
    FILE *stream;
    bool binary;
    bool closed;
};

static _Noreturn void ocean_file_fail(const char *message) {
    fprintf(stderr, "Ocean File error: %s\n", message);
    exit(EXIT_FAILURE);
}

static struct ocean_file_handle *ocean_file_require(
    ocean_file_handle_t file
) {
    if (!file || file->closed || !file->stream) {
        ocean_file_fail("operation on a closed file");
    }
    return file;
}

ocean_file_handle_t ocean_file_open(const char *path, const char *mode) {
    if (!path || !mode) ocean_file_fail("path and mode are required");
    FILE *stream = fopen(path, mode);
    if (!stream) {
        char message[512];
        snprintf(message, sizeof(message), "cannot open '%s'", path);
        ocean_file_fail(message);
    }
    ocean_file_handle_t file =
        (ocean_file_handle_t)calloc(1, sizeof(*file));
    if (!file) {
        fclose(stream);
        ocean_file_fail("out of memory allocating File");
    }
    file->stream = stream;
    file->binary = strchr(mode, 'b') != NULL;
    file->closed = false;
    return file;
}

void ocean_file_close(ocean_file_handle_t file) {
    if (!file) return;
    if (!file->closed && file->stream) {
        if (fclose(file->stream) != 0) {
            ocean_file_fail("cannot close file");
        }
        file->stream = NULL;
        file->closed = true;
    }
    free(file);
}

static char *ocean_file_read_buffer(
    ocean_file_handle_t file,
    bool stop_at_newline
) {
    struct ocean_file_handle *checked = ocean_file_require(file);
    size_t length = 0;
    size_t capacity = 256;
    char *result = (char *)malloc(capacity);
    if (!result) ocean_file_fail("out of memory reading file");

    int value;
    while ((value = fgetc(checked->stream)) != EOF) {
        if (length + 1 >= capacity) {
            if (capacity > (size_t)-1 / 2) {
                free(result);
                ocean_file_fail("file content is too large");
            }
            capacity *= 2;
            char *grown = (char *)realloc(result, capacity);
            if (!grown) {
                free(result);
                ocean_file_fail("out of memory growing file buffer");
            }
            result = grown;
        }
        result[length++] = (char)value;
        if (stop_at_newline && value == '\n') break;
    }
    if (ferror(checked->stream)) {
        free(result);
        ocean_file_fail("cannot read file");
    }
    result[length] = '\0';
    return result;
}

char *ocean_file_read(ocean_file_handle_t file) {
    return ocean_file_read_buffer(file, false);
}

char *ocean_file_readline(ocean_file_handle_t file) {
    return ocean_file_read_buffer(file, true);
}

void ocean_file_write(ocean_file_handle_t file, const char *value) {
    struct ocean_file_handle *checked = ocean_file_require(file);
    if (!value) ocean_file_fail("cannot write a null string");
    size_t length = strlen(value);
    if (length && fwrite(value, 1, length, checked->stream) != length) {
        ocean_file_fail("cannot write file");
    }
}

void ocean_file_flush(ocean_file_handle_t file) {
    struct ocean_file_handle *checked = ocean_file_require(file);
    if (fflush(checked->stream) != 0) ocean_file_fail("cannot flush file");
}

bool ocean_file_eof(ocean_file_handle_t file) {
    struct ocean_file_handle *checked = ocean_file_require(file);
    int value = fgetc(checked->stream);
    if (value == EOF) {
        if (ferror(checked->stream)) ocean_file_fail("cannot inspect file end");
        return true;
    }
    if (ungetc(value, checked->stream) == EOF) {
        ocean_file_fail("cannot inspect file end");
    }
    return false;
}

int ocean_file_read_byte(ocean_file_handle_t file) {
    struct ocean_file_handle *checked = ocean_file_require(file);
    int value = fgetc(checked->stream);
    if (value == EOF && ferror(checked->stream)) {
        ocean_file_fail("cannot read byte");
    }
    return value;
}

void ocean_file_write_byte(ocean_file_handle_t file, int value) {
    struct ocean_file_handle *checked = ocean_file_require(file);
    if (value < 0 || value > 255) ocean_file_fail("byte must be in range 0..255");
    if (fputc(value, checked->stream) == EOF) ocean_file_fail("cannot write byte");
}
