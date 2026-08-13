#ifndef OCEAN_STD_IO_FILE_RUNTIME_H
#define OCEAN_STD_IO_FILE_RUNTIME_H

#include <stdbool.h>

typedef struct ocean_file_handle *ocean_file_handle_t;

ocean_file_handle_t ocean_file_open(const char *path, const char *mode);
void ocean_file_close(ocean_file_handle_t file);
char *ocean_file_read(ocean_file_handle_t file);
char *ocean_file_readline(ocean_file_handle_t file);
void ocean_file_write(ocean_file_handle_t file, const char *value);
void ocean_file_flush(ocean_file_handle_t file);
bool ocean_file_eof(ocean_file_handle_t file);
int ocean_file_read_byte(ocean_file_handle_t file);
void ocean_file_write_byte(ocean_file_handle_t file, int value);

#endif
