#ifndef OCEAN_STD_LOGGING_LOGGING_RUNTIME_H
#define OCEAN_STD_LOGGING_LOGGING_RUNTIME_H

#include <stdbool.h>

enum {
    OCEAN_LOG_DEBUG = 10,
    OCEAN_LOG_INFO = 20,
    OCEAN_LOG_WARNING = 30,
    OCEAN_LOG_ERROR = 40,
    OCEAN_LOG_CRITICAL = 50,
    OCEAN_LOG_OFF = 100
};

void ocean_logging_set_level(int level);
int ocean_logging_get_level(void);
bool ocean_logging_enabled(int level);

void ocean_logging_set_timestamps(bool enabled);
bool ocean_logging_get_timestamps(void);

void ocean_logging_to_stderr(void);
void ocean_logging_to_stdout(void);
void ocean_logging_to_file(const char *path, bool append);

void ocean_logging_write(int level, const char *logger_name, const char *message);
void ocean_logging_flush(void);
void ocean_logging_shutdown(void);

#endif
