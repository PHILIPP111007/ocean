#define _POSIX_C_SOURCE 200809L

#include "logging_runtime.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static atomic_int ocean_logging_global_level = ATOMIC_VAR_INIT(OCEAN_LOG_INFO);
static atomic_bool ocean_logging_timestamps = ATOMIC_VAR_INIT(true);
static atomic_bool ocean_logging_colors = ATOMIC_VAR_INIT(true);

static atomic_flag ocean_logging_lock_flag = ATOMIC_FLAG_INIT;
static FILE *ocean_logging_stream = NULL;
static bool ocean_logging_stream_owned = false;

static void ocean_logging_lock(void) {
    while (
        atomic_flag_test_and_set_explicit(
            &ocean_logging_lock_flag,
            memory_order_acquire
        )
    ) {
        /* Logging is a short critical section. */
    }
}

static void ocean_logging_unlock(void) {
    atomic_flag_clear_explicit(
        &ocean_logging_lock_flag,
        memory_order_release
    );
}

static FILE *ocean_logging_current_stream(void) {
    return ocean_logging_stream ? ocean_logging_stream : stderr;
}

static const char *ocean_logging_level_name(int level) {
    if (level >= OCEAN_LOG_CRITICAL) return "CRITICAL";
    if (level >= OCEAN_LOG_ERROR) return "ERROR";
    if (level >= OCEAN_LOG_WARNING) return "WARNING";
    if (level >= OCEAN_LOG_INFO) return "INFO";
    return "DEBUG";
}

static const char *ocean_logging_level_color(int level) {
    if (level >= OCEAN_LOG_CRITICAL) return "\x1b[1;35m";
    if (level >= OCEAN_LOG_ERROR) return "\x1b[1;31m";
    if (level >= OCEAN_LOG_WARNING) return "\x1b[1;33m";
    if (level >= OCEAN_LOG_INFO) return "\x1b[1;32m";
    return "\x1b[1;36m";
}

static const char *ocean_logging_color_reset(void) {
    return "\x1b[0m";
}

static void ocean_logging_close_owned_stream_locked(void) {
    if (ocean_logging_stream_owned && ocean_logging_stream) {
        fflush(ocean_logging_stream);
        fclose(ocean_logging_stream);
    }

    ocean_logging_stream = NULL;
    ocean_logging_stream_owned = false;
}

void ocean_logging_set_level(int level) {
    atomic_store_explicit(
        &ocean_logging_global_level,
        level,
        memory_order_release
    );
}

int ocean_logging_get_level(void) {
    return atomic_load_explicit(
        &ocean_logging_global_level,
        memory_order_acquire
    );
}

bool ocean_logging_enabled(int level) {
    return level >= ocean_logging_get_level();
}

void ocean_logging_set_timestamps(bool enabled) {
    atomic_store_explicit(
        &ocean_logging_timestamps,
        enabled,
        memory_order_release
    );
}

bool ocean_logging_get_timestamps(void) {
    return atomic_load_explicit(
        &ocean_logging_timestamps,
        memory_order_acquire
    );
}

void ocean_logging_set_colors(bool enabled) {
    atomic_store_explicit(
        &ocean_logging_colors,
        enabled,
        memory_order_release
    );
}

bool ocean_logging_get_colors(void) {
    return atomic_load_explicit(
        &ocean_logging_colors,
        memory_order_acquire
    );
}

void ocean_logging_to_stderr(void) {
    ocean_logging_lock();
    ocean_logging_close_owned_stream_locked();
    ocean_logging_stream = stderr;
    ocean_logging_stream_owned = false;
    ocean_logging_unlock();
}

void ocean_logging_to_stdout(void) {
    ocean_logging_lock();
    ocean_logging_close_owned_stream_locked();
    ocean_logging_stream = stdout;
    ocean_logging_stream_owned = false;
    ocean_logging_unlock();
}

void ocean_logging_to_file(const char *path, bool append) {
    if (!path || path[0] == '\0') {
        fprintf(stderr, "Ocean Logging error: log file path is empty\n");
        exit(1);
    }

    FILE *next = fopen(path, append ? "a" : "w");
    if (!next) {
        fprintf(
            stderr,
            "Ocean Logging error: cannot open '%s'\n",
            path
        );
        exit(1);
    }

    ocean_logging_lock();
    ocean_logging_close_owned_stream_locked();
    ocean_logging_stream = next;
    ocean_logging_stream_owned = true;
    ocean_logging_unlock();
}

static void ocean_logging_write_timestamp(FILE *stream) {
    struct timespec ts;

    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        fputs("0000-00-00 00:00:00.000", stream);
        return;
    }

    time_t seconds = ts.tv_sec;
    struct tm local_tm;

    if (!localtime_r(&seconds, &local_tm)) {
        fputs("0000-00-00 00:00:00.000", stream);
        return;
    }

    char date_time[32];

    if (
        strftime(
            date_time,
            sizeof(date_time),
            "%Y-%m-%d %H:%M:%S",
            &local_tm
        ) == 0
    ) {
        fputs("0000-00-00 00:00:00.000", stream);
        return;
    }

    const long milliseconds = ts.tv_nsec / 1000000L;

    fprintf(
        stream,
        "%s.%03ld",
        date_time,
        milliseconds
    );
}

void ocean_logging_write(
    int level,
    const char *logger_name,
    const char *message
) {
    if (!ocean_logging_enabled(level)) {
        return;
    }

    const char *safe_name =
        (logger_name && logger_name[0] != '\0')
        ? logger_name
        : "root";

    const char *safe_message = message ? message : "";

    ocean_logging_lock();

    FILE *stream = ocean_logging_current_stream();

    if (ocean_logging_get_timestamps()) {
        ocean_logging_write_timestamp(stream);
        fputs(" | ", stream);
    }

    const bool terminal_stream =
        stream == stdout || stream == stderr;
    const bool use_color =
        terminal_stream && ocean_logging_get_colors();

    if (use_color) {
        fputs(ocean_logging_level_color(level), stream);
    }

    fprintf(stream, "%-8s", ocean_logging_level_name(level));

    if (use_color) {
        fputs(ocean_logging_color_reset(), stream);
    }

    fprintf(stream, " | %s | %s\n", safe_name, safe_message);
    fflush(stream);

    ocean_logging_unlock();
}

void ocean_logging_flush(void) {
    ocean_logging_lock();
    fflush(ocean_logging_current_stream());
    ocean_logging_unlock();
}

void ocean_logging_shutdown(void) {
    ocean_logging_lock();
    ocean_logging_close_owned_stream_locked();
    ocean_logging_unlock();
}
