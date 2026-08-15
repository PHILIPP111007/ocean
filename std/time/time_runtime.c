#define _POSIX_C_SOURCE 200809L

#include "time_runtime.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void ocean_time_fail(const char *operation) {
    fprintf(stderr, "Ocean Time error: %s failed\n", operation);
    exit(1);
}

static int64_t ocean_time_timespec_to_ns(const struct timespec *value) {
    return ((int64_t)value->tv_sec * INT64_C(1000000000)) +
           (int64_t)value->tv_nsec;
}

static double ocean_time_timespec_to_seconds(const struct timespec *value) {
    return (double)value->tv_sec + ((double)value->tv_nsec / 1000000000.0);
}

static struct timespec ocean_time_get_clock(clockid_t clock_id, const char *name) {
    struct timespec value;
    if (clock_gettime(clock_id, &value) != 0) {
        ocean_time_fail(name);
    }
    return value;
}

double ocean_time_now(void) {
    const struct timespec value =
        ocean_time_get_clock(CLOCK_REALTIME, "clock_gettime(CLOCK_REALTIME)");
    return ocean_time_timespec_to_seconds(&value);
}

int64_t ocean_time_now_ns(void) {
    const struct timespec value =
        ocean_time_get_clock(CLOCK_REALTIME, "clock_gettime(CLOCK_REALTIME)");
    return ocean_time_timespec_to_ns(&value);
}

int64_t ocean_time_unix(void) {
    const time_t value = time(NULL);
    if (value == (time_t)-1) {
        ocean_time_fail("time");
    }
    return (int64_t)value;
}

double ocean_time_monotonic(void) {
    const struct timespec value =
        ocean_time_get_clock(CLOCK_MONOTONIC, "clock_gettime(CLOCK_MONOTONIC)");
    return ocean_time_timespec_to_seconds(&value);
}

int64_t ocean_time_monotonic_ns(void) {
    const struct timespec value =
        ocean_time_get_clock(CLOCK_MONOTONIC, "clock_gettime(CLOCK_MONOTONIC)");
    return ocean_time_timespec_to_ns(&value);
}

double ocean_time_process(void) {
    const struct timespec value = ocean_time_get_clock(
        CLOCK_PROCESS_CPUTIME_ID,
        "clock_gettime(CLOCK_PROCESS_CPUTIME_ID)"
    );
    return ocean_time_timespec_to_seconds(&value);
}

int64_t ocean_time_process_ns(void) {
    const struct timespec value = ocean_time_get_clock(
        CLOCK_PROCESS_CPUTIME_ID,
        "clock_gettime(CLOCK_PROCESS_CPUTIME_ID)"
    );
    return ocean_time_timespec_to_ns(&value);
}

static void ocean_time_sleep_timespec(struct timespec request) {
    while (nanosleep(&request, &request) != 0) {
        if (errno != EINTR) {
            ocean_time_fail("nanosleep");
        }
    }
}

void ocean_time_sleep(double seconds) {
    if (seconds < 0.0) {
        fprintf(stderr, "Ocean Time error: sleep duration must be non-negative\n");
        exit(1);
    }

    const time_t whole_seconds = (time_t)seconds;
    double fractional = seconds - (double)whole_seconds;
    long nanoseconds = (long)(fractional * 1000000000.0);

    if (nanoseconds >= 1000000000L) {
        nanoseconds = 999999999L;
    }
    if (nanoseconds < 0L) {
        nanoseconds = 0L;
    }

    struct timespec request;
    request.tv_sec = whole_seconds;
    request.tv_nsec = nanoseconds;
    ocean_time_sleep_timespec(request);
}

void ocean_time_sleep_ms(int64_t milliseconds) {
    if (milliseconds < 0) {
        fprintf(stderr, "Ocean Time error: sleep_ms duration must be non-negative\n");
        exit(1);
    }

    struct timespec request;
    request.tv_sec = (time_t)(milliseconds / INT64_C(1000));
    request.tv_nsec = (long)(
        (milliseconds % INT64_C(1000)) * INT64_C(1000000)
    );
    ocean_time_sleep_timespec(request);
}

void ocean_time_sleep_us(int64_t microseconds) {
    if (microseconds < 0) {
        fprintf(stderr, "Ocean Time error: sleep_us duration must be non-negative\n");
        exit(1);
    }

    struct timespec request;
    request.tv_sec = (time_t)(microseconds / INT64_C(1000000));
    request.tv_nsec = (long)(
        (microseconds % INT64_C(1000000)) * INT64_C(1000)
    );
    ocean_time_sleep_timespec(request);
}

static char *ocean_time_empty_string(void) {
    char *result = (char *)malloc(1);
    if (!result) {
        ocean_time_fail("malloc");
    }
    result[0] = '\0';
    return result;
}

static char *ocean_time_format(
    int64_t timestamp,
    const char *format,
    int utc
) {
    if (!format) {
        fprintf(stderr, "Ocean Time error: format must not be null\n");
        exit(1);
    }

    if (format[0] == '\0') {
        return ocean_time_empty_string();
    }

    const time_t raw_time = (time_t)timestamp;
    struct tm broken_down;
    struct tm *converted = utc
        ? gmtime_r(&raw_time, &broken_down)
        : localtime_r(&raw_time, &broken_down);

    if (!converted) {
        ocean_time_fail(utc ? "gmtime_r" : "localtime_r");
    }

    size_t capacity = 128;
    while (capacity <= (size_t)(1024 * 1024)) {
        char *buffer = (char *)malloc(capacity);
        if (!buffer) {
            ocean_time_fail("malloc");
        }

        const size_t length = strftime(
            buffer,
            capacity,
            format,
            &broken_down
        );

        if (length > 0) {
            char *shrunk = (char *)realloc(buffer, length + 1);
            return shrunk ? shrunk : buffer;
        }

        free(buffer);
        capacity *= 2;
    }

    fprintf(stderr, "Ocean Time error: formatted time exceeds 1 MiB\n");
    exit(1);
}

char *ocean_time_format_local(int64_t timestamp, const char *format) {
    return ocean_time_format(timestamp, format, 0);
}

char *ocean_time_format_utc(int64_t timestamp, const char *format) {
    return ocean_time_format(timestamp, format, 1);
}
