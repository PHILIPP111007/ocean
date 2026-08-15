#ifndef OCEAN_STD_TIME_TIME_RUNTIME_H
#define OCEAN_STD_TIME_TIME_RUNTIME_H

#include <stdint.h>

double ocean_time_now(void);
int64_t ocean_time_now_ns(void);
int64_t ocean_time_unix(void);

double ocean_time_monotonic(void);
int64_t ocean_time_monotonic_ns(void);

double ocean_time_process(void);
int64_t ocean_time_process_ns(void);

void ocean_time_sleep(double seconds);
void ocean_time_sleep_ms(int64_t milliseconds);
void ocean_time_sleep_us(int64_t microseconds);

char *ocean_time_format_local(int64_t timestamp, const char *format);
char *ocean_time_format_utc(int64_t timestamp, const char *format);

#endif
