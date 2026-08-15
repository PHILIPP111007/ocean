#ifndef OCEAN_STD_OS_OS_RUNTIME_H
#define OCEAN_STD_OS_OS_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>

typedef struct ocean_os_dir_list *ocean_os_dir_list_t;

char *ocean_os_getcwd(void);
void ocean_os_chdir(const char *path);

void ocean_os_mkdir(const char *path);
void ocean_os_makedirs(const char *path);
void ocean_os_remove(const char *path);
void ocean_os_rmdir(const char *path);
void ocean_os_rename(const char *source, const char *destination);

bool ocean_os_exists(const char *path);
bool ocean_os_is_file(const char *path);
bool ocean_os_is_dir(const char *path);
bool ocean_os_is_symlink(const char *path);

bool ocean_os_has_env(const char *name);
char *ocean_os_getenv_copy(const char *name, const char *default_value);
void ocean_os_setenv(const char *name, const char *value, bool overwrite);
void ocean_os_unsetenv(const char *name);

int64_t ocean_os_pid(void);
int64_t ocean_os_ppid(void);
int ocean_os_cpu_count(void);
char *ocean_os_hostname(void);
char *ocean_os_platform(void);

ocean_os_dir_list_t ocean_os_listdir(const char *path);
int ocean_os_dir_list_size(ocean_os_dir_list_t list);
char *ocean_os_dir_list_get_copy(ocean_os_dir_list_t list, int index);
void ocean_os_dir_list_release(ocean_os_dir_list_t list);

#endif
