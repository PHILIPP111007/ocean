#define _POSIX_C_SOURCE 200809L

#include "os_runtime.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

struct ocean_os_dir_list {
    char **items;
    int size;
    int capacity;
};

static void ocean_os_fail(const char *operation, const char *path) {
    if (path) {
        fprintf(
            stderr,
            "Ocean OS error: %s '%s': %s\n",
            operation,
            path,
            strerror(errno)
        );
    } else {
        fprintf(
            stderr,
            "Ocean OS error: %s: %s\n",
            operation,
            strerror(errno)
        );
    }
    exit(1);
}

static void *ocean_os_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        errno = ENOMEM;
        ocean_os_fail("memory allocation failed", NULL);
    }
    return ptr;
}

static void *ocean_os_realloc(void *ptr, size_t size) {
    void *next = realloc(ptr, size);
    if (!next) {
        errno = ENOMEM;
        ocean_os_fail("memory reallocation failed", NULL);
    }
    return next;
}

static char *ocean_os_strdup(const char *value) {
    const char *source = value ? value : "";
    const size_t size = strlen(source) + 1;
    char *copy = (char *)ocean_os_malloc(size);
    memcpy(copy, source, size);
    return copy;
}

char *ocean_os_getcwd(void) {
    size_t capacity = 256;

    while (capacity <= (size_t)(16 * 1024 * 1024)) {
        char *buffer = (char *)ocean_os_malloc(capacity);

        errno = 0;
        if (getcwd(buffer, capacity)) {
            return buffer;
        }

        const int saved_errno = errno;
        free(buffer);

        if (saved_errno != ERANGE) {
            errno = saved_errno;
            ocean_os_fail("getcwd", NULL);
        }

        capacity *= 2;
    }

    errno = ERANGE;
    ocean_os_fail("getcwd path too long", NULL);
    return NULL;
}

void ocean_os_chdir(const char *path) {
    if (!path) {
        errno = EINVAL;
        ocean_os_fail("chdir", NULL);
    }

    if (chdir(path) != 0) {
        ocean_os_fail("chdir", path);
    }
}

static bool ocean_os_stat(const char *path, struct stat *info) {
    if (!path || !info) {
        return false;
    }
    return stat(path, info) == 0;
}

bool ocean_os_exists(const char *path) {
    struct stat info;
    return ocean_os_stat(path, &info);
}

bool ocean_os_is_file(const char *path) {
    struct stat info;
    return ocean_os_stat(path, &info) && S_ISREG(info.st_mode);
}

bool ocean_os_is_dir(const char *path) {
    struct stat info;
    return ocean_os_stat(path, &info) && S_ISDIR(info.st_mode);
}

bool ocean_os_is_symlink(const char *path) {
    if (!path) {
        return false;
    }

    struct stat info;
    return lstat(path, &info) == 0 && S_ISLNK(info.st_mode);
}

static void ocean_os_mkdir_one(const char *path) {
    if (!path || path[0] == '\0') {
        return;
    }

    if (mkdir(path, 0777) == 0) {
        return;
    }

    if (errno == EEXIST && ocean_os_is_dir(path)) {
        return;
    }

    ocean_os_fail("mkdir", path);
}

void ocean_os_mkdir(const char *path) {
    if (!path || path[0] == '\0') {
        errno = EINVAL;
        ocean_os_fail("mkdir", path);
    }

    ocean_os_mkdir_one(path);
}

void ocean_os_makedirs(const char *path) {
    if (!path || path[0] == '\0') {
        errno = EINVAL;
        ocean_os_fail("makedirs", path);
    }

    char *copy = ocean_os_strdup(path);
    size_t length = strlen(copy);

    while (length > 1 && copy[length - 1] == '/') {
        copy[length - 1] = '\0';
        --length;
    }

    for (char *cursor = copy + 1; *cursor; ++cursor) {
        if (*cursor != '/') {
            continue;
        }

        *cursor = '\0';
        if (copy[0] != '\0') {
            ocean_os_mkdir_one(copy);
        }
        *cursor = '/';
    }

    ocean_os_mkdir_one(copy);
    free(copy);
}

void ocean_os_remove(const char *path) {
    if (!path) {
        errno = EINVAL;
        ocean_os_fail("remove", NULL);
    }

    if (unlink(path) != 0) {
        ocean_os_fail("remove", path);
    }
}

void ocean_os_rmdir(const char *path) {
    if (!path) {
        errno = EINVAL;
        ocean_os_fail("rmdir", NULL);
    }

    if (rmdir(path) != 0) {
        ocean_os_fail("rmdir", path);
    }
}

void ocean_os_rename(const char *source, const char *destination) {
    if (!source || !destination) {
        errno = EINVAL;
        ocean_os_fail("rename", NULL);
    }

    if (rename(source, destination) != 0) {
        ocean_os_fail("rename", source);
    }
}

bool ocean_os_has_env(const char *name) {
    return name && getenv(name) != NULL;
}

char *ocean_os_getenv_copy(const char *name, const char *default_value) {
    if (!name) {
        errno = EINVAL;
        ocean_os_fail("getenv", NULL);
    }

    const char *value = getenv(name);
    if (!value) {
        value = default_value ? default_value : "";
    }

    return ocean_os_strdup(value);
}

void ocean_os_setenv(const char *name, const char *value, bool overwrite) {
    if (!name || !value) {
        errno = EINVAL;
        ocean_os_fail("setenv", NULL);
    }

    if (setenv(name, value, overwrite ? 1 : 0) != 0) {
        ocean_os_fail("setenv", name);
    }
}

void ocean_os_unsetenv(const char *name) {
    if (!name) {
        errno = EINVAL;
        ocean_os_fail("unsetenv", NULL);
    }

    if (unsetenv(name) != 0) {
        ocean_os_fail("unsetenv", name);
    }
}

int64_t ocean_os_pid(void) {
    return (int64_t)getpid();
}

int64_t ocean_os_ppid(void) {
    return (int64_t)getppid();
}

int ocean_os_cpu_count(void) {
    const long count = sysconf(_SC_NPROCESSORS_ONLN);
    if (count < 1) {
        return 1;
    }
    if (count > 2147483647L) {
        return 2147483647;
    }
    return (int)count;
}

char *ocean_os_hostname(void) {
    size_t capacity = 256;

    while (capacity <= (size_t)(1024 * 1024)) {
        char *buffer = (char *)ocean_os_malloc(capacity);
        memset(buffer, 0, capacity);

        errno = 0;
        if (gethostname(buffer, capacity - 1) == 0) {
            buffer[capacity - 1] = '\0';
            return buffer;
        }

        const int saved_errno = errno;
        free(buffer);

        if (saved_errno != ENAMETOOLONG) {
            errno = saved_errno;
            ocean_os_fail("gethostname", NULL);
        }

        capacity *= 2;
    }

    errno = ENAMETOOLONG;
    ocean_os_fail("hostname too long", NULL);
    return NULL;
}

char *ocean_os_platform(void) {
#if defined(__linux__)
    return ocean_os_strdup("linux");
#elif defined(__APPLE__)
    return ocean_os_strdup("macos");
#elif defined(__unix__) || defined(__unix)
    return ocean_os_strdup("unix");
#else
    return ocean_os_strdup("posix");
#endif
}

static void ocean_os_dir_list_append(
    ocean_os_dir_list_t list,
    const char *name
) {
    if (list->size >= list->capacity) {
        int new_capacity = list->capacity > 0
            ? list->capacity * 2
            : 16;

        list->items = (char **)ocean_os_realloc(
            list->items,
            (size_t)new_capacity * sizeof(char *)
        );
        list->capacity = new_capacity;
    }

    list->items[list->size++] = ocean_os_strdup(name);
}

ocean_os_dir_list_t ocean_os_listdir(const char *path) {
    if (!path) {
        errno = EINVAL;
        ocean_os_fail("listdir", NULL);
    }

    DIR *directory = opendir(path);
    if (!directory) {
        ocean_os_fail("listdir", path);
    }

    ocean_os_dir_list_t result =
        (ocean_os_dir_list_t)ocean_os_malloc(
            sizeof(struct ocean_os_dir_list)
        );

    result->items = NULL;
    result->size = 0;
    result->capacity = 0;

    errno = 0;

    for (;;) {
        struct dirent *entry = readdir(directory);
        if (!entry) {
            if (errno != 0) {
                const int saved_errno = errno;
                closedir(directory);
                errno = saved_errno;
                ocean_os_fail("readdir", path);
            }
            break;
        }

        if (
            strcmp(entry->d_name, ".") == 0
            || strcmp(entry->d_name, "..") == 0
        ) {
            continue;
        }

        ocean_os_dir_list_append(result, entry->d_name);
    }

    if (closedir(directory) != 0) {
        ocean_os_dir_list_release(result);
        ocean_os_fail("closedir", path);
    }

    return result;
}

int ocean_os_dir_list_size(ocean_os_dir_list_t list) {
    return list ? list->size : 0;
}

char *ocean_os_dir_list_get_copy(
    ocean_os_dir_list_t list,
    int index
) {
    if (!list || index < 0 || index >= list->size) {
        errno = EINVAL;
        ocean_os_fail("listdir index out of bounds", NULL);
    }

    return ocean_os_strdup(list->items[index]);
}

void ocean_os_dir_list_release(ocean_os_dir_list_t list) {
    if (!list) {
        return;
    }

    for (int i = 0; i < list->size; ++i) {
        free(list->items[i]);
    }

    free(list->items);
    free(list);
}
