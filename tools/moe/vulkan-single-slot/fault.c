// Inject an I/O error or an abort request on the third whole-expert read of layer 0.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdatomic.h>
#include <sys/uio.h>

static size_t start, end, expert_bytes;
static atomic_int mode, count, seen;

void les_probe_arm(size_t offset, size_t size, size_t stride, int kind) {
    start = offset;
    end = offset + size;
    expert_bytes = stride;
    atomic_store(&seen, 0);
    atomic_store(&count, 0);
    atomic_store(&mode, kind);
}

int les_probe_seen(void) {
    return atomic_load(&seen);
}

ssize_t preadv(int fd, const struct iovec * iov, int n, off_t off) {
    ssize_t (*real)(int, const struct iovec *, int, off_t) = dlsym(RTLD_NEXT, "preadv");
    int kind = atomic_load(&mode);
    if (kind && (size_t) off >= start && (size_t) off < end && n == 1 && iov[0].iov_len == expert_bytes &&
            atomic_fetch_add(&count, 1) == 2) {
        atomic_store(&seen, kind);
        atomic_store(&mode, 0);
        if (kind == 1) {
            errno = EIO;
            return -1;
        }
    }
    return real(fd, iov, n, off);
}
