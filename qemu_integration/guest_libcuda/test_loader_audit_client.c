/* The audit fixture needs a genuine cross-DSO cu* relocation.  It deliberately
 * ignores cuInit's device result: loader binding happens before the shim can
 * discover a Type-2 device, and this test only owns the loader boundary. */
#include <sys/wait.h>
#include <unistd.h>

extern int cuInit(unsigned int flags);

int main(void) {
    pid_t child = fork();
    if (child < 0) {
        return 1;
    }
    if (child == 0) {
        (void)cuInit(0);
        _exit(0);
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return 1;
    }
    (void)cuInit(0);
    return 0;
}
