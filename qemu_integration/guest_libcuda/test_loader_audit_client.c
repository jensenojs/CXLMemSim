/* The audit fixture needs a genuine cross-DSO cu* relocation.  It deliberately
 * ignores cuInit's device result: loader binding happens before the shim can
 * discover a Type-2 device, and this test only owns the loader boundary. */
extern int cuInit(unsigned int flags);

int main(void) {
    (void)cuInit(0);
    return 0;
}
