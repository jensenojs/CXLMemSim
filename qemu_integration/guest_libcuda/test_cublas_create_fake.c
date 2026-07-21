static int initialized;

int cublasCreate_v2(void **handle) {
    initialized = 1;
    *handle = (void *)0x1;
    return 0;
}

int cublasDestroy_v2(void *handle) {
    return handle == (void *)0x1 ? 0 : 1;
}

int cuModuleGetLoadingMode(int *mode) {
    if (!initialized) {
        return 3;
    }
    *mode = 2;
    return 0;
}
