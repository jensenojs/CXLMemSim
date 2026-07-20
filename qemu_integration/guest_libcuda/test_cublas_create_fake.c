int cublasCreate_v2(void **handle) {
    *handle = (void *)0x1;
    return 0;
}

int cublasDestroy_v2(void *handle) {
    return handle == (void *)0x1 ? 0 : 1;
}
