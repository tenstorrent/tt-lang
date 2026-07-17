#include "api/debug/dprint.h"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif


void dummy_print() {
    DPRINT("Hello World\n");
}

void dummy_int_add(int a, int b) {
    int c = a + b;
    DPRINT("c: {}\n", c);
}

void dummy_fp_neq(unsigned int a, unsigned int b) {
    if (a != b) {
        DPRINT("NEQ\n");
    }
    else {
        DPRINT("EQ\n");
    }
}

void dummy_cb_index(int cb) {
    DPRINT("received cb index: {}\n", cb);
}

// Read the front tile of a CB and print its first 16-bit word. Guarded to
// data-movement threads (NCRISC/BRISC) where the dataflow API exists.
// For bf16 1.0 the first word is 0x3F80 == 16256.
void dump_cb_front(int cb) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    uint32_t addr = get_read_ptr(cb);
    uint16_t* p = reinterpret_cast<uint16_t*>(addr);
    DPRINT("cb {} front word: {}\n", cb, (int)p[0]);
#endif
}
