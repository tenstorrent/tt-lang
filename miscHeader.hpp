#include "api/debug/dprint.h" 


void dummy_print() {
    DRPINT("Hello World");  
}

void dummy_add(int a, int b) {
    int c = a + b;
    DPRINT("c: %d", c);
}