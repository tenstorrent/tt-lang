# Introduction
This document aims to list various cases of undefined behaviors in the Circular Buffer API as defined [here](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.html
).


# List of Undefined Behaviors

## Any cb_* call with number of tiles that does not evenly divide the cb size

````
// Assume size of cb with cb_id is 8
// cb_wait_front(cb_id, 3) // this results in undefined behavior (UB).
````

this stems from:
["Important note: number of tiles used in all cb_* calls must evenly divide the cb size and must be the same number in all cb_wait_front calls in the same kernel. "](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/cb_wait_front.html)

## Push/Pop without reserve/wait

````
// ... no prior `cb_wait_front(cb_id);`
cb_pop_front(cb_id, 1); //  Results in UB
````

````
// ... no prior `cb_reserve_back(cb_id);`
cb_push_back(cb_id, 1); //  Results in UB
````

## Push/Pop more than was reserved/waited for

````
cb_wait_front(cb_id, 1);
void* p0 = get_read_ptr(cb_id);
// ... use p0 ...
cb_pop_front(cb_id, 2) // popping more than was waited for
````

similarly for push:

````
cb_reserve_back(cb_id, 1);
void* p0 = get_write_ptr(cb_id);
// ... use p0 ...
cb_push_back(cb_id, 2) // pushing more than was reserved
````

## Multiple waits without cumulative tile counts

````
cb_wait_front(cb_id, 2);
// ... no subsequent corresponding cb_pop_front ...
cb_wait_front(cb_id, 2); // This results in UB
// if additional blocks need to be waited for, then they need
// to be added on top of the previous wait amount, e.g:
// cb_wait_front(cb_id, 4); // this should work
...
````

this stems from:

["Important note: number of tiles used in all cb_* calls must evenly divide the cb size and must be the same number in all cb_wait_front calls in the same kernel. "](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/cb_wait_front.html)

Note: `cb_reserve_back()` does not have a similar condition

## Multiple waits but cummulative tile counts are not incremented by the initial tile count

````
cb_wait_front(cb_id, 4);
// ... no subsequent corresponding cb_pop_front ...
cb_wait_front(cb_id, 12); // This results in UB.
// cb_wait_front(cb_id, 8); // This should work
````

This stems from:
"Important note: number of tiles used in all cb_* calls must evenly divide the cb size and must be the same number in all cb_wait_front calls in the same kernel. "

Note: `cb_reserve_back()` does not have a similar condition
Note: (Kostas) This is my interpretation of the rule "must be the same number in all cb_wait_front calls in the same kernel". They are not really the same number, just that the additional tiles requested are always the same number.

## Using get_read(write)_ptr outside the wait(reserve)->pop(push) window

for read:

````
// ... no previous wait ...
void *p0 = get_read_ptr(cb_id);
x = *p0 // results in UB
````

````
cb_wait_front(cb_id, 1);
void *p0 = get_read_ptr();
// ...
cb_pop_front(cb_id, 1);
x = *p0 // results in UB
````

for write:

````
// ... no previous reserve ...
void *p0 = get_write_ptr();
*p0 = x; // results in UB
````

````
cb_reserve_back(cb_id, 1);
void *p0 = get_write_ptr();
// ...
cb_push_back(cb_id, 1);
*p0 = x; // results in UB
````

## Out-of-bounds access

````
cb_reserve_back(cb_id, 1);
void *p0 = get_write_ptr();
*p0[100] = x; // results in UB
cb_push_back(cb_id, 1);
````

## Multiple cb_wait_front(...)/cb_push_back(...) calls from different threads on the same circular buffer

````
// Thread 1 (consumer):
cb_wait_front(cb_id, 1);
...

// Thread 2 (consumer):
cb_wait_front(cb_id, 1);
...

// Thread 3 (producer):
cb_push_back(cb_id, 1); // This results in UB

````

and similarly for producers
