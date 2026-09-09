# Matmul comparisons and historical regressions

The current figure uses TT-Metal device-kernel duration. Historical CSVs use
host wall time, so a device regression versus those CSVs cannot be computed.
The host regression table below is retained, not reclassified as a speedup.

## Historical host-time regressions

The 2026-09-09 sweep passes PCC >= 0.99 for both implementations on all 22
matrix sizes. Every TT-Lang latency is higher than the historical CSV.

Historical reference: `29b13acfa5d387997387fd09db7959d7565ae3c7:benchmarks/matmul/ksplit_sweep.csv`
(committed 2026-05-12 18:10:26 UTC). The historical execution timestamp and
installed binary identities were not recorded; these are observed differences,
not an isolated compiler-regression attribution. The three historical rows
with NaN/PCC < 0.99 are invalid correctness baselines.

The [archived host provenance](https://gist.githubusercontent.com/brnorris03/79c57b196efe09355699d40165780088/raw/d644bd94ae8c11012b55ca7b001a28da252d39a5/matmul_host_ksplit_sweep.json)
records source and binary hashes; the corresponding
[CSV](https://gist.githubusercontent.com/brnorris03/79c57b196efe09355699d40165780088/raw/a70de7a56d14bd123f3e6677a56d9dcc4443d94a/matmul_host_ksplit_sweep.csv) is archived separately on GitHub. The original
host-timed PNG remains in the measurement machine's archive.
Block/grid configurations, BF16 inputs, HiFi4/FP32 accumulation, and
3 warmups followed by minimum of 5 synchronized calls match the old sweep.
The native compute config is now selected for the actual device architecture.
The two TT-Lang kernels exclude the sender from their multicast destinations:
the sender already owns the DRAM-loaded block, and a self-receive after a
blocking send violates the current receiver-posted protocol. Without this
correction, all 22 cases fail compiler validation on current main.

Positive deltas below mean slower. No rows or regressions are omitted.

| Matrix size | Old TT-Lang ms | New TT-Lang ms | TT-Lang delta | Native delta | TT-Lang PCC old -> new |
| --- | ---: | ---: | ---: | ---: | --- |
| 1k^3 | 0.2495 | 1.357 | 443.9% | -16.1% | 1.0 -> 0.999985 |
| 1k x 2k x 1k | 0.2408 | 1.1638 | 383.3% | -6.9% | 0.999987 -> 0.999986 |
| 2k^3 | 0.4462 | 1.9245 | 331.3% | 13.5% | nan -> 1.0 |
| 3k x 1k x 3k (short K) | 0.4001 | 1.7399 | 334.9% | 13.2% | 0.999987 -> 0.999908 |
| 2.5k x 2k x 3k | 0.4667 | 2.063 | 342.0% | -7.0% | 0.999933 -> 0.99977 |
| 2k x 4k x 2k | 0.5577 | 2.1012 | 276.8% | 1.2% | 0.878537 -> 0.999931 |
| 2.5k x 4k x 3k | 0.6756 | 2.3325 | 245.2% | 2.2% | 0.999894 -> 0.99987 |
| 2k x 8k x 2k (long K) | 0.9223 | 2.4998 | 171.0% | 0.4% | 0.898025 -> 0.999967 |
| 3k x 4k x 3k | 0.938 | 2.1874 | 133.2% | 3.4% | 0.999883 -> 0.999897 |
| 1k x 16k x 2.5k (tall K) | 1.1105 | 2.297 | 106.8% | 3.3% | 0.999985 -> 0.999868 |
| 5k x 2k x 5k (short K) | 1.2074 | 2.7291 | 126.0% | 1.7% | 0.999863 -> 0.999945 |
| 2.5k x 8k x 3k (120 cores) | 1.101 | 2.7318 | 148.1% | 4.6% | 0.999898 -> 0.999924 |
| 4k^3 | 1.591 | 2.7723 | 74.2% | 3.6% | 0.999795 -> 0.999877 |
| 2.5k x 8k x 3.3k (130 cores) | 1.0871 | 2.7542 | 153.4% | 6.0% | 0.999863 -> 0.999812 |
| 6k x 2k x 6k (short K) | 1.7613 | 3.36 | 90.8% | 1.8% | 0.99991 -> 0.999844 |
| 4k x 8k x 4k | 2.9422 | 4.1009 | 39.4% | 4.7% | 0.999772 -> 0.999864 |
| 2.5k x 16k x 3.3k | 2.0965 | 3.6785 | 75.5% | 1.4% | 0.999738 -> 0.99975 |
| 2.5k x 32k x 3.3k | 4.372 | 5.9414 | 35.9% | 0.3% | 0.999765 -> 0.999687 |
| 8k^3 | 10.7107 | 12.4966 | 16.7% | 6.5% | 0.999917 -> 0.999919 |
| 10k x 8k x 13k (130 cores, 4x4) | 20.1746 | 21.9176 | 8.6% | -0.3% | 1.0 -> 1.0 |
| 5k x 32k x 6.5k | 19.6941 | 21.8045 | 10.7% | 1.2% | 0.999786 -> 0.999792 |
| 10k x 16k x 13k | 41.5479 | 44.0388 | 6.0% | 1.1% | 1.0 -> 1.0 |

## Current device-time comparison

All 22 cases pass PCC >= 0.99 for both implementations. The comparison uses
minimum of five TT-Metal `device_kernel_duration` samples after three warmups;
12 cases are slower than native. These are implementation differences, not
measured regressions against an older device-timed revision.
See [archived provenance and raw sample locations](https://gist.githubusercontent.com/brnorris03/79c57b196efe09355699d40165780088/raw/65a8900caa9cda4115847a9e27b2711fdec72bee/matmul_device_ksplit_sweep.json).

| Matrix size | TT-Lang device ms | Native device ms | TT-Lang/native |
| --- | ---: | ---: | ---: |
| 1k^3 | 0.056 | 0.0446 | 1.2554 |
| 1k x 2k x 1k | 0.0726 | 0.0794 | 0.9146 |
| 2k^3 | 0.2206 | 0.1929 | 1.1439 |
| 3k x 1k x 3k (short K) | 0.2349 | 0.2175 | 1.08 |
| 2.5k x 2k x 3k | 0.2956 | 0.2937 | 1.0065 |
| 2k x 4k x 2k | 0.376 | 0.3524 | 1.0668 |
| 2.5k x 4k x 3k | 0.5036 | 0.5237 | 0.9618 |
| 2k x 8k x 2k (long K) | 0.6843 | 0.6693 | 1.0225 |
| 3k x 4k x 3k | 0.6949 | 0.6401 | 1.0857 |
| 1k x 16k x 2.5k (tall K) | 1.0053 | 1.0074 | 0.9979 |
| 5k x 2k x 5k (short K) | 1.0196 | 0.9212 | 1.1069 |
| 2.5k x 8k x 3k (120 cores) | 0.9052 | 0.9819 | 0.9219 |
| 4k^3 | 1.3486 | 1.1389 | 1.1841 |
| 2.5k x 8k x 3.3k (130 cores) | 0.8971 | 0.9835 | 0.9121 |
| 6k x 2k x 6k (short K) | 1.5536 | 1.3378 | 1.1613 |
| 4k x 8k x 4k | 2.583 | 2.1542 | 1.1991 |
| 2.5k x 16k x 3.3k | 1.7196 | 1.9072 | 0.9016 |
| 2.5k x 32k x 3.3k | 3.3643 | 3.7418 | 0.8991 |
| 8k^3 | 8.6385 | 8.6098 | 1.0033 |
| 10k x 8k x 13k (130 cores, 4x4) | 13.8912 | 15.7069 | 0.8844 |
| 5k x 32k x 6.5k | 13.3759 | 14.9684 | 0.8936 |
| 10k x 16k x 13k | 27.0876 | 30.4444 | 0.8897 |
