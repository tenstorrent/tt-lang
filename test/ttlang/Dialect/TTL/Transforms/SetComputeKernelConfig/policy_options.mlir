// Verify explicit pass policy selects one complete kernel configuration.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{fp32-dest-acc-en=enabled dst-full-sync-en=enabled}))' | FileCheck %s --check-prefix=ENABLED
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{fp32-dest-acc-en=disabled dst-full-sync-en=disabled}))' | FileCheck %s --check-prefix=DISABLED

// ENABLED-LABEL: func.func @policy_options
// ENABLED-SAME: dst_full_sync_en = true
// ENABLED-SAME: fp32_dest_acc_en = true
// ENABLED-SAME: ttl.unpack_to_dest_fp32 = array<i32>
// DISABLED-LABEL: func.func @policy_options
// DISABLED-SAME: dst_full_sync_en = false
// DISABLED-SAME: fp32_dest_acc_en = false
// DISABLED-SAME: ttl.unpack_to_dest_fp32 = array<i32>
func.func @policy_options() {
  return
}
