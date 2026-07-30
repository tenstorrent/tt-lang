// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/CostEstimator.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>

namespace mlir::tt {

namespace {

constexpr llvm::StringLiteral kThreadAttrName("ttkernel.thread");
constexpr llvm::StringLiteral kTTKernelDialect("ttkernel");
constexpr llvm::StringLiteral kTTLDialect("ttl");

using Lane = CostEstimator::Lane;

/// Functions one TTKernel operation expands to on each RISC.
///
/// `dm` is the expansion when the operation appears in a data-movement kernel.
/// It is a single list rather than one per DM core because the operation does
/// not choose which core it runs on: `ttl.noc_index` on the enclosing function
/// does. The other three are the per-TRISC expansions inside a compute kernel.
/// The `= {}` defaults let an entry name only the lanes it uses; without them
/// -Wmissing-field-initializers rejects the omitted trailing members.
struct ThreadWork {
  llvm::SmallVector<llvm::StringRef, 3> dm = {};
  llvm::SmallVector<llvm::StringRef, 3> unpack = {};
  llvm::SmallVector<llvm::StringRef, 3> math = {};
  llvm::SmallVector<llvm::StringRef, 3> pack = {};

  /// True when the operation is known to cost nothing anywhere. Distinct from
  /// being absent from the table, which means unknown.
  bool isFree() const {
    return dm.empty() && unpack.empty() && math.empty() && pack.empty();
  }
};

/// Thread affinity for TTKernel operations, keyed by operation name.
///
/// A compute kernel is one source file compiled three times, once per TRISC,
/// with -DTRISC_UNPACK / -DTRISC_MATH / -DTRISC_PACK. The UNPACK(), MATH() and
/// PACK() macros in api/compute/common_globals.h keep only the calls belonging
/// to the thread being compiled and erase the rest, so each compute-API call
/// expands to a different number of LLK calls per thread. The counts below are
/// read off the non-Quasar branch of the tt-metal headers:
///
///   circular_buffer.h:31-69 (COMPILE_FOR_TRISC path)
///     wait_front   -> UNPACK llk_wait_tiles
///     pop_front    -> UNPACK llk_pop_tiles
///     reserve_back -> PACK   llk_wait_for_free_tiles
///     push_back    -> PACK   llk_push_tiles
///
///   reg_api.h:45-89
///     tile_regs_acquire -> MATH llk_math_wait_for_dest_available
///     tile_regs_commit  -> MATH llk_math_dest_section_done
///     tile_regs_wait    -> PACK llk_packer_wait_for_math_done
///     tile_regs_release -> PACK llk_pack_dest_section_done
///
///   eltwise_binary.h:31-55  binary_op_init_common
///     UNPACK llk_unpack_hw_configure, llk_unpack_AB_init
///     MATH   llk_math_pack_sync_init, llk_math_hw_configure
///     PACK   llk_pack_hw_configure, llk_pack_init, llk_pack_dest_init
///
///   eltwise_binary.h:72-83, 128-132  add_tiles_init
///     via binary_tiles_init<full_init = true, ELWADD>
///     MATH   llk_math_eltwise_binary_init
///     UNPACK llk_unpack_AB_init   (guarded by `if constexpr (full_init)`)
///
///   eltwise_binary.h:206-214  add_tiles
///     UNPACK llk_unpack_AB
///     MATH   llk_math_eltwise_binary
///
///   pack.h:128-135  pack_tile_block
///     PACK   llk_matmul_pack
///
/// `state_configure()` and `LLK_SAN_FUNCTION()` appear in several of these but
/// are sentinel/sanitizer hooks that compile to nothing in a normal build, so
/// they are not counted.
const llvm::StringMap<ThreadWork> &getThreadWorkTable() {
  static const llvm::StringMap<ThreadWork> table = [] {
    llvm::StringMap<ThreadWork> t;
    auto entry = [&t](llvm::StringRef op) -> ThreadWork & {
      return t[("ttkernel." + op).str()];
    };

    // Member order is dm, unpack, math, pack; trailing members left off are
    // empty. Written with /*name=*/ comments because designated initializers
    // are C++20 and this builds as C++17.

    // -- Circular buffers, api/dataflow/circular_buffer.h:31-69 -------------
    // Under COMPILE_FOR_TRISC the four methods are wrapped PACK/PACK/UNPACK/
    // UNPACK; otherwise they call the plain dataflow functions.
    entry("cb_wait_front") =
        ThreadWork{/*dm=*/{"cb_wait_front"}, /*unpack=*/{"llk_wait_tiles"}};
    entry("cb_pop_front") =
        ThreadWork{/*dm=*/{"cb_pop_front"}, /*unpack=*/{"llk_pop_tiles"}};
    entry("cb_reserve_back") =
        ThreadWork{/*dm=*/{"cb_reserve_back"}, /*unpack=*/{}, /*math=*/{},
                   /*pack=*/{"llk_wait_for_free_tiles"}};
    entry("cb_push_back") =
        ThreadWork{/*dm=*/{"cb_push_back"}, /*unpack=*/{}, /*math=*/{},
                   /*pack=*/{"llk_push_tiles"}};

    // -- DST lifecycle, api/compute/reg_api.h:45-89 ------------------------
    entry("tile_regs_acquire") = ThreadWork{
        /*dm=*/{}, /*unpack=*/{}, /*math=*/{"llk_math_wait_for_dest_available"}};
    entry("tile_regs_commit") = ThreadWork{
        /*dm=*/{}, /*unpack=*/{}, /*math=*/{"llk_math_dest_section_done"}};
    entry("tile_regs_wait") =
        ThreadWork{/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                   /*pack=*/{"llk_packer_wait_for_math_done"}};
    entry("tile_regs_release") =
        ThreadWork{/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                   /*pack=*/{"llk_pack_dest_section_done"}};

    // -- Eltwise binary, api/compute/eltwise_binary.h ----------------------
    // binary_op_init_common, lines 31-55.
    entry("binary_op_init_common") = ThreadWork{
        /*dm=*/{},
        /*unpack=*/{"llk_unpack_hw_configure", "llk_unpack_AB_init"},
        /*math=*/{"llk_math_pack_sync_init", "llk_math_hw_configure"},
        /*pack=*/{"llk_pack_hw_configure", "llk_pack_init",
                  "llk_pack_dest_init"}};
    // add_tiles_init, lines 128-132, via binary_tiles_init lines 72-83. It
    // passes full_init = true, so the UNPACK call inside the
    // `if constexpr (full_init)` guard is kept.
    entry("add_tiles_init") =
        ThreadWork{/*dm=*/{}, /*unpack=*/{"llk_unpack_AB_init"},
                   /*math=*/{"llk_math_eltwise_binary_init"}};
    // add_tiles, lines 206-214.
    entry("add_tiles") =
        ThreadWork{/*dm=*/{}, /*unpack=*/{"llk_unpack_AB"},
                   /*math=*/{"llk_math_eltwise_binary"}};

    // -- Pack, api/compute/pack.h:128-135 ----------------------------------
    // llk_matmul_pack is llk_pack hoisted out of a loop; the name is vestigial
    // and pack_tile_block is its only caller.
    entry("pack_tile_block") = ThreadWork{/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                                          /*pack=*/{"llk_matmul_pack"}};

    // -- Data movement, api/dataflow/{noc,dataflow_api,circular_buffer}.h --
    // A barrier's cost is the transfer it waits on, not the call itself.
    entry("noc_async_read_tile") = ThreadWork{/*dm=*/{"Noc::async_read"}};
    entry("noc_async_write_tile") = ThreadWork{/*dm=*/{"Noc::async_write"}};
    entry("noc_async_read_barrier") =
        ThreadWork{/*dm=*/{"Noc::async_read_barrier"}};
    entry("noc_async_write_barrier") =
        ThreadWork{/*dm=*/{"Noc::async_write_barrier"}};
    entry("get_common_arg_val") = ThreadWork{/*dm=*/{"get_common_arg_val"}};
    entry("get_write_ptr") =
        ThreadWork{/*dm=*/{"CircularBuffer::get_write_ptr"}};
    entry("get_read_ptr") = ThreadWork{/*dm=*/{"CircularBuffer::get_read_ptr"}};
    entry("TensorAccessor") =
        ThreadWork{/*dm=*/{"TensorAccessor::TensorAccessor"}};

    // -- Known to be free --------------------------------------------------
    // Present with no calls, which is how the table says "costs nothing", as
    // opposed to being absent, which means unknown. get_compile_time_arg_val
    // is a compile-time constant that only constructs a CircularBuffer handle
    // holding an id; TensorAccessorArgs is a template instantiation.
    entry("get_compile_time_arg_val") = {};
    entry("TensorAccessorArgs") = {};

    return t;
  }();
  return table;
}

/// Compact source position for the report, e.g. "eltwise_add.py:34". Empty when
/// the location is not a file/line, which keeps the report readable for IR that
/// carries no debug info (hand-written test cases, for instance).
std::string formatLoc(Location loc) {
  auto fileLine = dyn_cast<FileLineColLoc>(loc);
  if (!fileLine) {
    return "";
  }
  llvm::StringRef path = fileLine.getFilename().getValue();
  llvm::StringRef base = path.rsplit('/').second;
  if (base.empty()) {
    base = path;
  }
  return (base + ":" + std::to_string(fileLine.getLine())).str();
}

/// Lane a data-movement kernel runs on. `ttl.noc_index` 0 is the reader on
/// RISCV_1 (NCRISC) and 1 is the writer on RISCV_0 (BRISC), matching
/// `getNocIndex` in TTLOpsUtils.h and the Metalium reader/writer presets.
/// A function without the attribute takes the reader default, as lowering does.
Lane getDataMovementLane(func::FuncOp funcOp) {
  auto nocIndex = funcOp->getAttrOfType<IntegerAttr>(ttl::kNocIndexAttrName);
  if (nocIndex && nocIndex.getInt() == 1) {
    return Lane::Brisc;
  }
  return Lane::Ncrisc;
}

} // namespace

llvm::ArrayRef<CostEstimator::Lane> CostEstimator::getAllLanes() {
  static constexpr Lane lanes[kNumLanes] = {
      Lane::Ncrisc, Lane::Trisc0Unpack, Lane::Trisc1Math, Lane::Trisc2Pack,
      Lane::Brisc};
  return lanes;
}

llvm::StringRef CostEstimator::getLaneName(Lane lane) {
  switch (lane) {
  case Lane::Ncrisc:
    return "NCRISC reader";
  case Lane::Trisc0Unpack:
    return "TRISC0 unpack";
  case Lane::Trisc1Math:
    return "TRISC1 math";
  case Lane::Trisc2Pack:
    return "TRISC2 pack";
  case Lane::Brisc:
    return "BRISC writer";
  }
  return "unknown lane";
}

std::string CostEstimator::Report::render() const {
  std::string text;
  llvm::raw_string_ostream out(text);

  out << "cost estimate: operation placement only, no cycle costs yet\n";

  out << "  kernels:";
  for (llvm::StringRef kernel : kernels) {
    out << " " << kernel;
  }
  if (kernels.empty()) {
    out << " (none)";
  }
  out << "\n";

  // Widest op name across all lanes, so the source column lines up.
  size_t nameWidth = 0;
  for (const LaneReport &lane : lanes) {
    for (const PlacedOp &op : lane.ops) {
      nameWidth = std::max(nameWidth, op.name.size());
    }
  }

  for (Lane lane : getAllLanes()) {
    const LaneReport &laneReport = lanes[getLaneIndex(lane)];
    out << "\n  " << getLaneName(lane) << " (" << laneReport.ops.size()
        << " ops, " << laneReport.llkCalls() << " llk calls)\n";
    if (laneReport.ops.empty()) {
      out << "    (idle)\n";
      continue;
    }
    for (const PlacedOp &op : laneReport.ops) {
      out << "    " << llvm::left_justify(op.name, nameWidth) << "  ";
      llvm::interleave(op.calls, out, ", ");
      std::string where = formatLoc(op.loc);
      if (!where.empty()) {
        out << "  [" << where << "]";
      }
      out << "\n";
    }
  }

  if (isComplete()) {
    out << "\n  complete: every operation was placed\n";
    return text;
  }

  out << "\n  incomplete: " << unknowns.size() << " unaccounted\n";
  for (const Unknown &unknown : unknowns) {
    out << "    " << unknown.message << "\n";
  }
  return text;
}

class CostEstimator::Impl {
public:
  Impl(ModuleOp module, Options options) : module(module), options(options) {}

  FailureOr<Report> estimate() {
    // Reject IR from before convert-ttl-to-ttkernel: the per-thread operation
    // sequence does not exist yet, so any estimate would be of the wrong
    // program.
    WalkResult preLowering = module.walk([](Operation *op) {
      if (op->getName().getDialectNamespace() == kTTLDialect &&
          op->getName().stripDialect() == "compute") {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (preLowering.wasInterrupted()) {
      return module.emitError()
             << "cost estimator needs TTKernel IR, but the module still "
                "contains ttl.compute; run it after convert-ttl-to-ttkernel";
    }

    Report report;
    uint64_t ttkernelOps = 0;

    for (auto funcOp : module.getOps<func::FuncOp>()) {
      auto thread =
          funcOp->getAttrOfType<ttkernel::ThreadTypeAttr>(kThreadAttrName);
      if (!thread) {
        continue;
      }
      report.kernels.push_back(funcOp.getSymName().str());

      switch (thread.getValue()) {
      case ttkernel::ThreadType::Noc:
        ttkernelOps += placeFunc(funcOp, getDataMovementLane(funcOp), report);
        break;
      case ttkernel::ThreadType::Compute:
        ttkernelOps += placeFunc(funcOp, std::nullopt, report);
        break;
      default:
        report.unknowns.push_back(
            {("unsupported kernel thread on '" + funcOp.getSymName() + "'")
                 .str(),
             funcOp.getLoc()});
        break;
      }
    }

    // Reject IR from after convert-ttkernel-to-emitc: circular-buffer calls
    // have become opaque verbatim strings by then, so an empty walk here means
    // the module is past the stage this estimator reads, not that it is
    // trivial.
    if (ttkernelOps == 0 && !report.kernels.empty()) {
      return module.emitError()
             << "cost estimator found no ttkernel operations in "
             << report.kernels.size()
             << " kernel function(s); the module is probably already lowered "
                "to EmitC";
    }

    return report;
  }

private:
  /// Place every TTKernel operation in one kernel function.
  ///
  /// `dmLane` is set for a data-movement kernel, which compiles for a single
  /// RISC so all of its work lands on that one lane. For a compute kernel it is
  /// nullopt and each operation fans out onto the TRISCs whose macro keeps it.
  /// Returns the number of TTKernel operations seen, placed or not.
  uint64_t placeFunc(func::FuncOp funcOp, std::optional<Lane> dmLane,
                     Report &report) {
    const llvm::StringMap<ThreadWork> &table = getThreadWorkTable();
    llvm::StringSet<> reportedNames;
    uint64_t seen = 0;

    funcOp.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() != kTTKernelDialect) {
        return;
      }
      llvm::StringRef name = op->getName().getStringRef();
      ++seen;

      auto found = table.find(name);
      if (found == table.end()) {
        // One unknown per distinct name keeps the report readable while still
        // naming everything the affinity table is missing.
        if (reportedNames.insert(name).second) {
          report.unknowns.push_back(
              {("no thread affinity for '" + name + "'").str(), op->getLoc()});
        }
        return;
      }
      const ThreadWork &work = found->second;

      auto place = [&](Lane lane, llvm::ArrayRef<llvm::StringRef> calls) {
        if (calls.empty()) {
          return false;
        }
        llvm::SmallVector<llvm::StringRef> callList(calls.begin(), calls.end());
        report.lanes[getLaneIndex(lane)].ops.push_back(
            PlacedOp{name.str(), op->getLoc(), std::move(callList)});
        return true;
      };

      bool placed = false;
      if (dmLane) {
        placed = place(*dmLane, work.dm);
      } else {
        placed |= place(Lane::Trisc0Unpack, work.unpack);
        placed |= place(Lane::Trisc1Math, work.math);
        placed |= place(Lane::Trisc2Pack, work.pack);
      }

      // The table knows this operation, but not for the kind of kernel it
      // appeared in. Placing nothing would silently report it as free.
      if (!placed && !work.isFree() && reportedNames.insert(name).second) {
        report.unknowns.push_back(
            {("'" + name + "' has no expansion for a " +
              (dmLane ? "data-movement" : "compute") + " kernel")
                 .str(),
             op->getLoc()});
      }
    });
    return seen;
  }

  ModuleOp module;
  Options options;
};

CostEstimator::CostEstimator(ModuleOp module)
    : CostEstimator(module, Options{}) {}

CostEstimator::CostEstimator(ModuleOp module, Options options)
    : impl(std::make_unique<Impl>(module, options)) {}

CostEstimator::~CostEstimator() = default;
CostEstimator::CostEstimator(CostEstimator &&) noexcept = default;
CostEstimator &CostEstimator::operator=(CostEstimator &&) noexcept = default;

FailureOr<CostEstimator::Report> CostEstimator::estimate() {
  return impl->estimate();
}

} // namespace mlir::tt
