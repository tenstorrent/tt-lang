// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//
// Transfer graph validation, deterministic edge enumeration, and compact
// lowering formulas for logical-device relations.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <limits>

namespace mlir::tt::ttl {

LogicalResult
verifyComponentCoordinates(DeviceDomainComponentAttr component,
                           DenseI64ArrayAttr coordinate,
                           llvm::function_ref<InFlightDiagnostic()> emitError,
                           StringRef context, bool allowUpperBound) {
  ArrayRef<int64_t> extent = component.getExtent().asArrayRef();
  ArrayRef<int64_t> values = coordinate.asArrayRef();
  if (values.size() != extent.size()) {
    return emitError() << context << " component '"
                       << component.getName().getValue() << "' has rank "
                       << values.size() << ", expected " << extent.size();
  }
  for (auto [axis, value] : llvm::enumerate(values)) {
    bool upperBoundValid =
        allowUpperBound ? value <= extent[axis] : value < extent[axis];
    if (value < 0 || !upperBoundValid) {
      return emitError() << context << " component '"
                         << component.getName().getValue() << "' axis " << axis
                         << " is out of bounds for extent " << extent[axis]
                         << ", got " << value;
    }
  }
  return success();
}

LogicalResult
verifyDeviceRefInDomain(DeviceDomainAttr domain, DeviceRefAttr deviceRef,
                        llvm::function_ref<InFlightDiagnostic()> emitError,
                        StringRef context, bool allowUpperBound) {
  ArrayRef<DeviceDomainComponentAttr> components = domain.getComponents();
  ArrayRef<DenseI64ArrayAttr> coordinates = deviceRef.getCoordinates();
  if (coordinates.size() != components.size()) {
    return emitError() << context << " has " << coordinates.size()
                       << " component coordinates, expected "
                       << components.size();
  }

  for (auto [component, coordinate] : llvm::zip(components, coordinates)) {
    if (failed(verifyComponentCoordinates(component, coordinate, emitError,
                                          context, allowUpperBound))) {
      return failure();
    }
  }
  return success();
}

LogicalResult
verifyTransferEdgeInDomain(DeviceDomainAttr domain, TransferEdgeAttr edge,
                           llvm::function_ref<InFlightDiagnostic()> emitError,
                           StringRef context) {
  if (failed(
          verifyDeviceRefInDomain(domain, edge.getSource(), emitError,
                                  (llvm::Twine(context) + ".source").str()))) {
    return failure();
  }
  if (DeviceRefAttr destination = edge.getDestination()) {
    if (failed(verifyDeviceRefInDomain(
            domain, destination, emitError,
            (llvm::Twine(context) + ".destination").str()))) {
      return failure();
    }
    if (destination == edge.getSource()) {
      return emitError() << context << " source must differ from destination";
    }
    return success();
  }

  DeviceRangeAttr destinationRange = edge.getDestinationRange();
  if (failed(verifyDeviceRefInDomain(
          domain, destinationRange.getLo(), emitError,
          (llvm::Twine(context) + ".destination_range.lo").str())) ||
      failed(verifyDeviceRefInDomain(
          domain, destinationRange.getHi(), emitError,
          (llvm::Twine(context) + ".destination_range.hi").str(), true))) {
    return failure();
  }
  if (deviceRangeContains(destinationRange, edge.getSource())) {
    return emitError()
           << context
           << " source must not be contained in its destination range";
  }
  return success();
}

namespace {

void enumerateComponentCoordinates(
    MLIRContext *context, ArrayRef<int64_t> extent, std::size_t axis,
    SmallVectorImpl<int64_t> &coordinates,
    SmallVectorImpl<DenseI64ArrayAttr> &componentCoordinates) {
  if (axis == extent.size()) {
    componentCoordinates.push_back(
        DenseI64ArrayAttr::get(context, coordinates));
    return;
  }

  for (int64_t coordinate = 0; coordinate < extent[axis]; ++coordinate) {
    coordinates.push_back(coordinate);
    enumerateComponentCoordinates(context, extent, axis + 1, coordinates,
                                  componentCoordinates);
    coordinates.pop_back();
  }
}

SmallVector<DeviceRefAttr> enumerateDomainDevices(DeviceDomainAttr domain) {
  MLIRContext *context = domain.getContext();
  SmallVector<SmallVector<DenseI64ArrayAttr>> deviceCoordinates(1);
  for (DeviceDomainComponentAttr component : domain.getComponents()) {
    SmallVector<DenseI64ArrayAttr> componentCoordinates;
    SmallVector<int64_t> coordinates;
    enumerateComponentCoordinates(context, component.getExtent().asArrayRef(),
                                  0, coordinates, componentCoordinates);

    SmallVector<SmallVector<DenseI64ArrayAttr>> expandedCoordinates;
    expandedCoordinates.reserve(deviceCoordinates.size() *
                                componentCoordinates.size());
    for (ArrayRef<DenseI64ArrayAttr> prefix : deviceCoordinates) {
      for (DenseI64ArrayAttr componentCoordinate : componentCoordinates) {
        SmallVector<DenseI64ArrayAttr> completeCoordinates(prefix);
        completeCoordinates.push_back(componentCoordinate);
        expandedCoordinates.push_back(std::move(completeCoordinates));
      }
    }
    deviceCoordinates = std::move(expandedCoordinates);
  }

  SmallVector<DeviceRefAttr> devices;
  devices.reserve(deviceCoordinates.size());
  for (ArrayRef<DenseI64ArrayAttr> coordinates : deviceCoordinates) {
    devices.push_back(DeviceRefAttr::get(context, coordinates));
  }
  return devices;
}

std::optional<std::size_t> findDomainComponentIndex(DeviceDomainAttr domain,
                                                    StringAttr componentName) {
  auto componentIt = llvm::find_if(
      domain.getComponents(), [&](DeviceDomainComponentAttr component) {
        return component.getName() == componentName;
      });
  if (componentIt == domain.getComponents().end()) {
    return std::nullopt;
  }
  return std::distance(domain.getComponents().begin(), componentIt);
}

TransferEdgeAttr getPointTransferEdge(MLIRContext *context,
                                      DeviceRefAttr source,
                                      DeviceRefAttr destination) {
  return TransferEdgeAttr::get(context, source, destination, DeviceRangeAttr());
}

FailureOr<std::uint64_t> getExtentElementCount(ArrayRef<int64_t> extent) {
  std::uint64_t count = 1;
  for (int64_t dimension : extent) {
    std::optional<std::uint64_t> product =
        llvm::checkedMulUnsigned(count, static_cast<std::uint64_t>(dimension));
    if (!product) {
      return failure();
    }
    count = *product;
  }
  return count;
}

FailureOr<std::uint64_t> getDomainDeviceCount(DeviceDomainAttr domain) {
  std::uint64_t count = 1;
  for (DeviceDomainComponentAttr component : domain.getComponents()) {
    FailureOr<std::uint64_t> componentCount =
        getExtentElementCount(component.getExtent().asArrayRef());
    if (failed(componentCount)) {
      return failure();
    }
    std::optional<std::uint64_t> product =
        llvm::checkedMulUnsigned(count, *componentCount);
    if (!product) {
      return failure();
    }
    count = *product;
  }
  return count;
}

struct StencilOffsetDescriptor {
  DenseI64ArrayAttr offset;
  SmallVector<int64_t> sourceLowerBounds;
  SmallVector<int64_t> sourceExtents;
  std::uint64_t sourceCount = 0;
};

FailureOr<SmallVector<StencilOffsetDescriptor>>
getStencilOffsetDescriptors(DeviceDomainComponentAttr component,
                            ArrayAttr offsets, bool wrap) {
  MLIRContext *context = component.getContext();
  ArrayRef<int64_t> componentExtent = component.getExtent().asArrayRef();
  llvm::DenseSet<DenseI64ArrayAttr> emittedOffsets;
  SmallVector<StencilOffsetDescriptor> descriptors;
  for (Attribute offsetAttribute : offsets) {
    auto offset = mlir::cast<DenseI64ArrayAttr>(offsetAttribute);
    SmallVector<int64_t> effectiveOffset(offset.asArrayRef());
    SmallVector<int64_t> sourceLowerBounds(componentExtent.size(), 0);
    SmallVector<int64_t> sourceExtents(componentExtent);
    bool hasEdges = true;
    for (auto [axis, delta] : llvm::enumerate(effectiveOffset)) {
      int64_t extent = componentExtent[axis];
      if (wrap) {
        delta %= extent;
        if (delta < 0) {
          delta += extent;
        }
        effectiveOffset[axis] = delta;
        continue;
      }
      if (delta <= -extent || delta >= extent) {
        hasEdges = false;
        break;
      }
      if (delta < 0) {
        sourceLowerBounds[axis] = -delta;
        sourceExtents[axis] = extent + delta;
      } else {
        sourceExtents[axis] = extent - delta;
      }
    }
    if (!hasEdges || llvm::all_of(effectiveOffset,
                                  [](int64_t delta) { return delta == 0; })) {
      continue;
    }

    DenseI64ArrayAttr effectiveOffsetAttr =
        DenseI64ArrayAttr::get(context, effectiveOffset);
    if (!emittedOffsets.insert(effectiveOffsetAttr).second) {
      continue;
    }
    FailureOr<std::uint64_t> sourceCount = getExtentElementCount(sourceExtents);
    if (failed(sourceCount)) {
      return failure();
    }
    descriptors.push_back(StencilOffsetDescriptor{
        effectiveOffsetAttr, std::move(sourceLowerBounds),
        std::move(sourceExtents), *sourceCount});
  }
  return descriptors;
}

Value buildIndexTableLookup(OpBuilder &builder, Location loc,
                            ArrayRef<int64_t> values, Value index) {
  assert(!values.empty() && "transfer graph index table must not be empty");
  return ttkernel::ConstantTableLookupOp::create(
      builder, loc, builder.getIndexType(), index,
      builder.getDenseI64ArrayAttr(values));
}

TransferGraphEdgeIndexValues
buildEdgeIndexTableLookups(DeviceDomainAttr domain,
                           ArrayRef<TransferEdgeAttr> edges, OpBuilder &builder,
                           Location loc, Value edgeIndex) {
  SmallVector<int64_t> sourceIndices;
  SmallVector<int64_t> destinationIndices;
  sourceIndices.reserve(edges.size());
  destinationIndices.reserve(edges.size());
  for (TransferEdgeAttr edge : edges) {
    sourceIndices.push_back(getLogicalDeviceIndex(domain, edge.getSource()));
    destinationIndices.push_back(
        getLogicalDeviceIndex(domain, edge.getDestination()));
  }
  return {edgeIndex,
          buildIndexTableLookup(builder, loc, sourceIndices, edgeIndex),
          buildIndexTableLookup(builder, loc, destinationIndices, edgeIndex)};
}

struct ExplicitIncidentEdgeTables {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> counts;
  SmallVector<int64_t> edgeOrdinals;
};

ExplicitIncidentEdgeTables buildExplicitIncidentEdgeTables(
    DeviceDomainAttr domain, ArrayRef<TransferEdgeAttr> edges, PipeRole role) {
  assert(role != PipeRole::Active &&
         "dynamic incident iteration requires one endpoint role");
  FailureOr<std::uint64_t> deviceCount = getDomainDeviceCount(domain);
  assert(succeeded(deviceCount) &&
         "graph verification must reject overflowing domain extents");
  SmallVector<SmallVector<int64_t>> edgeOrdinalsByDevice(*deviceCount);
  for (auto [edgeOrdinal, edge] : llvm::enumerate(edges)) {
    DeviceRefAttr endpoint =
        role == PipeRole::Source ? edge.getSource() : edge.getDestination();
    edgeOrdinalsByDevice[getLogicalDeviceIndex(domain, endpoint)].push_back(
        edgeOrdinal);
  }

  ExplicitIncidentEdgeTables tables;
  for (ArrayRef<int64_t> deviceEdgeOrdinals : edgeOrdinalsByDevice) {
    tables.offsets.push_back(tables.edgeOrdinals.size());
    tables.counts.push_back(deviceEdgeOrdinals.size());
    tables.edgeOrdinals.append(deviceEdgeOrdinals.begin(),
                               deviceEdgeOrdinals.end());
  }
  return tables;
}

class ExplicitTransferGraph final : public TransferGraph {
public:
  using TransferGraph::TransferGraph;

  LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> emitError) const override {
    if (getComponentName()) {
      return emitError()
             << "explicit transfer graph must not name a domain component";
    }
    DictionaryAttr properties = getProperties();
    ArrayAttr edges = properties.getAs<ArrayAttr>("edges");
    if (!edges || edges.empty() || properties.size() != 1) {
      return emitError()
             << "explicit transfer graph requires only a nonempty edges array";
    }
    llvm::DenseSet<TransferEdgeAttr> uniqueEdges;
    for (auto [edgeIndex, edgeAttribute] : llvm::enumerate(edges)) {
      auto edge = mlir::dyn_cast<TransferEdgeAttr>(edgeAttribute);
      if (!edge) {
        return emitError() << "explicit transfer graph edge " << edgeIndex
                           << " is not a #ttl.transfer_edge attribute";
      }
      if (!edge.getDestination()) {
        return emitError() << "explicit transfer graph edge " << edgeIndex
                           << " requires one destination device";
      }
      if (failed(verifyTransferEdgeInDomain(
              getDomain(), edge, emitError,
              (llvm::Twine("transfer graph edge ") + llvm::Twine(edgeIndex))
                  .str()))) {
        return failure();
      }
      if (!uniqueEdges.insert(edge).second) {
        return emitError() << "explicit transfer graph edge " << edgeIndex
                           << " duplicates an earlier edge";
      }
    }
    return verifyNonemptyEdgeCount(emitError);
  }

  void forEachEdge(
      llvm::function_ref<void(TransferEdgeAttr)> callback) const override {
    for (Attribute edge : getProperties().getAs<ArrayAttr>("edges")) {
      callback(mlir::cast<TransferEdgeAttr>(edge));
    }
  }

  FailureOr<std::uint64_t> getEdgeCount() const override {
    return getProperties().getAs<ArrayAttr>("edges").size();
  }

  TransferGraphEdgeIndexValues
  buildEdgeIndexValues(OpBuilder &builder, Location loc,
                       Value edgeIndex) const override {
    return buildEdgeIndexTableLookups(getDomain(), getEdges(), builder, loc,
                                      edgeIndex);
  }

  Value buildIncidentEdgeCount(OpBuilder &builder, Location loc,
                               Value deviceIndex,
                               PipeRole role) const override {
    ExplicitIncidentEdgeTables tables =
        buildExplicitIncidentEdgeTables(getDomain(), getEdges(), role);
    return buildIndexTableLookup(builder, loc, tables.counts, deviceIndex);
  }

  TransferGraphEdgeIndexValues
  buildIncidentEdgeIndexValues(OpBuilder &builder, Location loc,
                               Value deviceIndex, Value incidentEdgeIndex,
                               PipeRole role) const override {
    SmallVector<TransferEdgeAttr> edges = getEdges();
    ExplicitIncidentEdgeTables tables =
        buildExplicitIncidentEdgeTables(getDomain(), edges, role);
    Value deviceOffset =
        buildIndexTableLookup(builder, loc, tables.offsets, deviceIndex);
    Value flattenedIndex =
        arith::AddIOp::create(builder, loc, deviceOffset, incidentEdgeIndex);
    Value edgeOrdinal = buildIndexTableLookup(builder, loc, tables.edgeOrdinals,
                                              flattenedIndex);
    return buildEdgeIndexValues(builder, loc, edgeOrdinal);
  }
};

class StructuredTransferGraph : public TransferGraph {
public:
  StructuredTransferGraph(DeviceDomainAttr graphDomain,
                          TransferGraphKind graphKind,
                          StringAttr graphComponentName,
                          DictionaryAttr graphProperties)
      : StructuredTransferGraph(
            graphDomain, graphKind, graphComponentName, graphProperties,
            findDomainComponentIndex(graphDomain, graphComponentName)) {}

private:
  StructuredTransferGraph(DeviceDomainAttr graphDomain,
                          TransferGraphKind graphKind,
                          StringAttr graphComponentName,
                          DictionaryAttr graphProperties,
                          std::optional<std::size_t> graphComponentIndex)
      : TransferGraph(graphDomain, graphKind, graphComponentName,
                      graphProperties),
        context(graphDomain.getContext()), domain(graphDomain),
        componentIndex(graphComponentIndex.value_or(0)),
        component(graphComponentIndex ? domain.getComponents()[componentIndex]
                                      : DeviceDomainComponentAttr()) {}

protected:
  LogicalResult
  verifyStructured(llvm::function_ref<InFlightDiagnostic()> emitError) const {
    StringAttr componentName = getComponentName();
    if (!componentName) {
      return emitError()
             << "structured transfer graph requires a domain component name";
    }
    if (!component) {
      return emitError() << "structured transfer graph references unknown "
                            "domain component '"
                         << componentName.getValue() << "'";
    }
    FailureOr<std::uint64_t> deviceCount = getDeviceCount();
    if (failed(deviceCount) ||
        *deviceCount >
            static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
      return emitError()
             << "structured transfer graph device count exceeds the supported "
                "index range";
    }
    return success();
  }

  DeviceRefAttr replaceComponent(DeviceRefAttr device,
                                 DenseI64ArrayAttr coordinates) const {
    SmallVector<DenseI64ArrayAttr> completeCoordinates(device.getCoordinates());
    completeCoordinates[componentIndex] = coordinates;
    return DeviceRefAttr::get(context, completeCoordinates);
  }

  FailureOr<std::uint64_t> getComponentSize() const {
    return getExtentElementCount(component.getExtent().asArrayRef());
  }

  FailureOr<std::uint64_t> getDeviceCount() const {
    return getDomainDeviceCount(domain);
  }

  ArrayRef<DeviceRefAttr> getDevices() const {
    if (!devices) {
      devices = enumerateDomainDevices(domain);
    }
    return *devices;
  }

  FailureOr<std::uint64_t> getTrailingComponentSize() const {
    std::uint64_t count = 1;
    for (DeviceDomainComponentAttr trailingComponent :
         domain.getComponents().drop_front(componentIndex + 1)) {
      FailureOr<std::uint64_t> componentSize =
          getExtentElementCount(trailingComponent.getExtent().asArrayRef());
      if (failed(componentSize)) {
        return failure();
      }
      std::optional<std::uint64_t> product =
          llvm::checkedMulUnsigned(count, *componentSize);
      if (!product) {
        return failure();
      }
      count = *product;
    }
    return count;
  }

  Value getComponentIndex(OpBuilder &builder, Location loc,
                          Value deviceIndex) const {
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    FailureOr<std::uint64_t> trailingSize = getTrailingComponentSize();
    assert(succeeded(componentSize) && succeeded(trailingSize) &&
           "edge count validation must reject overflowing domain extents");
    Value trailing = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*trailingSize));
    Value componentExtent = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*componentSize));
    Value withoutTrailing =
        arith::DivSIOp::create(builder, loc, deviceIndex, trailing);
    return arith::RemSIOp::create(builder, loc, withoutTrailing,
                                  componentExtent);
  }

  Value getComponentAxisCoordinate(OpBuilder &builder, Location loc,
                                   Value deviceIndex, std::size_t axis) const {
    std::uint64_t coordinateStride = 1;
    for (int64_t extent :
         component.getExtent().asArrayRef().drop_front(axis + 1)) {
      coordinateStride *= extent;
    }
    Value componentIndex = getComponentIndex(builder, loc, deviceIndex);
    Value stride = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(coordinateStride));
    Value axisExtent = arith::ConstantIndexOp::create(
        builder, loc, component.getExtent()[axis]);
    Value withoutTrailing =
        arith::DivSIOp::create(builder, loc, componentIndex, stride);
    return arith::RemSIOp::create(builder, loc, withoutTrailing, axisExtent);
  }

  SmallVector<Value>
  getComponentCoordinatesFromDevice(OpBuilder &builder, Location loc,
                                    Value deviceIndex) const {
    SmallVector<Value> coordinates;
    coordinates.reserve(component.getExtent().size());
    for (int64_t axis = 0; axis < component.getExtent().size(); ++axis) {
      coordinates.push_back(
          getComponentAxisCoordinate(builder, loc, deviceIndex, axis));
    }
    return coordinates;
  }

  Value buildComponentLinearIndex(OpBuilder &builder, Location loc,
                                  ArrayRef<Value> coordinates) const {
    Value linearIndex = arith::ConstantIndexOp::create(builder, loc, 0);
    for (auto [coordinate, extent] :
         llvm::zip_equal(coordinates, component.getExtent().asArrayRef())) {
      Value extentValue = arith::ConstantIndexOp::create(builder, loc, extent);
      Value withAxis =
          arith::MulIOp::create(builder, loc, linearIndex, extentValue);
      linearIndex = arith::AddIOp::create(builder, loc, withAxis, coordinate);
    }
    return linearIndex;
  }

  Value replaceComponentAxisCoordinate(OpBuilder &builder, Location loc,
                                       Value deviceIndex, std::size_t axis,
                                       Value replacementCoordinate) const {
    std::uint64_t coordinateStride = 1;
    for (int64_t extent :
         component.getExtent().asArrayRef().drop_front(axis + 1)) {
      coordinateStride *= extent;
    }
    Value currentCoordinate =
        getComponentAxisCoordinate(builder, loc, deviceIndex, axis);
    Value stride = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(coordinateStride));
    Value currentOffset =
        arith::MulIOp::create(builder, loc, currentCoordinate, stride);
    Value replacementOffset =
        arith::MulIOp::create(builder, loc, replacementCoordinate, stride);
    Value withoutCurrent =
        arith::SubIOp::create(builder, loc, deviceIndex, currentOffset);
    return arith::AddIOp::create(builder, loc, withoutCurrent,
                                 replacementOffset);
  }

  Value buildCountFromPredicate(OpBuilder &builder, Location loc,
                                Value predicate, int64_t trueCount = 1) const {
    Value count = arith::ConstantIndexOp::create(builder, loc, trueCount);
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    return arith::SelectOp::create(builder, loc, predicate, count, zero);
  }

  Value compressDeviceOrdinalExcludingComponent(
      OpBuilder &builder, Location loc, Value deviceIndex,
      std::uint64_t excludedComponentIndex) const {
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    FailureOr<std::uint64_t> trailingSize = getTrailingComponentSize();
    assert(succeeded(componentSize) && succeeded(trailingSize) &&
           *componentSize > 1 &&
           "graph verification must reject invalid structured domains");
    std::uint64_t completeBlockSize = *componentSize * *trailingSize;
    std::uint64_t compressedBlockSize = (*componentSize - 1) * *trailingSize;
    Value completeBlock = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(completeBlockSize));
    Value compressedBlock = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(compressedBlockSize));
    Value trailing = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*trailingSize));
    Value outerIndex =
        arith::DivSIOp::create(builder, loc, deviceIndex, completeBlock);
    Value completeRemainder =
        arith::RemSIOp::create(builder, loc, deviceIndex, completeBlock);
    Value componentIndex =
        arith::DivSIOp::create(builder, loc, completeRemainder, trailing);
    Value suffix =
        arith::RemSIOp::create(builder, loc, completeRemainder, trailing);
    Value excluded = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(excludedComponentIndex));
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    Value isAfterExcluded = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, componentIndex, excluded);
    Value compressedAfter =
        arith::SubIOp::create(builder, loc, componentIndex, one);
    Value compressedComponent = arith::SelectOp::create(
        builder, loc, isAfterExcluded, compressedAfter, componentIndex);
    Value outerOffset =
        arith::MulIOp::create(builder, loc, outerIndex, compressedBlock);
    Value componentOffset =
        arith::MulIOp::create(builder, loc, compressedComponent, trailing);
    Value withComponent =
        arith::AddIOp::create(builder, loc, outerOffset, componentOffset);
    return arith::AddIOp::create(builder, loc, withComponent, suffix);
  }

  Value replaceComponentIndex(OpBuilder &builder, Location loc,
                              Value deviceIndex,
                              Value replacementComponentIndex) const {
    FailureOr<std::uint64_t> trailingSize = getTrailingComponentSize();
    assert(succeeded(trailingSize) &&
           "edge count validation must reject overflowing domain extents");
    Value trailing = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*trailingSize));
    Value currentComponentIndex = getComponentIndex(builder, loc, deviceIndex);
    Value currentComponentOffset =
        arith::MulIOp::create(builder, loc, currentComponentIndex, trailing);
    Value componentBase = arith::SubIOp::create(builder, loc, deviceIndex,
                                                currentComponentOffset);
    Value replacementOffset = arith::MulIOp::create(
        builder, loc, replacementComponentIndex, trailing);
    return arith::AddIOp::create(builder, loc, componentBase,
                                 replacementOffset);
  }

  Value
  expandComponentOrdinalExcluding(OpBuilder &builder, Location loc,
                                  Value compressedIndex,
                                  std::uint64_t excludedComponentIndex) const {
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    FailureOr<std::uint64_t> trailingSize = getTrailingComponentSize();
    assert(succeeded(componentSize) && succeeded(trailingSize) &&
           *componentSize > 1 &&
           "edge count validation must reject invalid structured domains");
    std::uint64_t compressedBlockSize = (*componentSize - 1) * *trailingSize;
    Value blockSize = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(compressedBlockSize));
    Value componentExtent = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*componentSize));
    Value trailing = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*trailingSize));
    Value outerIndex =
        arith::DivSIOp::create(builder, loc, compressedIndex, blockSize);
    Value blockRemainder =
        arith::RemSIOp::create(builder, loc, compressedIndex, blockSize);
    Value compressedComponentIndex =
        arith::DivSIOp::create(builder, loc, blockRemainder, trailing);
    Value suffix =
        arith::RemSIOp::create(builder, loc, blockRemainder, trailing);
    Value excluded = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(excludedComponentIndex));
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    Value componentAfterExcluded =
        arith::AddIOp::create(builder, loc, compressedComponentIndex, one);
    Value isAtOrAfterExcluded =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                              compressedComponentIndex, excluded);
    Value expandedComponentIndex = arith::SelectOp::create(
        builder, loc, isAtOrAfterExcluded, componentAfterExcluded,
        compressedComponentIndex);
    Value completeBlockSize =
        arith::MulIOp::create(builder, loc, componentExtent, trailing);
    Value outerOffset =
        arith::MulIOp::create(builder, loc, outerIndex, completeBlockSize);
    Value componentOffset =
        arith::MulIOp::create(builder, loc, expandedComponentIndex, trailing);
    Value withComponent =
        arith::AddIOp::create(builder, loc, outerOffset, componentOffset);
    return arith::AddIOp::create(builder, loc, withComponent, suffix);
  }

  Value
  expandComponentIndexExcluding(OpBuilder &builder, Location loc,
                                Value compressedComponentIndex,
                                std::uint64_t excludedComponentIndex) const {
    Value excluded = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(excludedComponentIndex));
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    Value afterExcluded =
        arith::AddIOp::create(builder, loc, compressedComponentIndex, one);
    Value isAtOrAfterExcluded =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                              compressedComponentIndex, excluded);
    return arith::SelectOp::create(builder, loc, isAtOrAfterExcluded,
                                   afterExcluded, compressedComponentIndex);
  }

  std::uint64_t getComponentLinearIndex(DenseI64ArrayAttr coordinates) const {
    std::uint64_t index = 0;
    for (auto [coordinate, extent] : llvm::zip_equal(
             coordinates.asArrayRef(), component.getExtent().asArrayRef())) {
      index = index * extent + coordinate;
    }
    return index;
  }

  MLIRContext *context;
  DeviceDomainAttr domain;
  std::size_t componentIndex;
  DeviceDomainComponentAttr component;
  mutable std::optional<SmallVector<DeviceRefAttr>> devices;
};

class AxisNeighborTransferGraph final : public StructuredTransferGraph {
public:
  using StructuredTransferGraph::StructuredTransferGraph;

  LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> emitError) const override {
    if (failed(verifyStructured(emitError))) {
      return failure();
    }
    DictionaryAttr properties = getProperties();
    IntegerAttr axis = properties.getAs<IntegerAttr>("axis");
    IntegerAttr offset = properties.getAs<IntegerAttr>("offset");
    BoolAttr wrap = properties.getAs<BoolAttr>("wrap");
    if (!axis || !offset || !wrap || properties.size() != 3) {
      return emitError() << "axis-neighbor transfer graph requires only axis, "
                            "offset, and wrap properties";
    }
    ArrayRef<int64_t> extent = component.getExtent().asArrayRef();
    if (axis.getInt() < 0 ||
        static_cast<std::size_t>(axis.getInt()) >= extent.size()) {
      return emitError() << "axis-neighbor axis " << axis.getInt()
                         << " is out of bounds for component rank "
                         << extent.size();
    }
    if (offset.getInt() <= 0) {
      return emitError() << "axis-neighbor offset must be positive";
    }
    int64_t axisExtent = extent[axis.getInt()];
    if ((!wrap.getValue() && offset.getInt() >= axisExtent) ||
        (wrap.getValue() && offset.getInt() % axisExtent == 0)) {
      return emitError()
             << "axis-neighbor transfer relation must contain a non-self edge";
    }
    return verifyNonemptyEdgeCount(emitError);
  }

  void forEachEdge(
      llvm::function_ref<void(TransferEdgeAttr)> callback) const override {
    int64_t axis = getProperties().getAs<IntegerAttr>("axis").getInt();
    int64_t offset = getProperties().getAs<IntegerAttr>("offset").getInt();
    bool wrap = getProperties().getAs<BoolAttr>("wrap").getValue();
    int64_t axisExtent = component.getExtent()[axis];
    if (wrap) {
      offset %= axisExtent;
    }
    for (DeviceRefAttr source : getDevices()) {
      SmallVector<DenseI64ArrayAttr> destinationCoordinates(
          source.getCoordinates());
      SmallVector<int64_t> destinationComponent(
          destinationCoordinates[componentIndex].asArrayRef());
      int64_t destinationAxis = destinationComponent[axis] + offset;
      if (destinationAxis >= axisExtent) {
        if (!wrap) {
          continue;
        }
        destinationAxis %= axisExtent;
      }
      destinationComponent[axis] = destinationAxis;
      destinationCoordinates[componentIndex] =
          DenseI64ArrayAttr::get(context, destinationComponent);
      callback(getPointTransferEdge(
          context, source,
          DeviceRefAttr::get(context, destinationCoordinates)));
    }
  }

  FailureOr<std::uint64_t> getEdgeCount() const override {
    int64_t axis = getProperties().getAs<IntegerAttr>("axis").getInt();
    int64_t offset = getProperties().getAs<IntegerAttr>("offset").getInt();
    bool wrap = getProperties().getAs<BoolAttr>("wrap").getValue();
    FailureOr<std::uint64_t> deviceCount = getDeviceCount();
    if (failed(deviceCount)) {
      return failure();
    }
    std::uint64_t axisExtent = component.getExtent()[axis];
    if (wrap) {
      return *deviceCount;
    }
    return (*deviceCount / axisExtent) * (axisExtent - offset);
  }

  TransferGraphEdgeIndexValues
  buildEdgeIndexValues(OpBuilder &builder, Location loc,
                       Value edgeIndex) const override {
    int64_t axis = getProperties().getAs<IntegerAttr>("axis").getInt();
    int64_t offset = getProperties().getAs<IntegerAttr>("offset").getInt();
    bool wrap = getProperties().getAs<BoolAttr>("wrap").getValue();
    std::uint64_t axisStride = 1;
    for (int64_t extent :
         component.getExtent().asArrayRef().drop_front(axis + 1)) {
      axisStride *= extent;
    }
    FailureOr<std::uint64_t> trailingSize = getTrailingComponentSize();
    assert(succeeded(trailingSize) &&
           "edge count validation must reject overflowing domain extents");
    axisStride *= *trailingSize;

    int64_t axisExtent = component.getExtent()[axis];
    if (wrap) {
      offset %= axisExtent;
    }
    Value stride = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(axisStride));
    Value extent = arith::ConstantIndexOp::create(builder, loc, axisExtent);
    Value source;
    if (wrap) {
      source = edgeIndex;
    } else {
      int64_t validAxisExtent = axisExtent - offset;
      Value validBlockSize = arith::ConstantIndexOp::create(
          builder, loc, validAxisExtent * static_cast<int64_t>(axisStride));
      Value completeBlockSize = arith::ConstantIndexOp::create(
          builder, loc, axisExtent * static_cast<int64_t>(axisStride));
      Value outerIndex =
          arith::DivSIOp::create(builder, loc, edgeIndex, validBlockSize);
      Value blockRemainder =
          arith::RemSIOp::create(builder, loc, edgeIndex, validBlockSize);
      Value outerOffset =
          arith::MulIOp::create(builder, loc, outerIndex, completeBlockSize);
      source = arith::AddIOp::create(builder, loc, outerOffset, blockRemainder);
    }

    Value sourceAxisWithInner =
        arith::DivSIOp::create(builder, loc, source, stride);
    Value sourceAxis =
        arith::RemSIOp::create(builder, loc, sourceAxisWithInner, extent);
    Value offsetValue = arith::ConstantIndexOp::create(builder, loc, offset);
    Value translatedAxis =
        arith::AddIOp::create(builder, loc, sourceAxis, offsetValue);
    if (wrap) {
      Value threshold =
          arith::ConstantIndexOp::create(builder, loc, axisExtent - offset);
      Value wraps = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sge, sourceAxis, threshold);
      Value wrapped =
          arith::SubIOp::create(builder, loc, sourceAxis, threshold);
      translatedAxis =
          arith::SelectOp::create(builder, loc, wraps, wrapped, translatedAxis);
    }
    Value sourceAxisOffset =
        arith::MulIOp::create(builder, loc, sourceAxis, stride);
    Value axisBase =
        arith::SubIOp::create(builder, loc, source, sourceAxisOffset);
    Value destinationAxisOffset =
        arith::MulIOp::create(builder, loc, translatedAxis, stride);
    Value destination =
        arith::AddIOp::create(builder, loc, axisBase, destinationAxisOffset);
    return {edgeIndex, source, destination};
  }

  Value buildIncidentEdgeCount(OpBuilder &builder, Location loc,
                               Value deviceIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    int64_t axis = getProperties().getAs<IntegerAttr>("axis").getInt();
    int64_t offset = getProperties().getAs<IntegerAttr>("offset").getInt();
    bool wrap = getProperties().getAs<BoolAttr>("wrap").getValue();
    int64_t axisExtent = component.getExtent()[axis];
    if (wrap) {
      return arith::ConstantIndexOp::create(builder, loc, 1);
    }
    Value coordinate =
        getComponentAxisCoordinate(builder, loc, deviceIndex, axis);
    Value boundary = arith::ConstantIndexOp::create(
        builder, loc, role == PipeRole::Source ? axisExtent - offset : offset);
    arith::CmpIPredicate predicate = role == PipeRole::Source
                                         ? arith::CmpIPredicate::slt
                                         : arith::CmpIPredicate::sge;
    Value hasIncidentEdge =
        arith::CmpIOp::create(builder, loc, predicate, coordinate, boundary);
    return buildCountFromPredicate(builder, loc, hasIncidentEdge);
  }

  TransferGraphEdgeIndexValues
  buildIncidentEdgeIndexValues(OpBuilder &builder, Location loc,
                               Value deviceIndex, Value incidentEdgeIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    int64_t axis = getProperties().getAs<IntegerAttr>("axis").getInt();
    int64_t offset = getProperties().getAs<IntegerAttr>("offset").getInt();
    bool wrap = getProperties().getAs<BoolAttr>("wrap").getValue();
    int64_t axisExtent = component.getExtent()[axis];
    if (wrap) {
      offset %= axisExtent;
    }

    Value endpointCoordinate =
        getComponentAxisCoordinate(builder, loc, deviceIndex, axis);
    Value source = deviceIndex;
    Value destination = deviceIndex;
    if (role == PipeRole::Source) {
      Value offsetValue = arith::ConstantIndexOp::create(builder, loc, offset);
      Value unwrapped =
          arith::AddIOp::create(builder, loc, endpointCoordinate, offsetValue);
      Value destinationCoordinate = unwrapped;
      if (wrap) {
        Value threshold =
            arith::ConstantIndexOp::create(builder, loc, axisExtent - offset);
        Value wraps =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                  endpointCoordinate, threshold);
        Value wrapped =
            arith::SubIOp::create(builder, loc, endpointCoordinate, threshold);
        destinationCoordinate =
            arith::SelectOp::create(builder, loc, wraps, wrapped, unwrapped);
      }
      destination = replaceComponentAxisCoordinate(builder, loc, deviceIndex,
                                                   axis, destinationCoordinate);
    } else {
      Value offsetValue = arith::ConstantIndexOp::create(builder, loc, offset);
      Value direct =
          arith::SubIOp::create(builder, loc, endpointCoordinate, offsetValue);
      Value sourceCoordinate = direct;
      if (wrap) {
        Value wraps =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                  endpointCoordinate, offsetValue);
        Value wrapDelta =
            arith::ConstantIndexOp::create(builder, loc, axisExtent - offset);
        Value wrapped =
            arith::AddIOp::create(builder, loc, endpointCoordinate, wrapDelta);
        sourceCoordinate =
            arith::SelectOp::create(builder, loc, wraps, wrapped, direct);
      }
      source = replaceComponentAxisCoordinate(builder, loc, deviceIndex, axis,
                                              sourceCoordinate);
    }
    return {buildEdgeOrdinalForSource(builder, loc, source), source,
            destination};
  }

private:
  Value buildEdgeOrdinalForSource(OpBuilder &builder, Location loc,
                                  Value source) const {
    int64_t axis = getProperties().getAs<IntegerAttr>("axis").getInt();
    int64_t offset = getProperties().getAs<IntegerAttr>("offset").getInt();
    bool wrap = getProperties().getAs<BoolAttr>("wrap").getValue();
    if (wrap) {
      return source;
    }
    std::uint64_t axisStride = 1;
    for (int64_t extent :
         component.getExtent().asArrayRef().drop_front(axis + 1)) {
      axisStride *= extent;
    }
    FailureOr<std::uint64_t> trailingSize = getTrailingComponentSize();
    assert(succeeded(trailingSize) &&
           "graph verification must reject overflowing domain extents");
    axisStride *= *trailingSize;
    int64_t axisExtent = component.getExtent()[axis];
    Value completeBlock = arith::ConstantIndexOp::create(
        builder, loc, axisExtent * static_cast<int64_t>(axisStride));
    Value validBlock = arith::ConstantIndexOp::create(
        builder, loc, (axisExtent - offset) * static_cast<int64_t>(axisStride));
    Value outerIndex =
        arith::DivSIOp::create(builder, loc, source, completeBlock);
    Value blockRemainder =
        arith::RemSIOp::create(builder, loc, source, completeBlock);
    Value outerOffset =
        arith::MulIOp::create(builder, loc, outerIndex, validBlock);
    return arith::AddIOp::create(builder, loc, outerOffset, blockRemainder);
  }
};

class StencilTransferGraph final : public StructuredTransferGraph {
public:
  using StructuredTransferGraph::StructuredTransferGraph;

  LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> emitError) const override {
    if (failed(verifyStructured(emitError))) {
      return failure();
    }
    DictionaryAttr properties = getProperties();
    ArrayAttr offsets = properties.getAs<ArrayAttr>("offsets");
    BoolAttr wrap = properties.getAs<BoolAttr>("wrap");
    if (!offsets || offsets.empty() || !wrap || properties.size() != 2) {
      return emitError()
             << "stencil transfer graph requires only nonempty offsets and "
                "wrap properties";
    }

    llvm::DenseSet<DenseI64ArrayAttr> effectiveOffsets;
    bool hasEdge = false;
    ArrayRef<int64_t> extent = component.getExtent().asArrayRef();
    for (auto [offsetIndex, offsetAttribute] : llvm::enumerate(offsets)) {
      auto offset = mlir::dyn_cast<DenseI64ArrayAttr>(offsetAttribute);
      if (!offset || static_cast<std::size_t>(offset.size()) != extent.size()) {
        return emitError() << "stencil offset " << offsetIndex
                           << " must have component rank " << extent.size();
      }
      if (llvm::all_of(offset.asArrayRef(),
                       [](int64_t delta) { return delta == 0; })) {
        return emitError() << "stencil offset " << offsetIndex
                           << " must not be zero";
      }

      SmallVector<int64_t> effectiveOffset(offset.asArrayRef());
      bool offsetHasEdge = true;
      for (auto [axis, delta] : llvm::enumerate(effectiveOffset)) {
        if (wrap.getValue()) {
          delta %= extent[axis];
          if (delta < 0) {
            delta += extent[axis];
          }
          effectiveOffset[axis] = delta;
        } else if (delta <= -extent[axis] || delta >= extent[axis]) {
          offsetHasEdge = false;
        }
      }
      if (wrap.getValue() && llvm::all_of(effectiveOffset, [](int64_t delta) {
            return delta == 0;
          })) {
        return emitError() << "stencil offset " << offsetIndex
                           << " produces only self edges after wrapping";
      }
      DenseI64ArrayAttr effectiveOffsetAttr =
          DenseI64ArrayAttr::get(context, effectiveOffset);
      if (!effectiveOffsets.insert(effectiveOffsetAttr).second) {
        return emitError() << "stencil offset " << offsetIndex
                           << " duplicates an earlier effective offset";
      }
      hasEdge |= offsetHasEdge;
    }
    if (!hasEdge) {
      return emitError() << "stencil transfer relation contains no edges";
    }
    return verifyNonemptyEdgeCount(emitError);
  }

  void forEachEdge(
      llvm::function_ref<void(TransferEdgeAttr)> callback) const override {
    SmallVector<StencilOffsetDescriptor> descriptors = getDescriptors();
    for (DeviceRefAttr source : getDevices()) {
      for (const StencilOffsetDescriptor &descriptor : descriptors) {
        std::optional<DeviceRefAttr> destination = translateComponent(
            source, descriptor.offset.asArrayRef(), /*direction=*/1);
        if (destination) {
          callback(getPointTransferEdge(context, source, *destination));
        }
      }
    }
  }

  FailureOr<std::uint64_t> getEdgeCount() const override {
    FailureOr<std::uint64_t> deviceCount = getDeviceCount();
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    FailureOr<SmallVector<StencilOffsetDescriptor>> descriptors =
        getMaybeDescriptors();
    if (failed(deviceCount) || failed(componentSize) || failed(descriptors)) {
      return failure();
    }
    std::uint64_t surroundingContextCount = *deviceCount / *componentSize;
    std::uint64_t edgeCount = 0;
    for (const StencilOffsetDescriptor &descriptor : *descriptors) {
      std::optional<std::uint64_t> offsetEdgeCount = llvm::checkedMulUnsigned(
          surroundingContextCount, descriptor.sourceCount);
      if (!offsetEdgeCount) {
        return failure();
      }
      std::optional<std::uint64_t> updatedEdgeCount =
          llvm::checkedAddUnsigned(edgeCount, *offsetEdgeCount);
      if (!updatedEdgeCount) {
        return failure();
      }
      edgeCount = *updatedEdgeCount;
    }
    return edgeCount;
  }

  TransferGraphEdgeIndexValues
  buildEdgeIndexValues(OpBuilder &builder, Location loc,
                       Value edgeIndex) const override {
    SmallVector<int64_t> sourceEdgeOffsets = getSourceEdgeOffsets();
    Value source = arith::ConstantIndexOp::create(builder, loc, 0);
    Value sourceEdgeBase = arith::ConstantIndexOp::create(builder, loc, 0);
    for (std::size_t sourceIndex = 0; sourceIndex < getDevices().size();
         ++sourceIndex) {
      Value start = arith::ConstantIndexOp::create(
          builder, loc, sourceEdgeOffsets[sourceIndex]);
      Value end = arith::ConstantIndexOp::create(
          builder, loc, sourceEdgeOffsets[sourceIndex + 1]);
      Value afterStart = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sge, edgeIndex, start);
      Value beforeEnd = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::slt, edgeIndex, end);
      Value selectsSource =
          arith::AndIOp::create(builder, loc, afterStart, beforeEnd);
      Value sourceIndexValue = arith::ConstantIndexOp::create(
          builder, loc, static_cast<int64_t>(sourceIndex));
      source = arith::SelectOp::create(builder, loc, selectsSource,
                                       sourceIndexValue, source);
      sourceEdgeBase = arith::SelectOp::create(builder, loc, selectsSource,
                                               start, sourceEdgeBase);
    }

    Value sourceLocalEdgeIndex =
        arith::SubIOp::create(builder, loc, edgeIndex, sourceEdgeBase);
    Value destination = source;
    Value validOffsetCount = arith::ConstantIndexOp::create(builder, loc, 0);
    for (auto [descriptorIndex, descriptor] :
         llvm::enumerate(getDescriptors())) {
      DynamicStencilEdge candidate = buildIncidentCandidate(
          builder, loc, source, PipeRole::Source, descriptor, descriptorIndex);
      Value hasSelectedOrdinal =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                sourceLocalEdgeIndex, validOffsetCount);
      Value selectsCandidate = arith::AndIOp::create(
          builder, loc, candidate.isValid, hasSelectedOrdinal);
      destination = arith::SelectOp::create(builder, loc, selectsCandidate,
                                            candidate.destination, destination);
      Value contribution =
          buildCountFromPredicate(builder, loc, candidate.isValid);
      validOffsetCount =
          arith::AddIOp::create(builder, loc, validOffsetCount, contribution);
    }
    return {edgeIndex, source, destination};
  }

  Value buildIncidentEdgeCount(OpBuilder &builder, Location loc,
                               Value deviceIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    Value count = arith::ConstantIndexOp::create(builder, loc, 0);
    for (const StencilOffsetDescriptor &descriptor : getDescriptors()) {
      Value isValid =
          buildOffsetValidity(builder, loc, deviceIndex, role, descriptor);
      Value contribution = buildCountFromPredicate(builder, loc, isValid);
      count = arith::AddIOp::create(builder, loc, count, contribution);
    }
    return count;
  }

  TransferGraphEdgeIndexValues
  buildIncidentEdgeIndexValues(OpBuilder &builder, Location loc,
                               Value deviceIndex, Value incidentEdgeIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    Value selectedOrdinal = arith::ConstantIndexOp::create(builder, loc, 0);
    Value selectedSource = deviceIndex;
    Value selectedDestination = deviceIndex;
    Value validPrefix = arith::ConstantIndexOp::create(builder, loc, 0);
    SmallVector<StencilOffsetDescriptor> descriptors = getDescriptors();
    SmallVector<DynamicStencilEdge> candidates;
    candidates.reserve(descriptors.size());
    for (auto [descriptorIndex, descriptor] : llvm::enumerate(descriptors)) {
      candidates.push_back(buildIncidentCandidate(
          builder, loc, deviceIndex, role, descriptor, descriptorIndex));
    }
    for (const DynamicStencilEdge &candidate : candidates) {
      Value localEdgeIndex = validPrefix;
      if (role == PipeRole::Destination) {
        // Global stencil records are source-major, so a destination orders
        // incoming candidates by their source device rather than offset order.
        localEdgeIndex = arith::ConstantIndexOp::create(builder, loc, 0);
        for (const DynamicStencilEdge &otherCandidate : candidates) {
          Value sourcePrecedes =
              arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                    otherCandidate.source, candidate.source);
          Value precedingCandidate = arith::AndIOp::create(
              builder, loc, otherCandidate.isValid, sourcePrecedes);
          Value contribution =
              buildCountFromPredicate(builder, loc, precedingCandidate);
          localEdgeIndex =
              arith::AddIOp::create(builder, loc, localEdgeIndex, contribution);
        }
      }
      Value hasOrdinal =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                incidentEdgeIndex, localEdgeIndex);
      Value selectsCandidate =
          arith::AndIOp::create(builder, loc, candidate.isValid, hasOrdinal);
      selectedOrdinal =
          arith::SelectOp::create(builder, loc, selectsCandidate,
                                  candidate.edgeOrdinal, selectedOrdinal);
      selectedSource = arith::SelectOp::create(
          builder, loc, selectsCandidate, candidate.source, selectedSource);
      selectedDestination =
          arith::SelectOp::create(builder, loc, selectsCandidate,
                                  candidate.destination, selectedDestination);
      Value contribution =
          buildCountFromPredicate(builder, loc, candidate.isValid);
      if (role == PipeRole::Source) {
        validPrefix =
            arith::AddIOp::create(builder, loc, validPrefix, contribution);
      }
    }
    return {selectedOrdinal, selectedSource, selectedDestination};
  }

private:
  struct DynamicStencilEdge {
    Value isValid;
    Value edgeOrdinal;
    Value source;
    Value destination;
  };

  DynamicStencilEdge
  buildIncidentCandidate(OpBuilder &builder, Location loc, Value endpointDevice,
                         PipeRole role,
                         const StencilOffsetDescriptor &descriptor,
                         std::size_t descriptorIndex) const {
    bool wrap = getProperties().getAs<BoolAttr>("wrap").getValue();
    SmallVector<Value> endpointCoordinates =
        getComponentCoordinatesFromDevice(builder, loc, endpointDevice);
    SmallVector<Value> sourceCoordinates;
    SmallVector<Value> destinationCoordinates;
    sourceCoordinates.reserve(endpointCoordinates.size());
    destinationCoordinates.reserve(endpointCoordinates.size());
    Value isValid =
        buildOffsetValidity(builder, loc, endpointDevice, role, descriptor);

    for (auto [endpointCoordinate, delta, extent] :
         llvm::zip_equal(endpointCoordinates, descriptor.offset.asArrayRef(),
                         component.getExtent().asArrayRef())) {
      Value translatedCoordinate = endpointCoordinate;
      if (wrap) {
        assert(delta >= 0 && delta < extent &&
               "verified wrapped stencil offset must be normalized");
        if (delta != 0) {
          Value deltaValue =
              arith::ConstantIndexOp::create(builder, loc, delta);
          if (role == PipeRole::Source) {
            Value threshold =
                arith::ConstantIndexOp::create(builder, loc, extent - delta);
            Value wraps =
                arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                      endpointCoordinate, threshold);
            Value wrapped = arith::SubIOp::create(
                builder, loc, endpointCoordinate, threshold);
            Value unwrapped = arith::AddIOp::create(
                builder, loc, endpointCoordinate, deltaValue);
            translatedCoordinate = arith::SelectOp::create(builder, loc, wraps,
                                                           wrapped, unwrapped);
          } else {
            Value wraps =
                arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                      endpointCoordinate, deltaValue);
            Value wrapDelta =
                arith::ConstantIndexOp::create(builder, loc, extent - delta);
            Value wrapped = arith::AddIOp::create(
                builder, loc, endpointCoordinate, wrapDelta);
            Value unwrapped = arith::SubIOp::create(
                builder, loc, endpointCoordinate, deltaValue);
            translatedCoordinate = arith::SelectOp::create(builder, loc, wraps,
                                                           wrapped, unwrapped);
          }
        }
      } else {
        int64_t directedDelta = role == PipeRole::Source ? delta : -delta;
        if (directedDelta > 0) {
          Value deltaValue =
              arith::ConstantIndexOp::create(builder, loc, directedDelta);
          translatedCoordinate = arith::AddIOp::create(
              builder, loc, endpointCoordinate, deltaValue);
        } else if (directedDelta < 0) {
          Value magnitude =
              arith::ConstantIndexOp::create(builder, loc, -directedDelta);
          translatedCoordinate = arith::SubIOp::create(
              builder, loc, endpointCoordinate, magnitude);
        }
      }

      if (role == PipeRole::Source) {
        sourceCoordinates.push_back(endpointCoordinate);
        destinationCoordinates.push_back(translatedCoordinate);
      } else {
        sourceCoordinates.push_back(translatedCoordinate);
        destinationCoordinates.push_back(endpointCoordinate);
      }
    }

    Value sourceComponentIndex =
        buildComponentLinearIndex(builder, loc, sourceCoordinates);
    Value destinationComponentIndex =
        buildComponentLinearIndex(builder, loc, destinationCoordinates);
    Value source = replaceComponentIndex(builder, loc, endpointDevice,
                                         sourceComponentIndex);
    Value destination = replaceComponentIndex(builder, loc, endpointDevice,
                                              destinationComponentIndex);
    Value edgeOrdinal =
        buildEdgeOrdinalForSource(builder, loc, source, descriptorIndex);
    return {isValid, edgeOrdinal, source, destination};
  }

  Value buildOffsetValidity(OpBuilder &builder, Location loc,
                            Value endpointDevice, PipeRole role,
                            const StencilOffsetDescriptor &descriptor) const {
    if (getProperties().getAs<BoolAttr>("wrap").getValue()) {
      return arith::ConstantIntOp::create(builder, loc, 1, 1);
    }
    SmallVector<Value> endpointCoordinates =
        getComponentCoordinatesFromDevice(builder, loc, endpointDevice);
    Value isValid = arith::ConstantIntOp::create(builder, loc, 1, 1);
    for (auto [endpointCoordinate, delta, extent] :
         llvm::zip_equal(endpointCoordinates, descriptor.offset.asArrayRef(),
                         component.getExtent().asArrayRef())) {
      int64_t directedDelta = role == PipeRole::Source ? delta : -delta;
      if (directedDelta > 0) {
        Value boundary = arith::ConstantIndexOp::create(builder, loc,
                                                        extent - directedDelta);
        Value axisValid =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                  endpointCoordinate, boundary);
        isValid = arith::AndIOp::create(builder, loc, isValid, axisValid);
      } else if (directedDelta < 0) {
        Value magnitude =
            arith::ConstantIndexOp::create(builder, loc, -directedDelta);
        Value axisValid =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                  endpointCoordinate, magnitude);
        isValid = arith::AndIOp::create(builder, loc, isValid, axisValid);
      }
    }
    return isValid;
  }

  SmallVector<int64_t> getSourceEdgeOffsets() const {
    // Source-prefix counts preserve the established source-major callback
    // order without storing one table row per edge.
    SmallVector<StencilOffsetDescriptor> descriptors = getDescriptors();
    SmallVector<int64_t> offsets;
    offsets.reserve(getDevices().size() + 1);
    int64_t edgeCount = 0;
    for (DeviceRefAttr source : getDevices()) {
      offsets.push_back(edgeCount);
      for (const StencilOffsetDescriptor &descriptor : descriptors) {
        if (translateComponent(source, descriptor.offset.asArrayRef(),
                               /*direction=*/1)) {
          ++edgeCount;
        }
      }
    }
    offsets.push_back(edgeCount);
    return offsets;
  }

  Value buildEdgeOrdinalForSource(OpBuilder &builder, Location loc,
                                  Value source,
                                  std::size_t descriptorIndex) const {
    Value sourceEdgeBase =
        buildIndexTableLookup(builder, loc, getSourceEdgeOffsets(), source);
    Value validOffsetCount = arith::ConstantIndexOp::create(builder, loc, 0);
    SmallVector<StencilOffsetDescriptor> descriptors = getDescriptors();
    for (const StencilOffsetDescriptor &earlierDescriptor :
         ArrayRef<StencilOffsetDescriptor>(descriptors)
             .take_front(descriptorIndex)) {
      Value isValid = buildOffsetValidity(builder, loc, source,
                                          PipeRole::Source, earlierDescriptor);
      Value contribution = buildCountFromPredicate(builder, loc, isValid);
      validOffsetCount =
          arith::AddIOp::create(builder, loc, validOffsetCount, contribution);
    }
    return arith::AddIOp::create(builder, loc, sourceEdgeBase,
                                 validOffsetCount);
  }

  FailureOr<SmallVector<StencilOffsetDescriptor>> getMaybeDescriptors() const {
    return getStencilOffsetDescriptors(
        component, getProperties().getAs<ArrayAttr>("offsets"),
        getProperties().getAs<BoolAttr>("wrap").getValue());
  }

  SmallVector<StencilOffsetDescriptor> getDescriptors() const {
    FailureOr<SmallVector<StencilOffsetDescriptor>> descriptors =
        getMaybeDescriptors();
    assert(succeeded(descriptors) &&
           "graph verification must reject overflowing stencil offsets");
    return std::move(*descriptors);
  }

  std::optional<DeviceRefAttr> translateComponent(DeviceRefAttr device,
                                                  ArrayRef<int64_t> offset,
                                                  int64_t direction) const {
    bool wrap = getProperties().getAs<BoolAttr>("wrap").getValue();
    SmallVector<int64_t> translatedCoordinates(
        device.getCoordinates()[componentIndex].asArrayRef());
    for (auto [axis, coordinate, delta, extent] :
         llvm::enumerate(translatedCoordinates, offset,
                         component.getExtent().asArrayRef())) {
      if (wrap) {
        int64_t normalizedDelta = delta % extent;
        if (normalizedDelta < 0) {
          normalizedDelta += extent;
        }
        if (direction > 0) {
          int64_t threshold = extent - normalizedDelta;
          translatedCoordinates[axis] = coordinate >= threshold
                                            ? coordinate - threshold
                                            : coordinate + normalizedDelta;
        } else {
          translatedCoordinates[axis] =
              coordinate < normalizedDelta
                  ? coordinate + (extent - normalizedDelta)
                  : coordinate - normalizedDelta;
        }
        continue;
      }
      int64_t translated = coordinate + direction * delta;
      if (translated < 0 || translated >= extent) {
        return std::nullopt;
      }
      translatedCoordinates[axis] = translated;
    }
    return replaceComponent(
        device, DenseI64ArrayAttr::get(context, translatedCoordinates));
  }
};

class GatherTransferGraph final : public StructuredTransferGraph {
public:
  using StructuredTransferGraph::StructuredTransferGraph;

  LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> emitError) const override {
    if (failed(verifyStructured(emitError))) {
      return failure();
    }
    DictionaryAttr properties = getProperties();
    DeviceRefAttr root = properties.getAs<DeviceRefAttr>("root");
    if (!root || properties.size() != 1 || root.getCoordinates().size() != 1) {
      return emitError()
             << "gather transfer graph requires only one component-local root "
                "property";
    }
    if (failed(verifyComponentCoordinates(component, root.getCoordinates()[0],
                                          emitError, "root"))) {
      return failure();
    }
    return verifyNonemptyEdgeCount(emitError);
  }

  void forEachEdge(
      llvm::function_ref<void(TransferEdgeAttr)> callback) const override {
    DeviceRefAttr endpoint = getProperties().getAs<DeviceRefAttr>("root");
    DenseI64ArrayAttr endpointCoordinates = endpoint.getCoordinates().front();
    for (DeviceRefAttr source : getDevices()) {
      DeviceRefAttr destination = replaceComponent(source, endpointCoordinates);
      if (source != destination) {
        callback(getPointTransferEdge(context, source, destination));
      }
    }
  }

  FailureOr<std::uint64_t> getEdgeCount() const override {
    FailureOr<std::uint64_t> deviceCount = getDeviceCount();
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    if (failed(deviceCount) || failed(componentSize)) {
      return failure();
    }
    return (*deviceCount / *componentSize) * (*componentSize - 1);
  }

  TransferGraphEdgeIndexValues
  buildEdgeIndexValues(OpBuilder &builder, Location loc,
                       Value edgeIndex) const override {
    DeviceRefAttr root = getProperties().getAs<DeviceRefAttr>("root");
    std::uint64_t rootIndex =
        getComponentLinearIndex(root.getCoordinates().front());
    Value source =
        expandComponentOrdinalExcluding(builder, loc, edgeIndex, rootIndex);
    Value rootValue = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(rootIndex));
    Value destination = replaceComponentIndex(builder, loc, source, rootValue);
    return {edgeIndex, source, destination};
  }

  Value buildIncidentEdgeCount(OpBuilder &builder, Location loc,
                               Value deviceIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    DeviceRefAttr root = getProperties().getAs<DeviceRefAttr>("root");
    std::uint64_t rootIndex =
        getComponentLinearIndex(root.getCoordinates().front());
    Value componentIndex = getComponentIndex(builder, loc, deviceIndex);
    Value rootValue = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(rootIndex));
    arith::CmpIPredicate predicate = role == PipeRole::Source
                                         ? arith::CmpIPredicate::ne
                                         : arith::CmpIPredicate::eq;
    Value hasIncidentEdges = arith::CmpIOp::create(builder, loc, predicate,
                                                   componentIndex, rootValue);
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    assert(succeeded(componentSize) && *componentSize > 1 &&
           "graph verification must reject an empty gather relation");
    return buildCountFromPredicate(
        builder, loc, hasIncidentEdges,
        role == PipeRole::Source ? 1
                                 : static_cast<int64_t>(*componentSize - 1));
  }

  TransferGraphEdgeIndexValues
  buildIncidentEdgeIndexValues(OpBuilder &builder, Location loc,
                               Value deviceIndex, Value incidentEdgeIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    DeviceRefAttr root = getProperties().getAs<DeviceRefAttr>("root");
    std::uint64_t rootIndex =
        getComponentLinearIndex(root.getCoordinates().front());
    Value rootValue = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(rootIndex));
    Value source = deviceIndex;
    Value destination = deviceIndex;
    if (role == PipeRole::Source) {
      destination = replaceComponentIndex(builder, loc, deviceIndex, rootValue);
    } else {
      Value sourceComponent = expandComponentIndexExcluding(
          builder, loc, incidentEdgeIndex, rootIndex);
      source =
          replaceComponentIndex(builder, loc, deviceIndex, sourceComponent);
    }
    Value edgeOrdinal = compressDeviceOrdinalExcludingComponent(
        builder, loc, source, rootIndex);
    return {edgeOrdinal, source, destination};
  }
};

class ScatterTransferGraph final : public StructuredTransferGraph {
public:
  using StructuredTransferGraph::StructuredTransferGraph;

  LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> emitError) const override {
    if (failed(verifyStructured(emitError))) {
      return failure();
    }
    DictionaryAttr properties = getProperties();
    DeviceRefAttr source = properties.getAs<DeviceRefAttr>("source");
    if (!source || properties.size() != 1 ||
        source.getCoordinates().size() != 1) {
      return emitError()
             << "scatter transfer graph requires only one component-local "
                "source property";
    }
    if (failed(verifyComponentCoordinates(component, source.getCoordinates()[0],
                                          emitError, "source"))) {
      return failure();
    }
    return verifyNonemptyEdgeCount(emitError);
  }

  void forEachEdge(
      llvm::function_ref<void(TransferEdgeAttr)> callback) const override {
    DeviceRefAttr endpoint = getProperties().getAs<DeviceRefAttr>("source");
    DenseI64ArrayAttr endpointCoordinates = endpoint.getCoordinates().front();
    for (DeviceRefAttr destination : getDevices()) {
      DeviceRefAttr source = replaceComponent(destination, endpointCoordinates);
      if (source != destination) {
        callback(getPointTransferEdge(context, source, destination));
      }
    }
  }

  FailureOr<std::uint64_t> getEdgeCount() const override {
    FailureOr<std::uint64_t> deviceCount = getDeviceCount();
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    if (failed(deviceCount) || failed(componentSize)) {
      return failure();
    }
    return (*deviceCount / *componentSize) * (*componentSize - 1);
  }

  TransferGraphEdgeIndexValues
  buildEdgeIndexValues(OpBuilder &builder, Location loc,
                       Value edgeIndex) const override {
    DeviceRefAttr sourceEndpoint =
        getProperties().getAs<DeviceRefAttr>("source");
    std::uint64_t sourceIndex =
        getComponentLinearIndex(sourceEndpoint.getCoordinates().front());
    Value destination =
        expandComponentOrdinalExcluding(builder, loc, edgeIndex, sourceIndex);
    Value sourceValue = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(sourceIndex));
    Value source =
        replaceComponentIndex(builder, loc, destination, sourceValue);
    return {edgeIndex, source, destination};
  }

  Value buildIncidentEdgeCount(OpBuilder &builder, Location loc,
                               Value deviceIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    DeviceRefAttr sourceEndpoint =
        getProperties().getAs<DeviceRefAttr>("source");
    std::uint64_t sourceIndex =
        getComponentLinearIndex(sourceEndpoint.getCoordinates().front());
    Value componentIndex = getComponentIndex(builder, loc, deviceIndex);
    Value sourceValue = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(sourceIndex));
    arith::CmpIPredicate predicate = role == PipeRole::Source
                                         ? arith::CmpIPredicate::eq
                                         : arith::CmpIPredicate::ne;
    Value hasIncidentEdges = arith::CmpIOp::create(builder, loc, predicate,
                                                   componentIndex, sourceValue);
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    assert(succeeded(componentSize) && *componentSize > 1 &&
           "graph verification must reject an empty scatter relation");
    return buildCountFromPredicate(
        builder, loc, hasIncidentEdges,
        role == PipeRole::Source ? static_cast<int64_t>(*componentSize - 1)
                                 : 1);
  }

  TransferGraphEdgeIndexValues
  buildIncidentEdgeIndexValues(OpBuilder &builder, Location loc,
                               Value deviceIndex, Value incidentEdgeIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    DeviceRefAttr sourceEndpoint =
        getProperties().getAs<DeviceRefAttr>("source");
    std::uint64_t sourceIndex =
        getComponentLinearIndex(sourceEndpoint.getCoordinates().front());
    Value sourceValue = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(sourceIndex));
    Value source = deviceIndex;
    Value destination = deviceIndex;
    if (role == PipeRole::Source) {
      Value destinationComponent = expandComponentIndexExcluding(
          builder, loc, incidentEdgeIndex, sourceIndex);
      destination = replaceComponentIndex(builder, loc, deviceIndex,
                                          destinationComponent);
    } else {
      source = replaceComponentIndex(builder, loc, deviceIndex, sourceValue);
    }
    Value edgeOrdinal = compressDeviceOrdinalExcludingComponent(
        builder, loc, destination, sourceIndex);
    return {edgeOrdinal, source, destination};
  }
};

class AllToAllTransferGraph final : public StructuredTransferGraph {
public:
  using StructuredTransferGraph::StructuredTransferGraph;

  LogicalResult
  verify(llvm::function_ref<InFlightDiagnostic()> emitError) const override {
    if (failed(verifyStructured(emitError))) {
      return failure();
    }
    if (!getProperties().empty()) {
      return emitError()
             << "all-to-all transfer graph does not accept properties";
    }
    return verifyNonemptyEdgeCount(emitError);
  }

  void forEachEdge(
      llvm::function_ref<void(TransferEdgeAttr)> callback) const override {
    SmallVector<DenseI64ArrayAttr> destinationComponents;
    SmallVector<int64_t> coordinates;
    enumerateComponentCoordinates(context, component.getExtent().asArrayRef(),
                                  0, coordinates, destinationComponents);
    for (DeviceRefAttr source : getDevices()) {
      for (DenseI64ArrayAttr destinationComponent : destinationComponents) {
        if (destinationComponent == source.getCoordinates()[componentIndex]) {
          continue;
        }
        SmallVector<DenseI64ArrayAttr> destinationCoordinates(
            source.getCoordinates());
        destinationCoordinates[componentIndex] = destinationComponent;
        callback(getPointTransferEdge(
            context, source,
            DeviceRefAttr::get(context, destinationCoordinates)));
      }
    }
  }

  FailureOr<std::uint64_t> getEdgeCount() const override {
    FailureOr<std::uint64_t> deviceCount = getDeviceCount();
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    if (failed(deviceCount) || failed(componentSize)) {
      return failure();
    }
    std::optional<std::uint64_t> edgeCount =
        llvm::checkedMulUnsigned(*deviceCount, *componentSize - 1);
    if (!edgeCount) {
      return failure();
    }
    return *edgeCount;
  }

  TransferGraphEdgeIndexValues
  buildEdgeIndexValues(OpBuilder &builder, Location loc,
                       Value edgeIndex) const override {
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    assert(succeeded(componentSize) && *componentSize > 1 &&
           "edge count validation must reject invalid all-to-all domains");
    Value peersPerSource = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*componentSize - 1));
    Value source =
        arith::DivSIOp::create(builder, loc, edgeIndex, peersPerSource);
    Value compressedDestination =
        arith::RemSIOp::create(builder, loc, edgeIndex, peersPerSource);
    Value sourceComponent = getComponentIndex(builder, loc, source);
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    Value destinationAfterSource =
        arith::AddIOp::create(builder, loc, compressedDestination, one);
    Value isAtOrAfterSource =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                              compressedDestination, sourceComponent);
    Value destinationComponent =
        arith::SelectOp::create(builder, loc, isAtOrAfterSource,
                                destinationAfterSource, compressedDestination);
    Value destination =
        replaceComponentIndex(builder, loc, source, destinationComponent);
    return {edgeIndex, source, destination};
  }

  Value buildIncidentEdgeCount(OpBuilder &builder, Location loc,
                               Value deviceIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    assert(succeeded(componentSize) && *componentSize > 1 &&
           "graph verification must reject an empty all-to-all relation");
    return arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*componentSize - 1));
  }

  TransferGraphEdgeIndexValues
  buildIncidentEdgeIndexValues(OpBuilder &builder, Location loc,
                               Value deviceIndex, Value incidentEdgeIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    assert(succeeded(componentSize) && *componentSize > 1 &&
           "graph verification must reject an empty all-to-all relation");
    Value source = deviceIndex;
    Value destination = deviceIndex;
    Value compressedDestination = incidentEdgeIndex;
    if (role == PipeRole::Source) {
      Value sourceComponent = getComponentIndex(builder, loc, deviceIndex);
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      Value afterSource =
          arith::AddIOp::create(builder, loc, incidentEdgeIndex, one);
      Value isAtOrAfterSource =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                incidentEdgeIndex, sourceComponent);
      Value destinationComponent = arith::SelectOp::create(
          builder, loc, isAtOrAfterSource, afterSource, incidentEdgeIndex);
      destination = replaceComponentIndex(builder, loc, deviceIndex,
                                          destinationComponent);
    } else {
      Value destinationComponent = getComponentIndex(builder, loc, deviceIndex);
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      Value afterDestination =
          arith::AddIOp::create(builder, loc, incidentEdgeIndex, one);
      Value isAtOrAfterDestination =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                incidentEdgeIndex, destinationComponent);
      Value sourceComponent =
          arith::SelectOp::create(builder, loc, isAtOrAfterDestination,
                                  afterDestination, incidentEdgeIndex);
      source =
          replaceComponentIndex(builder, loc, deviceIndex, sourceComponent);
      Value destinationAfterSource =
          arith::SubIOp::create(builder, loc, destinationComponent, one);
      Value destinationIsAfterSource =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt,
                                destinationComponent, sourceComponent);
      compressedDestination =
          arith::SelectOp::create(builder, loc, destinationIsAfterSource,
                                  destinationAfterSource, destinationComponent);
    }
    Value peersPerSource = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*componentSize - 1));
    Value sourceBlock =
        arith::MulIOp::create(builder, loc, source, peersPerSource);
    Value edgeOrdinal =
        arith::AddIOp::create(builder, loc, sourceBlock, compressedDestination);
    return {edgeOrdinal, source, destination};
  }
};

} // namespace

LogicalResult TransferGraph::verifyNonemptyEdgeCount(
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  FailureOr<std::uint64_t> edgeCount = getEdgeCount();
  if (failed(edgeCount) ||
      *edgeCount >
          static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
    return emitError()
           << "transfer graph edge count exceeds the supported index range";
  }
  if (*edgeCount == 0) {
    return emitError() << "transfer graph relation contains no edges";
  }
  return success();
}

SmallVector<TransferEdgeAttr> TransferGraph::getEdges() const {
  SmallVector<TransferEdgeAttr> edges;
  forEachEdge([&](TransferEdgeAttr edge) { edges.push_back(edge); });
  return edges;
}

std::unique_ptr<TransferGraph> createTransferGraph(TransferGraphAttr graph) {
  return createTransferGraph(graph.getDomain(), graph.getKind(),
                             graph.getComponentName(), graph.getProperties());
}

std::unique_ptr<TransferGraph> createTransferGraph(DeviceDomainAttr domain,
                                                   TransferGraphKind kind,
                                                   StringAttr componentName,
                                                   DictionaryAttr properties) {
  switch (kind) {
  case TransferGraphKind::Explicit:
    return std::make_unique<ExplicitTransferGraph>(domain, kind, componentName,
                                                   properties);
  case TransferGraphKind::AxisNeighbor:
    return std::make_unique<AxisNeighborTransferGraph>(
        domain, kind, componentName, properties);
  case TransferGraphKind::Stencil:
    return std::make_unique<StencilTransferGraph>(domain, kind, componentName,
                                                  properties);
  case TransferGraphKind::Gather:
    return std::make_unique<GatherTransferGraph>(domain, kind, componentName,
                                                 properties);
  case TransferGraphKind::Scatter:
    return std::make_unique<ScatterTransferGraph>(domain, kind, componentName,
                                                  properties);
  case TransferGraphKind::AllToAll:
    return std::make_unique<AllToAllTransferGraph>(domain, kind, componentName,
                                                   properties);
  }
  llvm_unreachable("unknown transfer graph kind");
}

FailureOr<std::uint64_t> getPipeRecordCount(PipeNetRecordsAttr records) {
  if (records.getMappings().empty()) {
    return records.getPipes().size();
  }

  std::uint64_t recordCount = 0;
  for (PipeMappingAttr mapping : records.getMappings()) {
    FailureOr<std::uint64_t> edgeCount =
        createTransferGraph(mapping.getGraph())->getEdgeCount();
    std::optional<std::uint64_t> mappingRecordCount =
        succeeded(edgeCount)
            ? llvm::checkedMulUnsigned(
                  *edgeCount,
                  static_cast<std::uint64_t>(mapping.getPipes().size()))
            : std::nullopt;
    std::optional<std::uint64_t> updatedRecordCount =
        mappingRecordCount
            ? llvm::checkedAddUnsigned(recordCount, *mappingRecordCount)
            : std::nullopt;
    if (!updatedRecordCount) {
      return failure();
    }
    recordCount = *updatedRecordCount;
  }
  return recordCount;
}

void forEachNodePipeRecord(PipeNetRecordsAttr records,
                           llvm::function_ref<void(PipeRecordAttr)> callback) {
  if (records.getMappings().empty()) {
    for (PipeRecordAttr record : records.getPipes()) {
      callback(record);
    }
    return;
  }
  for (PipeMappingAttr mapping : records.getMappings()) {
    for (PipeRecordAttr nodePipe : mapping.getPipes()) {
      callback(nodePipe);
    }
  }
}

FailureOr<PipeRecordAttr> getFirstNodePipeRecord(PipeNetRecordsAttr records) {
  if (records.getMappings().empty()) {
    return records.getPipes().empty()
               ? FailureOr<PipeRecordAttr>(failure())
               : FailureOr<PipeRecordAttr>(records.getPipes().front());
  }
  for (PipeMappingAttr mapping : records.getMappings()) {
    if (!mapping.getPipes().empty()) {
      return mapping.getPipes().front();
    }
  }
  return failure();
}

void forEachPipeRecord(
    PipeNetRecordsAttr records,
    llvm::function_ref<void(std::uint64_t, PipeRecordAttr)> callback) {
  std::uint64_t recordIndex = 0;
  if (records.getMappings().empty()) {
    for (PipeRecordAttr record : records.getPipes()) {
      callback(recordIndex++, record);
    }
    return;
  }

  MLIRContext *context = records.getContext();
  for (PipeMappingAttr mapping : records.getMappings()) {
    TransferGraphAttr graph = mapping.getGraph();
    createTransferGraph(graph)->forEachEdge([&](TransferEdgeAttr edge) {
      DeviceTransferAttr transfer =
          DeviceTransferAttr::get(context, graph.getDomain(), edge);
      for (PipeRecordAttr nodePipe : mapping.getPipes()) {
        callback(recordIndex++,
                 PipeRecordAttr::get(
                     context, nodePipe.getSrcX(), nodePipe.getSrcY(),
                     nodePipe.getDstStartX(), nodePipe.getDstStartY(),
                     nodePipe.getDstEndX(), nodePipe.getDstEndY(),
                     nodePipe.getIsCollective(), transfer));
      }
    });
  }
}

FailureOr<PipeRecordAttr> getPipeRecord(PipeNetRecordsAttr records,
                                        std::uint64_t recordIndex) {
  if (records.getMappings().empty()) {
    if (recordIndex >= records.getPipes().size()) {
      return failure();
    }
    return records.getPipes()[recordIndex];
  }

  for (PipeMappingAttr mapping : records.getMappings()) {
    std::unique_ptr<TransferGraph> graph =
        createTransferGraph(mapping.getGraph());
    FailureOr<std::uint64_t> edgeCount = graph->getEdgeCount();
    std::uint64_t nodePipeCount = mapping.getPipes().size();
    std::optional<std::uint64_t> mappingRecordCount =
        succeeded(edgeCount)
            ? llvm::checkedMulUnsigned(*edgeCount, nodePipeCount)
            : std::nullopt;
    if (!mappingRecordCount) {
      return failure();
    }
    if (recordIndex >= *mappingRecordCount) {
      recordIndex -= *mappingRecordCount;
      continue;
    }

    std::uint64_t edgeOrdinal = recordIndex / nodePipeCount;
    std::uint64_t nodePipeIndex = recordIndex % nodePipeCount;
    SmallVector<TransferEdgeAttr> edges = graph->getEdges();
    if (edgeOrdinal >= edges.size()) {
      return failure();
    }
    DeviceTransferAttr transfer = DeviceTransferAttr::get(
        records.getContext(), mapping.getGraph().getDomain(),
        edges[edgeOrdinal]);
    PipeRecordAttr nodePipe = mapping.getPipes()[nodePipeIndex];
    return PipeRecordAttr::get(
        records.getContext(), nodePipe.getSrcX(), nodePipe.getSrcY(),
        nodePipe.getDstStartX(), nodePipe.getDstStartY(), nodePipe.getDstEndX(),
        nodePipe.getDstEndY(), nodePipe.getIsCollective(), transfer);
  }
  return failure();
}

FailureOr<PipeRecordAttr> getFirstPipeRecord(PipeNetRecordsAttr records) {
  return getPipeRecord(records, 0);
}

FailureOr<SmallVector<PipeRecordLocalIndex>>
getPipeRecordLocalIndices(PipeNetRecordsAttr records, PipeRole role) {
  assert(role != PipeRole::Active &&
         "selected record indexing requires one endpoint role");
  if (records.getMappings().empty()) {
    SmallVector<PipeRecordLocalIndex> localIndices;
    localIndices.reserve(records.getPipes().size());
    for (std::uint64_t recordIndex = 0; recordIndex < records.getPipes().size();
         ++recordIndex) {
      localIndices.push_back(
          PipeRecordLocalIndex{recordIndex, records.getPipes().size()});
    }
    return localIndices;
  }

  SmallVector<PipeRecordLocalIndex> localIndices;
  for (PipeMappingAttr mapping : records.getMappings()) {
    std::unique_ptr<TransferGraph> graph =
        createTransferGraph(mapping.getGraph());
    std::uint64_t nodePipeCount = mapping.getPipes().size();
    llvm::DenseMap<DeviceRefAttr, std::uint64_t> incidentEdgeCounts;
    struct PendingLocalIndex {
      std::uint64_t index;
      DeviceRefAttr endpoint;
    };
    SmallVector<PendingLocalIndex> mappingIndices;
    bool overflow = false;
    graph->forEachEdge([&](TransferEdgeAttr edge) {
      if (overflow) {
        return;
      }
      DeviceRefAttr endpoint =
          role == PipeRole::Source ? edge.getSource() : edge.getDestination();
      std::uint64_t incidentEdgeOrdinal = incidentEdgeCounts[endpoint]++;
      std::optional<std::uint64_t> localBase =
          llvm::checkedMulUnsigned(incidentEdgeOrdinal, nodePipeCount);
      if (!localBase) {
        overflow = true;
        return;
      }
      for (std::uint64_t nodePipeIndex = 0; nodePipeIndex < nodePipeCount;
           ++nodePipeIndex) {
        std::optional<std::uint64_t> localIndex =
            llvm::checkedAddUnsigned(*localBase, nodePipeIndex);
        if (!localIndex) {
          overflow = true;
          return;
        }
        mappingIndices.push_back(PendingLocalIndex{*localIndex, endpoint});
      }
    });
    if (overflow) {
      return failure();
    }
    for (const PendingLocalIndex &pending : mappingIndices) {
      std::optional<std::uint64_t> localCount = llvm::checkedMulUnsigned(
          incidentEdgeCounts.lookup(pending.endpoint), nodePipeCount);
      if (!localCount) {
        return failure();
      }
      localIndices.push_back(PipeRecordLocalIndex{pending.index, *localCount});
    }
  }
  return localIndices;
}

} // namespace mlir::tt::ttl
