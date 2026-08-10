// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

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

struct ExplicitEdgeIncidentOrdinals {
  SmallVector<int64_t> source;
  SmallVector<int64_t> destination;
};

ExplicitEdgeIncidentOrdinals
buildExplicitEdgeIncidentOrdinals(DeviceDomainAttr domain,
                                  ArrayRef<TransferEdgeAttr> edges) {
  FailureOr<std::uint64_t> deviceCount = getDomainDeviceCount(domain);
  assert(succeeded(deviceCount) &&
         "graph verification must reject overflowing domain extents");
  SmallVector<int64_t> nextSourceOrdinal(*deviceCount, 0);
  SmallVector<int64_t> nextDestinationOrdinal(*deviceCount, 0);
  ExplicitEdgeIncidentOrdinals ordinals;
  ordinals.source.reserve(edges.size());
  ordinals.destination.reserve(edges.size());
  for (TransferEdgeAttr edge : edges) {
    int64_t sourceIndex = getLogicalDeviceIndex(domain, edge.getSource());
    int64_t destinationIndex =
        getLogicalDeviceIndex(domain, edge.getDestination());
    ordinals.source.push_back(nextSourceOrdinal[sourceIndex]++);
    ordinals.destination.push_back(nextDestinationOrdinal[destinationIndex]++);
  }
  return ordinals;
}

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

void appendMatchingIncidentEdges(ArrayRef<TransferEdgeAttr> candidates,
                                 DeviceRefAttr device, PipeRole role,
                                 SmallVectorImpl<TransferEdgeAttr> &edges) {
  for (TransferEdgeAttr edge : candidates) {
    bool sourceMatches = edge.getSource() == device;
    bool destinationMatches = edge.getDestination() == device;
    if ((role == PipeRole::Source && sourceMatches) ||
        (role == PipeRole::Destination && destinationMatches) ||
        (role == PipeRole::Active && (sourceMatches || destinationMatches))) {
      edges.push_back(edge);
    }
  }
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

  TransferGraphIncidentEdgeIndexValues
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
    TransferGraphEdgeIndexValues edge =
        buildEdgeIndexValues(builder, loc, edgeOrdinal);
    ExplicitEdgeIncidentOrdinals incidentOrdinals =
        buildExplicitEdgeIncidentOrdinals(getDomain(), edges);
    Value sourceIncidentOrdinal = buildIndexTableLookup(
        builder, loc, incidentOrdinals.source, edgeOrdinal);
    Value destinationIncidentOrdinal = buildIndexTableLookup(
        builder, loc, incidentOrdinals.destination, edgeOrdinal);
    return {edge.edgeOrdinal, edge.source, edge.destination,
            sourceIncidentOrdinal, destinationIncidentOrdinal};
  }

  void
  appendIncidentEdges(DeviceRefAttr device, PipeRole role,
                      SmallVectorImpl<TransferEdgeAttr> &edges) const override {
    appendMatchingIncidentEdges(getEdges(), device, role, edges);
  }
};

class StructuredTransferGraph : public TransferGraph {
public:
  StructuredTransferGraph(DeviceDomainAttr graphDomain,
                          TransferGraphKind graphKind,
                          StringAttr graphComponentName,
                          DictionaryAttr graphProperties)
      : TransferGraph(graphDomain, graphKind, graphComponentName,
                      graphProperties),
        context(graphDomain.getContext()), domain(graphDomain),
        componentIndex(
            findDomainComponentIndex(domain, graphComponentName).value_or(0)),
        component(findDomainComponentIndex(domain, graphComponentName)
                      ? domain.getComponents()[componentIndex]
                      : DeviceDomainComponentAttr()),
        devices(component ? enumerateDomainDevices(domain)
                          : SmallVector<DeviceRefAttr>()) {}

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

  SmallVector<DenseI64ArrayAttr> getComponentCoordinates() const {
    SmallVector<DenseI64ArrayAttr> coordinates;
    SmallVector<int64_t> current;
    enumerateComponentCoordinates(context, component.getExtent().asArrayRef(),
                                  0, current, coordinates);
    return coordinates;
  }

  FailureOr<std::uint64_t> getComponentSize() const {
    return getExtentElementCount(component.getExtent().asArrayRef());
  }

  FailureOr<std::uint64_t> getDeviceCount() const {
    return getDomainDeviceCount(domain);
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

  Value compressComponentIndexExcluding(OpBuilder &builder, Location loc,
                                        Value componentIndex,
                                        Value excludedComponentIndex) const {
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    Value afterExcluded =
        arith::SubIOp::create(builder, loc, componentIndex, one);
    Value isAfterExcluded =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt,
                              componentIndex, excludedComponentIndex);
    return arith::SelectOp::create(builder, loc, isAfterExcluded, afterExcluded,
                                   componentIndex);
  }

  std::uint64_t getComponentLinearIndex(DenseI64ArrayAttr coordinates) const {
    std::uint64_t index = 0;
    for (auto [coordinate, extent] : llvm::zip_equal(
             coordinates.asArrayRef(), component.getExtent().asArrayRef())) {
      index = index * extent + coordinate;
    }
    return index;
  }

  void appendIncidentEdgesFromCompleteRelation(
      DeviceRefAttr device, PipeRole role,
      SmallVectorImpl<TransferEdgeAttr> &edges) const {
    appendMatchingIncidentEdges(getEdges(), device, role, edges);
  }

  MLIRContext *context;
  DeviceDomainAttr domain;
  std::size_t componentIndex;
  DeviceDomainComponentAttr component;
  SmallVector<DeviceRefAttr> devices;
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
    for (DeviceRefAttr source : devices) {
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

  TransferGraphIncidentEdgeIndexValues
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
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    return {buildEdgeOrdinalForSource(builder, loc, source), source,
            destination, zero, zero};
  }

  void
  appendIncidentEdges(DeviceRefAttr device, PipeRole role,
                      SmallVectorImpl<TransferEdgeAttr> &edges) const override {
    int64_t axis = getProperties().getAs<IntegerAttr>("axis").getInt();
    int64_t offset = getProperties().getAs<IntegerAttr>("offset").getInt();
    bool wrap = getProperties().getAs<BoolAttr>("wrap").getValue();
    int64_t axisExtent = component.getExtent()[axis];
    if (wrap) {
      offset %= axisExtent;
    }
    llvm::SetVector<TransferEdgeAttr> incidentEdges;
    if (role == PipeRole::Source || role == PipeRole::Active) {
      SmallVector<int64_t> destinationComponent(
          device.getCoordinates()[componentIndex].asArrayRef());
      int64_t destinationAxis = destinationComponent[axis] + offset;
      if (destinationAxis < axisExtent || wrap) {
        destinationComponent[axis] = destinationAxis % axisExtent;
        DeviceRefAttr destination = replaceComponent(
            device, DenseI64ArrayAttr::get(context, destinationComponent));
        incidentEdges.insert(
            getPointTransferEdge(context, device, destination));
      }
    }
    if (role == PipeRole::Destination || role == PipeRole::Active) {
      SmallVector<int64_t> sourceComponent(
          device.getCoordinates()[componentIndex].asArrayRef());
      int64_t sourceAxis = sourceComponent[axis] - offset;
      if (sourceAxis >= 0 || wrap) {
        sourceAxis %= axisExtent;
        if (sourceAxis < 0) {
          sourceAxis += axisExtent;
        }
        sourceComponent[axis] = sourceAxis;
        DeviceRefAttr source = replaceComponent(
            device, DenseI64ArrayAttr::get(context, sourceComponent));
        incidentEdges.insert(getPointTransferEdge(context, source, device));
      }
    }
    edges.append(incidentEdges.begin(), incidentEdges.end());
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
    for (const StencilOffsetDescriptor &descriptor : getDescriptors()) {
      for (DeviceRefAttr source : devices) {
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
    FailureOr<std::uint64_t> deviceCount = getDeviceCount();
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    assert(succeeded(deviceCount) && succeeded(componentSize) &&
           "graph verification must reject overflowing stencil domains");
    std::uint64_t surroundingContextCount = *deviceCount / *componentSize;
    std::uint64_t blockStart = 0;
    std::optional<TransferGraphEdgeIndexValues> selectedIndices;
    for (const StencilOffsetDescriptor &descriptor : getDescriptors()) {
      TransferGraphEdgeIndexValues candidate = buildOffsetEdgeIndexValues(
          builder, loc, edgeIndex, descriptor, blockStart);
      if (!selectedIndices) {
        selectedIndices = candidate;
      } else {
        Value start = arith::ConstantIndexOp::create(
            builder, loc, static_cast<int64_t>(blockStart));
        Value selectsCandidate = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::sge, edgeIndex, start);
        selectedIndices = TransferGraphEdgeIndexValues{
            edgeIndex,
            arith::SelectOp::create(builder, loc, selectsCandidate,
                                    candidate.source, selectedIndices->source),
            arith::SelectOp::create(builder, loc, selectsCandidate,
                                    candidate.destination,
                                    selectedIndices->destination)};
      }
      blockStart += surroundingContextCount * descriptor.sourceCount;
    }
    assert(selectedIndices &&
           "graph verification must reject an empty stencil relation");
    return *selectedIndices;
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

  TransferGraphIncidentEdgeIndexValues
  buildIncidentEdgeIndexValues(OpBuilder &builder, Location loc,
                               Value deviceIndex, Value incidentEdgeIndex,
                               PipeRole role) const override {
    assert(role != PipeRole::Active &&
           "dynamic incident iteration requires one endpoint role");
    Value selectedOrdinal = arith::ConstantIndexOp::create(builder, loc, 0);
    Value selectedSource = deviceIndex;
    Value selectedDestination = deviceIndex;
    Value selectedSourceIncidentOrdinal =
        arith::ConstantIndexOp::create(builder, loc, 0);
    Value selectedDestinationIncidentOrdinal =
        arith::ConstantIndexOp::create(builder, loc, 0);
    Value validPrefix = arith::ConstantIndexOp::create(builder, loc, 0);
    std::uint64_t blockStart = 0;
    FailureOr<std::uint64_t> deviceCount = getDeviceCount();
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    assert(succeeded(deviceCount) && succeeded(componentSize) &&
           "graph verification must reject overflowing stencil domains");
    std::uint64_t surroundingContextCount = *deviceCount / *componentSize;
    SmallVector<StencilOffsetDescriptor> descriptors = getDescriptors();
    for (auto [descriptorIndex, descriptor] : llvm::enumerate(descriptors)) {
      DynamicStencilEdge candidate =
          buildIncidentCandidate(builder, loc, deviceIndex, role,
                                 descriptorIndex, descriptor, blockStart);
      Value hasOrdinal =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                incidentEdgeIndex, validPrefix);
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
      selectedSourceIncidentOrdinal = arith::SelectOp::create(
          builder, loc, selectsCandidate, candidate.sourceIncidentOrdinal,
          selectedSourceIncidentOrdinal);
      selectedDestinationIncidentOrdinal = arith::SelectOp::create(
          builder, loc, selectsCandidate, candidate.destinationIncidentOrdinal,
          selectedDestinationIncidentOrdinal);
      Value contribution =
          buildCountFromPredicate(builder, loc, candidate.isValid);
      validPrefix =
          arith::AddIOp::create(builder, loc, validPrefix, contribution);
      blockStart += surroundingContextCount * descriptor.sourceCount;
    }
    return {selectedOrdinal, selectedSource, selectedDestination,
            selectedSourceIncidentOrdinal, selectedDestinationIncidentOrdinal};
  }

  void
  appendIncidentEdges(DeviceRefAttr device, PipeRole role,
                      SmallVectorImpl<TransferEdgeAttr> &edges) const override {
    if (role == PipeRole::Active) {
      appendIncidentEdgesFromCompleteRelation(device, role, edges);
      return;
    }

    for (const StencilOffsetDescriptor &descriptor : getDescriptors()) {
      if (role == PipeRole::Source) {
        std::optional<DeviceRefAttr> destination = translateComponent(
            device, descriptor.offset.asArrayRef(), /*direction=*/1);
        if (destination && *destination != device) {
          edges.push_back(getPointTransferEdge(context, device, *destination));
        }
      } else {
        std::optional<DeviceRefAttr> source = translateComponent(
            device, descriptor.offset.asArrayRef(), /*direction=*/-1);
        if (source && *source != device) {
          edges.push_back(getPointTransferEdge(context, *source, device));
        }
      }
    }
  }

private:
  struct DynamicStencilEdge {
    Value isValid;
    Value edgeOrdinal;
    Value source;
    Value destination;
    Value sourceIncidentOrdinal;
    Value destinationIncidentOrdinal;
  };

  DynamicStencilEdge
  buildIncidentCandidate(OpBuilder &builder, Location loc, Value endpointDevice,
                         PipeRole role, std::size_t descriptorIndex,
                         const StencilOffsetDescriptor &descriptor,
                         std::uint64_t blockStart) const {
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
    Value edgeOrdinal = buildEdgeOrdinalForSource(
        builder, loc, source, sourceCoordinates, descriptor, blockStart);
    Value sourceIncidentOrdinal = buildIncidentOrdinalForEndpoint(
        builder, loc, source, PipeRole::Source, descriptorIndex);
    Value destinationIncidentOrdinal = buildIncidentOrdinalForEndpoint(
        builder, loc, destination, PipeRole::Destination, descriptorIndex);
    return {isValid,     edgeOrdinal,           source,
            destination, sourceIncidentOrdinal, destinationIncidentOrdinal};
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

  Value buildIncidentOrdinalForEndpoint(OpBuilder &builder, Location loc,
                                        Value endpointDevice, PipeRole role,
                                        std::size_t descriptorIndex) const {
    Value ordinal = arith::ConstantIndexOp::create(builder, loc, 0);
    SmallVector<StencilOffsetDescriptor> descriptors = getDescriptors();
    for (const StencilOffsetDescriptor &descriptor :
         ArrayRef<StencilOffsetDescriptor>(descriptors)
             .take_front(descriptorIndex)) {
      Value isValid =
          buildOffsetValidity(builder, loc, endpointDevice, role, descriptor);
      Value contribution = buildCountFromPredicate(builder, loc, isValid);
      ordinal = arith::AddIOp::create(builder, loc, ordinal, contribution);
    }
    return ordinal;
  }

  Value buildEdgeOrdinalForSource(OpBuilder &builder, Location loc,
                                  Value source,
                                  ArrayRef<Value> sourceCoordinates,
                                  const StencilOffsetDescriptor &descriptor,
                                  std::uint64_t blockStart) const {
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    FailureOr<std::uint64_t> trailingSize = getTrailingComponentSize();
    assert(succeeded(componentSize) && succeeded(trailingSize) &&
           "graph verification must reject overflowing stencil domains");
    std::uint64_t completeContextBlockSize = *componentSize * *trailingSize;
    Value completeContextBlock = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(completeContextBlockSize));
    Value contextIndex =
        arith::DivSIOp::create(builder, loc, source, completeContextBlock);
    Value trailing = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*trailingSize));
    Value trailingOrdinal =
        arith::RemSIOp::create(builder, loc, source, trailing);

    Value compressedComponentOrdinal =
        arith::ConstantIndexOp::create(builder, loc, 0);
    for (auto [coordinate, lowerBound, sourceExtent] :
         llvm::zip_equal(sourceCoordinates, descriptor.sourceLowerBounds,
                         descriptor.sourceExtents)) {
      Value lower = arith::ConstantIndexOp::create(builder, loc, lowerBound);
      Value normalized = arith::SubIOp::create(builder, loc, coordinate, lower);
      Value extent = arith::ConstantIndexOp::create(builder, loc, sourceExtent);
      Value withAxis = arith::MulIOp::create(
          builder, loc, compressedComponentOrdinal, extent);
      compressedComponentOrdinal =
          arith::AddIOp::create(builder, loc, withAxis, normalized);
    }
    Value descriptorBlockSize = arith::ConstantIndexOp::create(
        builder, loc,
        static_cast<int64_t>(descriptor.sourceCount * *trailingSize));
    Value contextOffset =
        arith::MulIOp::create(builder, loc, contextIndex, descriptorBlockSize);
    Value componentOffset = arith::MulIOp::create(
        builder, loc, compressedComponentOrdinal, trailing);
    Value localWithComponent =
        arith::AddIOp::create(builder, loc, contextOffset, componentOffset);
    Value localOrdinal = arith::AddIOp::create(builder, loc, localWithComponent,
                                               trailingOrdinal);
    Value start = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(blockStart));
    return arith::AddIOp::create(builder, loc, start, localOrdinal);
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

  TransferGraphEdgeIndexValues
  buildOffsetEdgeIndexValues(OpBuilder &builder, Location loc, Value edgeIndex,
                             const StencilOffsetDescriptor &descriptor,
                             std::uint64_t blockStart) const {
    FailureOr<std::uint64_t> componentSize = getComponentSize();
    FailureOr<std::uint64_t> trailingSize = getTrailingComponentSize();
    assert(succeeded(componentSize) && succeeded(trailingSize) &&
           "graph verification must reject overflowing stencil domains");

    Value start = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(blockStart));
    Value localOrdinal = arith::SubIOp::create(builder, loc, edgeIndex, start);
    std::uint64_t contextBlockSize = descriptor.sourceCount * *trailingSize;
    Value contextBlock = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(contextBlockSize));
    Value trailing = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(*trailingSize));
    Value contextIndex =
        arith::DivSIOp::create(builder, loc, localOrdinal, contextBlock);
    Value contextRemainder =
        arith::RemSIOp::create(builder, loc, localOrdinal, contextBlock);
    Value componentOrdinal =
        arith::DivSIOp::create(builder, loc, contextRemainder, trailing);
    Value trailingOrdinal =
        arith::RemSIOp::create(builder, loc, contextRemainder, trailing);

    SmallVector<Value> sourceCoordinates;
    SmallVector<Value> destinationCoordinates;
    ArrayRef<int64_t> componentExtent = component.getExtent().asArrayRef();
    bool wrap = getProperties().getAs<BoolAttr>("wrap").getValue();
    for (std::size_t axis = 0; axis < componentExtent.size(); ++axis) {
      std::uint64_t coordinateStride = 1;
      for (int64_t extent :
           ArrayRef<int64_t>(descriptor.sourceExtents).drop_front(axis + 1)) {
        coordinateStride *= extent;
      }
      Value stride = arith::ConstantIndexOp::create(
          builder, loc, static_cast<int64_t>(coordinateStride));
      Value axisExtent = arith::ConstantIndexOp::create(
          builder, loc, descriptor.sourceExtents[axis]);
      Value coordinateWithInner =
          arith::DivSIOp::create(builder, loc, componentOrdinal, stride);
      Value coordinateOffset =
          arith::RemSIOp::create(builder, loc, coordinateWithInner, axisExtent);
      Value lowerBound = arith::ConstantIndexOp::create(
          builder, loc, descriptor.sourceLowerBounds[axis]);
      Value sourceCoordinate =
          arith::AddIOp::create(builder, loc, coordinateOffset, lowerBound);
      sourceCoordinates.push_back(sourceCoordinate);

      int64_t delta = descriptor.offset[axis];
      Value destinationCoordinate = sourceCoordinate;
      if (wrap && delta != 0) {
        int64_t wrapThreshold = componentExtent[axis] - delta;
        Value threshold =
            arith::ConstantIndexOp::create(builder, loc, wrapThreshold);
        Value wraps =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                  sourceCoordinate, threshold);
        Value wrapped =
            arith::SubIOp::create(builder, loc, sourceCoordinate, threshold);
        Value deltaValue = arith::ConstantIndexOp::create(builder, loc, delta);
        Value unwrapped =
            arith::AddIOp::create(builder, loc, sourceCoordinate, deltaValue);
        destinationCoordinate =
            arith::SelectOp::create(builder, loc, wraps, wrapped, unwrapped);
      } else if (!wrap && delta > 0) {
        Value deltaValue = arith::ConstantIndexOp::create(builder, loc, delta);
        destinationCoordinate =
            arith::AddIOp::create(builder, loc, sourceCoordinate, deltaValue);
      } else if (!wrap && delta < 0) {
        Value magnitude = arith::ConstantIndexOp::create(builder, loc, -delta);
        destinationCoordinate =
            arith::SubIOp::create(builder, loc, sourceCoordinate, magnitude);
      }
      destinationCoordinates.push_back(destinationCoordinate);
    }

    auto buildComponentLinearIndex = [&](ArrayRef<Value> coordinates) {
      Value linearIndex = arith::ConstantIndexOp::create(builder, loc, 0);
      for (auto [coordinate, extent] :
           llvm::zip_equal(coordinates, componentExtent)) {
        Value extentValue =
            arith::ConstantIndexOp::create(builder, loc, extent);
        Value withAxis =
            arith::MulIOp::create(builder, loc, linearIndex, extentValue);
        linearIndex = arith::AddIOp::create(builder, loc, withAxis, coordinate);
      }
      return linearIndex;
    };
    Value sourceComponentIndex = buildComponentLinearIndex(sourceCoordinates);
    Value destinationComponentIndex =
        buildComponentLinearIndex(destinationCoordinates);
    std::uint64_t completeContextBlockSize = *componentSize * *trailingSize;
    Value completeContextBlock = arith::ConstantIndexOp::create(
        builder, loc, static_cast<int64_t>(completeContextBlockSize));
    Value contextBase =
        arith::MulIOp::create(builder, loc, contextIndex, completeContextBlock);
    auto buildDeviceIndex = [&](Value selectedComponentIndex) {
      Value componentOffset =
          arith::MulIOp::create(builder, loc, selectedComponentIndex, trailing);
      Value withComponent =
          arith::AddIOp::create(builder, loc, contextBase, componentOffset);
      return arith::AddIOp::create(builder, loc, withComponent,
                                   trailingOrdinal);
    };
    return {edgeIndex, buildDeviceIndex(sourceComponentIndex),
            buildDeviceIndex(destinationComponentIndex)};
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
    for (DeviceRefAttr source : devices) {
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

  TransferGraphIncidentEdgeIndexValues
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
    Value sourceIncidentOrdinal =
        arith::ConstantIndexOp::create(builder, loc, 0);
    Value sourceComponent = getComponentIndex(builder, loc, source);
    Value destinationIncidentOrdinal = compressComponentIndexExcluding(
        builder, loc, sourceComponent, rootValue);
    return {edgeOrdinal, source, destination, sourceIncidentOrdinal,
            destinationIncidentOrdinal};
  }

  void
  appendIncidentEdges(DeviceRefAttr device, PipeRole role,
                      SmallVectorImpl<TransferEdgeAttr> &edges) const override {
    DeviceRefAttr root = getProperties().getAs<DeviceRefAttr>("root");
    DenseI64ArrayAttr rootCoordinates = root.getCoordinates().front();
    DeviceRefAttr destination = replaceComponent(device, rootCoordinates);
    if ((role == PipeRole::Source || role == PipeRole::Active) &&
        device != destination) {
      edges.push_back(getPointTransferEdge(context, device, destination));
    }
    if ((role == PipeRole::Destination || role == PipeRole::Active) &&
        device.getCoordinates()[componentIndex] == rootCoordinates) {
      for (DenseI64ArrayAttr sourceCoordinates : getComponentCoordinates()) {
        DeviceRefAttr source = replaceComponent(device, sourceCoordinates);
        if (source != device) {
          edges.push_back(getPointTransferEdge(context, source, device));
        }
      }
    }
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
    for (DeviceRefAttr destination : devices) {
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

  TransferGraphIncidentEdgeIndexValues
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
    Value destinationComponent = getComponentIndex(builder, loc, destination);
    Value sourceIncidentOrdinal = compressComponentIndexExcluding(
        builder, loc, destinationComponent, sourceValue);
    Value destinationIncidentOrdinal =
        arith::ConstantIndexOp::create(builder, loc, 0);
    return {edgeOrdinal, source, destination, sourceIncidentOrdinal,
            destinationIncidentOrdinal};
  }

  void
  appendIncidentEdges(DeviceRefAttr device, PipeRole role,
                      SmallVectorImpl<TransferEdgeAttr> &edges) const override {
    DeviceRefAttr sourceEndpoint =
        getProperties().getAs<DeviceRefAttr>("source");
    DenseI64ArrayAttr sourceCoordinates =
        sourceEndpoint.getCoordinates().front();
    DeviceRefAttr source = replaceComponent(device, sourceCoordinates);
    if ((role == PipeRole::Destination || role == PipeRole::Active) &&
        source != device) {
      edges.push_back(getPointTransferEdge(context, source, device));
    }
    if ((role == PipeRole::Source || role == PipeRole::Active) &&
        device.getCoordinates()[componentIndex] == sourceCoordinates) {
      for (DenseI64ArrayAttr destinationCoordinates :
           getComponentCoordinates()) {
        DeviceRefAttr destination =
            replaceComponent(device, destinationCoordinates);
        if (destination != device) {
          edges.push_back(getPointTransferEdge(context, device, destination));
        }
      }
    }
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
    for (DeviceRefAttr source : devices) {
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

  TransferGraphIncidentEdgeIndexValues
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
    Value sourceComponent = getComponentIndex(builder, loc, source);
    Value destinationComponent = getComponentIndex(builder, loc, destination);
    Value destinationIncidentOrdinal = compressComponentIndexExcluding(
        builder, loc, sourceComponent, destinationComponent);
    return {edgeOrdinal, source, destination, compressedDestination,
            destinationIncidentOrdinal};
  }

  void
  appendIncidentEdges(DeviceRefAttr device, PipeRole role,
                      SmallVectorImpl<TransferEdgeAttr> &edges) const override {
    for (DenseI64ArrayAttr peerCoordinates : getComponentCoordinates()) {
      DeviceRefAttr peer = replaceComponent(device, peerCoordinates);
      if (peer == device) {
        continue;
      }
      if (role == PipeRole::Source || role == PipeRole::Active) {
        edges.push_back(getPointTransferEdge(context, device, peer));
      }
      if (role == PipeRole::Destination || role == PipeRole::Active) {
        edges.push_back(getPointTransferEdge(context, peer, device));
      }
    }
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

SmallVector<TransferEdgeAttr>
TransferGraph::getIncidentEdges(DeviceRefAttr device, PipeRole role) const {
  SmallVector<TransferEdgeAttr> edges;
  appendIncidentEdges(device, role, edges);
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
  if (!records.getMappings()) {
    return records.getPipes().size();
  }

  std::uint64_t recordCount = 0;
  for (Attribute mappingAttribute : records.getMappings()) {
    auto mapping = mlir::cast<PipeMappingAttr>(mappingAttribute);
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
  if (!records.getMappings()) {
    for (PipeRecordAttr record : records.getPipes()) {
      callback(record);
    }
    return;
  }
  for (Attribute mappingAttribute : records.getMappings()) {
    auto mapping = mlir::cast<PipeMappingAttr>(mappingAttribute);
    for (PipeRecordAttr nodePipe : mapping.getPipes()) {
      callback(nodePipe);
    }
  }
}

FailureOr<PipeRecordAttr> getFirstNodePipeRecord(PipeNetRecordsAttr records) {
  if (!records.getMappings()) {
    return records.getPipes().empty()
               ? FailureOr<PipeRecordAttr>(failure())
               : FailureOr<PipeRecordAttr>(records.getPipes().front());
  }
  for (Attribute mappingAttribute : records.getMappings()) {
    auto mapping = mlir::cast<PipeMappingAttr>(mappingAttribute);
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
  if (!records.getMappings()) {
    for (PipeRecordAttr record : records.getPipes()) {
      callback(recordIndex++, record);
    }
    return;
  }

  MLIRContext *context = records.getContext();
  for (Attribute mappingAttribute : records.getMappings()) {
    auto mapping = mlir::cast<PipeMappingAttr>(mappingAttribute);
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
  PipeRecordAttr selectedRecord;
  forEachPipeRecord(records, [&](std::uint64_t candidateIndex,
                                 PipeRecordAttr candidateRecord) {
    if (candidateIndex == recordIndex) {
      selectedRecord = candidateRecord;
    }
  });
  return selectedRecord ? FailureOr<PipeRecordAttr>(selectedRecord)
                        : FailureOr<PipeRecordAttr>(failure());
}

FailureOr<PipeRecordAttr> getFirstPipeRecord(PipeNetRecordsAttr records) {
  return getPipeRecord(records, 0);
}

FailureOr<SmallVector<PipeRecordLocalIndex>>
getPipeRecordLocalIndices(PipeNetRecordsAttr records, PipeRole role) {
  assert(role != PipeRole::Active &&
         "selected record indexing requires one endpoint role");
  if (!records.getMappings()) {
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
  for (Attribute mappingAttribute : records.getMappings()) {
    auto mapping = mlir::cast<PipeMappingAttr>(mappingAttribute);
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

FailureOr<ttcore::TileType> getTileType(Type type) {
  if (auto tileType = dyn_cast<ttcore::TileType>(type)) {
    return tileType;
  }
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType) {
    return failure();
  }
  auto tileType = dyn_cast<ttcore::TileType>(tensorType.getElementType());
  if (!tileType) {
    return failure();
  }
  return tileType;
}

LogicalResult verifyTypecastTileTypes(ttcore::TileType inputType,
                                      ttcore::TileType resultType,
                                      std::string &failureReason) {
  failureReason.clear();
  llvm::raw_string_ostream diagnostic(failureReason);
  if (inputType.getShape() != resultType.getShape()) {
    diagnostic << "input and result tile shapes must match, but got input: "
               << inputType << ", result: " << resultType;
    return failure();
  }
  if (!ttcore::isFloat(inputType.getDataType()) ||
      !ttcore::isFloat(resultType.getDataType())) {
    diagnostic
        << "only supports floating-point tile data types, but got input: "
        << inputType << ", result: " << resultType;
    return failure();
  }
  return success();
}

FailureOr<int64_t> getDFBId(Value cb) {
  auto bindOp = getDFBDeclaration(cb);
  if (!bindOp) {
    return failure();
  }
  auto dfbId = bindOp.getDfbId();
  if (!dfbId.has_value()) {
    return failure();
  }
  return dfbId->getSExtValue();
}

FailureOr<uint64_t> getDFBPagesPerBlock(CircularBufferType type) {
  uint64_t pagesPerBlock = 1;
  for (int64_t dimension : type.getShape()) {
    if (dimension <= 0) {
      return failure();
    }
    std::optional<uint64_t> product = llvm::checkedMulUnsigned(
        pagesPerBlock, static_cast<uint64_t>(dimension));
    if (!product) {
      return failure();
    }
    pagesPerBlock = *product;
  }
  return pagesPerBlock;
}

FailureOr<uint64_t> getDFBPageSizeBytes(CircularBufferType type) {
  Type elementType = type.getElementType();
  if (auto tileType = dyn_cast<ttcore::TileType>(elementType)) {
    return tileType.getSizeBytes();
  }
  if (!elementType.isIntOrFloat()) {
    return failure();
  }
  uint64_t bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0) {
    return failure();
  }
  return bitWidth / 8;
}

LogicalResult verifyDFBOperandIdentities(
    ModuleOp moduleOp, StringRef consumerPass,
    llvm::function_ref<bool(Operation *)> operationFilter,
    llvm::function_ref<FailureOr<int64_t>(Value)> identityResolver,
    StringRef operandDescription, DFBIdentityRequirement requirement) {
  WalkResult result = moduleOp.walk([&](Operation *operation) {
    if (!operationFilter(operation)) {
      return WalkResult::advance();
    }
    for (Value operand : operation->getOperands()) {
      if (!isa<CircularBufferType>(operand.getType())) {
        continue;
      }
      if (succeeded(identityResolver(operand))) {
        continue;
      }
      InFlightDiagnostic diagnostic = operation->emitOpError();
      diagnostic << "`" << consumerPass << "` requires every "
                 << operandDescription
                 << " operand to resolve to `ttl.bind_cb`";
      if (requirement == DFBIdentityRequirement::Finalized) {
        diagnostic << " with `dfb_id` after finalization";
      }
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult verifyResolvedDFBIdentities(ModuleOp moduleOp,
                                          StringRef consumerPass) {
  bool hasAllocationMetadata = moduleOp->hasAttr(kDFBAllocationsAttrName);
  bool hasDFB = false;
  WalkResult result =
      moduleOp.walk([&](Operation *nestedOperation) {
        auto access = dyn_cast<DFBAccessOpInterface>(nestedOperation);
        if (!isa<BindCBOp>(nestedOperation) &&
            (!access || access.getDFBProtocolEffects().empty())) {
          return WalkResult::advance();
        }
        hasDFB = true;
        if (!hasAllocationMetadata) {
          return WalkResult::interrupt();
        }
        if (auto bindOp = dyn_cast<BindCBOp>(nestedOperation);
            bindOp && !bindOp.getDfbId().has_value()) {
          bindOp.emitOpError()
              << "`" << consumerPass
              << "` requires every `ttl.bind_cb` to have `dfb_id` after "
                 "finalization";
          return WalkResult::interrupt();
        }

        return WalkResult::advance();
      });

  if (!hasDFB) {
    return success();
  }
  if (!hasAllocationMetadata) {
    moduleOp.emitOpError()
        << "`" << consumerPass
        << "` requires finalized DFB allocation metadata; run "
           "`ttl-finalize-dfb-indices` first";
    return failure();
  }
  if (result.wasInterrupted()) {
    return failure();
  }

  WalkResult effectResult = moduleOp.walk([&](Operation *operation) {
    auto access = dyn_cast<DFBAccessOpInterface>(operation);
    if (!access) {
      return WalkResult::advance();
    }
    for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
      if (succeeded(getDFBId(effect.dfb))) {
        continue;
      }
      operation->emitOpError()
          << "`" << consumerPass
          << "` requires every DFB protocol action to resolve to `ttl.bind_cb` "
             "with `dfb_id` after finalization";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(effectResult.wasInterrupted());
}

LogicalResult verifyMatmulTileTypes(ttcore::TileType lhsType,
                                    ttcore::TileType rhsType,
                                    ttcore::TileType resultType,
                                    bool transposeRhs,
                                    std::string &failureReason) {
  failureReason.clear();
  llvm::raw_string_ostream diagnostic(failureReason);
  if (lhsType.getDataType() != rhsType.getDataType()) {
    diagnostic << "element data type mismatch: lhs has " << lhsType
               << " but rhs has " << rhsType;
    return failure();
  }
  if (resultType.getDataType() != lhsType.getDataType()) {
    diagnostic << "result element data type " << resultType
               << " must match input element data type " << lhsType;
    return failure();
  }

  int64_t rhsK = transposeRhs ? rhsType.getWidth() : rhsType.getHeight();
  if (lhsType.getWidth() != rhsK) {
    diagnostic << "tile K dimension mismatch: lhs tile width "
               << lhsType.getWidth() << " does not match rhs tile "
               << (transposeRhs ? "width " : "height ") << rhsK;
    return failure();
  }

  int64_t expectedResultWidth =
      transposeRhs ? rhsType.getHeight() : rhsType.getWidth();
  if (resultType.getHeight() != lhsType.getHeight() ||
      resultType.getWidth() != expectedResultWidth) {
    diagnostic << "result tile dimensions " << resultType.getHeight() << "x"
               << resultType.getWidth() << " do not match expected "
               << lhsType.getHeight() << "x" << expectedResultWidth;
    return failure();
  }
  return success();
}

/// FPU binary execution requires both operands to address the same tile
/// coordinates.
static bool hasMatchingFPUInputIndices(Operation *operation) {
  assert(operation->getNumOperands() >= 2 &&
         "binary tile op with execution alternatives must have two data "
         "operands");
  Value lhs = operation->getOperand(0);
  Value rhs = operation->getOperand(1);

  if (auto lhsArgument = dyn_cast<BlockArgument>(lhs)) {
    auto rhsArgument = dyn_cast<BlockArgument>(rhs);
    if (!rhsArgument || lhsArgument.getOwner() != rhsArgument.getOwner()) {
      return false;
    }
    auto computeOp =
        dyn_cast_or_null<ComputeOp>(lhsArgument.getOwner()->getParentOp());
    if (!computeOp) {
      return false;
    }
    unsigned numInputs = computeOp.getNumInputs();
    if (lhsArgument.getArgNumber() >= numInputs ||
        rhsArgument.getArgNumber() >= numInputs) {
      return false;
    }
    auto indexingMaps = computeOp.getIndexingMapsArray();
    return indexingMaps[lhsArgument.getArgNumber()] ==
           indexingMaps[rhsArgument.getArgNumber()];
  }

  auto lhsExtract = lhs.getDefiningOp<tensor::ExtractOp>();
  auto rhsExtract = rhs.getDefiningOp<tensor::ExtractOp>();
  return lhsExtract && rhsExtract &&
         lhsExtract.getIndices() == rhsExtract.getIndices();
}

SmallVector<TileExecutionStrategy, 2>
getDefaultLegalTileExecutionStrategies(Operation *operation) {
  if (!operation->hasTrait<TTLStrategyDependentBinaryOpTrait>()) {
    return {};
  }

  SmallVector<TileExecutionStrategy, 2> strategies{TileExecutionStrategy::SFPU};
  auto resultType =
      dyn_cast<ttcore::TileType>(operation->getResult(0).getType());
  if (resultType && ttcore::isFloat(resultType.getDataType()) &&
      hasMatchingFPUInputIndices(operation)) {
    strategies.insert(strategies.begin(), TileExecutionStrategy::FPU);
  }
  return strategies;
}

static bool hasDstBackedTileProducer(Value value) {
  Operation *definingOp = value.getDefiningOp();
  return definingOp &&
         (isTileComputeOp(definingOp) || isa<DstIndexOp>(definingOp));
}

FailureOr<TileExecutionInfo>
getDefaultTileExecutionInfo(Operation *operation,
                            std::optional<TileExecutionStrategy> strategy) {
  TileExecutionInfo info;
  info.operandRoutes.assign(operation->getNumOperands(),
                            TileOperandRoute::None);
  info.dstOperandsMaterializedByOperation.resize(operation->getNumOperands());
  info.resultInDst = operation->hasTrait<TTLDstResultOpTrait>();

  if (isa<CopyTileOp>(operation)) {
    info.primitive = TilePrimitive::Copy;
    info.operandRoutes[0] = TileOperandRoute::DataflowBuffer;
    return info;
  }
  if (isa<CopyDstOp>(operation)) {
    info.primitive = TilePrimitive::Copy;
    info.operandRoutes[0] = TileOperandRoute::Dst;
    return info;
  }
  if (isa<DstIndexOp>(operation)) {
    info.primitive = TilePrimitive::DstIndex;
    return info;
  }
  if (isa<TileStoreOp>(operation)) {
    info.primitive = TilePrimitive::Store;
    info.operandRoutes[0] = TileOperandRoute::Dst;
    return info;
  }
  if (auto broadcast = dyn_cast<TileBcastOp>(operation)) {
    switch (broadcast.getBcastType()) {
    case BcastType::Col:
      info.primitive = TilePrimitive::BroadcastColumn;
      break;
    case BcastType::Row:
      info.primitive = TilePrimitive::BroadcastRow;
      break;
    case BcastType::Scalar:
      info.primitive = TilePrimitive::BroadcastScalar;
      break;
    }
    info.operandRoutes[0] = TileOperandRoute::DataflowBuffer;
    return info;
  }
  if (auto reduce = dyn_cast<TileReduceOp>(operation)) {
    info.primitive = TilePrimitive::Reduce;
    info.operandRoutes[0] = TileOperandRoute::DataflowBuffer;
    info.operandRoutes[1] = TileOperandRoute::DataflowBuffer;
    switch (reduce.getReduceDim()) {
    case ttkernel::ReduceDim::Row:
      info.fullFp32Accumulation = FullFp32AccumulationKind::ReduceRow;
      break;
    case ttkernel::ReduceDim::Col:
      info.fullFp32Accumulation = FullFp32AccumulationKind::ReduceColumn;
      break;
    case ttkernel::ReduceDim::Scalar:
      info.fullFp32Accumulation = FullFp32AccumulationKind::ReduceScalar;
      break;
    }
    return info;
  }
  if (isa<TileTransposeOp>(operation)) {
    info.primitive = TilePrimitive::Transpose;
    info.operandRoutes[0] = TileOperandRoute::DataflowBuffer;
    return info;
  }
  if (isa<TileFillOp>(operation)) {
    info.primitive = TilePrimitive::Fill;
    return info;
  }
  if (auto matmul = dyn_cast<TileMatmulBlockOp>(operation)) {
    info.primitive = TilePrimitive::Matmul;
    info.operandRoutes[0] = TileOperandRoute::DataflowBuffer;
    info.operandRoutes[1] = TileOperandRoute::DataflowBuffer;
    if (matmul.getAccumulator()) {
      info.operandRoutes[2] = TileOperandRoute::Dst;
      info.dstOperandsMaterializedByOperation.set(2);
    }
    info.fullFp32Accumulation = FullFp32AccumulationKind::Matmul;
    info.accumulatesIntoDst = true;
    return info;
  }
  if (auto accumulate = dyn_cast<TileAccumulateOp>(operation)) {
    info.primitive = TilePrimitive::ElementwiseBinary;
    info.operandRoutes[0] = TileOperandRoute::Dst;
    info.operandRoutes[1] =
        hasDstBackedTileProducer(accumulate.getContribution())
            ? TileOperandRoute::Dst
            : TileOperandRoute::DataflowBuffer;
    info.accumulatesIntoDst = true;
    return info;
  }
  if (operation->hasTrait<TTLStrategyDependentBinaryOpTrait>()) {
    if (!strategy) {
      return failure();
    }
    info.primitive = TilePrimitive::ElementwiseBinary;
    TileOperandRoute route = *strategy == TileExecutionStrategy::FPU
                                 ? TileOperandRoute::DataflowBuffer
                                 : TileOperandRoute::Dst;
    info.operandRoutes[0] = route;
    info.operandRoutes[1] = route;
    info.accumulatesIntoDst = *strategy == TileExecutionStrategy::FPU;
    return info;
  }
  if (operation->hasTrait<TTLTileBinaryOpTrait>()) {
    info.primitive = TilePrimitive::ElementwiseBinary;
    info.operandRoutes[0] = TileOperandRoute::Dst;
    info.operandRoutes[1] = TileOperandRoute::Dst;
    return info;
  }
  if (operation->hasTrait<TTLTileUnaryOpTrait>()) {
    info.primitive = TilePrimitive::ElementwiseUnary;
    info.operandRoutes[0] = TileOperandRoute::Dst;
    return info;
  }
  return failure();
}

LogicalResult verifyTileExecutionInfo(Operation *operation,
                                      const TileExecutionInfo &info) {
  if (info.primitive == TilePrimitive::Unknown) {
    operation->emitOpError("does not define a tile execution primitive");
    return failure();
  }
  if (info.operandRoutes.size() != operation->getNumOperands()) {
    operation->emitOpError() << "defines " << info.operandRoutes.size()
                             << " tile operand routes for "
                             << operation->getNumOperands() << " operands";
    return failure();
  }
  if (info.dstOperandsMaterializedByOperation.size() !=
      operation->getNumOperands()) {
    operation->emitOpError()
        << "defines " << info.dstOperandsMaterializedByOperation.size()
        << " DST operand materialization entries for "
        << operation->getNumOperands() << " operands";
    return failure();
  }
  return success();
}

FailureOr<TileExecutionStrategy>
getSelectedTileExecutionStrategy(Operation *operation) {
  auto strategyAttr = operation->getAttrOfType<TileExecutionStrategyAttr>(
      kTileExecutionStrategyAttrName);
  if (!strategyAttr) {
    return failure();
  }
  return strategyAttr.getValue();
}

FailureOr<TileExecutionInfo>
getSelectedTileExecutionInfo(Operation *operation) {
  auto executionOp = dyn_cast<TileExecutionOpInterface>(operation);
  if (!executionOp) {
    return failure();
  }
  if (executionOp.getLegalExecutionStrategies().empty()) {
    return executionOp.getTileExecutionInfo(std::nullopt);
  }
  FailureOr<TileExecutionStrategy> strategy =
      getSelectedTileExecutionStrategy(operation);
  if (failed(strategy)) {
    return failure();
  }
  return executionOp.getTileExecutionInfo(*strategy);
}

LogicalResult
verifyTileExecutionStrategy(Operation *operation,
                            ArrayRef<TileExecutionStrategy> legalStrategies) {
  Attribute rawStrategy = operation->getAttr(kTileExecutionStrategyAttrName);
  auto strategyAttr = dyn_cast_or_null<TileExecutionStrategyAttr>(rawStrategy);
  if (rawStrategy && !strategyAttr) {
    operation->emitOpError()
        << kTileExecutionStrategyAttrName
        << " must be a #ttl.tile_execution_strategy attribute";
    return failure();
  }
  if (legalStrategies.empty() && strategyAttr) {
    operation->emitOpError()
        << kTileExecutionStrategyAttrName
        << " is only valid on tile operations with execution-strategy "
           "alternatives";
    return failure();
  }
  if (strategyAttr &&
      !llvm::is_contained(legalStrategies, strategyAttr.getValue())) {
    operation->emitOpError() << "explicit " << kTileExecutionStrategyAttrName
                             << " is not legal for its operands";
    return failure();
  }
  return success();
}

/// Return an operand route after all required strategies have been selected.
static TileOperandRoute getRequiredOperandRoute(OpOperand &operand) {
  auto executionOp = dyn_cast<TileExecutionOpInterface>(operand.getOwner());
  if (!executionOp) {
    assert(!isTileComputeOp(operand.getOwner()) &&
           "tile operation must implement TileExecutionOpInterface");
    return TileOperandRoute::None;
  }
  FailureOr<TileExecutionInfo> info =
      getSelectedTileExecutionInfo(operand.getOwner());
  assert(succeeded(info) && "tile execution strategy must be resolved");
  assert(operand.getOperandNumber() < info->operandRoutes.size() &&
         "tile execution semantics must define every operand route");
  return info->operandRoutes[operand.getOperandNumber()];
}

bool isDstInput(OpOperand &operand) {
  return getRequiredOperandRoute(operand) == TileOperandRoute::Dst;
}

bool isDstInputMaterializedByOperation(OpOperand &operand) {
  FailureOr<TileExecutionInfo> info =
      getSelectedTileExecutionInfo(operand.getOwner());
  assert(succeeded(info) && "tile execution strategy must be resolved");
  assert(operand.getOperandNumber() <
             info->dstOperandsMaterializedByOperation.size() &&
         "tile execution semantics must define every DST materialization bit");
  return info->dstOperandsMaterializedByOperation.test(
      operand.getOperandNumber());
}

LogicalResult verifyTileExecutionSemantics(Operation *root) {
  WalkResult walkResult = root->walk([&](Operation *operation) {
    auto executionOp = dyn_cast<TileExecutionOpInterface>(operation);
    if (!executionOp) {
      if (isTileComputeOp(operation)) {
        operation->emitOpError("does not implement TileExecutionOpInterface");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    SmallVector<TileExecutionStrategy, 2> legalStrategies =
        executionOp.getLegalExecutionStrategies();
    if (failed(verifyTileExecutionStrategy(operation, legalStrategies))) {
      return WalkResult::interrupt();
    }
    FailureOr<TileExecutionInfo> info = getSelectedTileExecutionInfo(operation);
    if (failed(info)) {
      if (!legalStrategies.empty()) {
        operation->emitOpError()
            << "requires a selected " << kTileExecutionStrategyAttrName
            << " attribute; run ttl-set-compute-kernel-config before DST "
               "assignment, scheduling, or lowering";
      } else {
        operation->emitOpError("has no tile execution semantics");
      }
      return WalkResult::interrupt();
    }
    return failed(verifyTileExecutionInfo(operation, *info))
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

std::optional<BcastType> getTileBroadcastType(ArrayRef<int64_t> dims,
                                              int64_t rank) {
  llvm::SmallDenseSet<int64_t> normalizedDims = normalizeDimsToSet(dims, rank);
  bool broadcastsInnermost = rank >= 1 && normalizedDims.contains(rank - 1);
  bool broadcastsSecondInnermost =
      rank >= 2 && normalizedDims.contains(rank - 2);
  if (broadcastsInnermost && broadcastsSecondInnermost) {
    return BcastType::Scalar;
  }
  if (broadcastsSecondInnermost) {
    return BcastType::Row;
  }
  if (broadcastsInnermost) {
    return BcastType::Col;
  }
  return std::nullopt;
}

FailureOr<ttkernel::ReduceDim> getReduceDimension(ArrayRef<int64_t> dims,
                                                  int64_t rank) {
  if (rank < 2) {
    return failure();
  }
  llvm::SmallDenseSet<int64_t> normalizedDims = normalizeDimsToSet(dims, rank);
  // TTKernel names the surviving orientation: reducing height uses a column
  // reduction, while reducing width uses a row reduction.
  bool reducesSecondInnermost = normalizedDims.contains(rank - 2);
  bool reducesInnermost = normalizedDims.contains(rank - 1);
  if (reducesSecondInnermost && reducesInnermost) {
    return ttkernel::ReduceDim::Scalar;
  }
  if (reducesSecondInnermost) {
    return ttkernel::ReduceDim::Col;
  }
  if (reducesInnermost) {
    return ttkernel::ReduceDim::Row;
  }
  return failure();
}

FailureOr<SelectedPipeRecords> getSelectedPipeRecords(Value pipe) {
  pipe = traceUnrealizedCasts(pipe);
  if (auto selectedSrc = pipe.getDefiningOp<SelectPipeSrcOp>()) {
    return SelectedPipeRecords{selectedSrc.getRecords(), nullptr};
  }
  if (auto selectedDst = pipe.getDefiningOp<SelectPipeDstOp>()) {
    return SelectedPipeRecords{selectedDst.getRecords(), nullptr};
  }

  auto blockArgument = mlir::dyn_cast<BlockArgument>(pipe);
  if (!blockArgument || blockArgument.getArgNumber() != 0) {
    return failure();
  }
  Operation *owner = blockArgument.getOwner()->getParentOp();
  if (auto foreachSrc = mlir::dyn_cast<PipeNetForeachSrcOp>(owner)) {
    return SelectedPipeRecords{foreachSrc.getRecords(), owner};
  }
  if (auto foreachDst = mlir::dyn_cast<PipeNetForeachDstOp>(owner)) {
    return SelectedPipeRecords{foreachDst.getRecords(), owner};
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// DST access interface defaults
//===----------------------------------------------------------------------===//

static bool isTileValue(Value value) {
  return isa<ttcore::TileType>(value.getType());
}

/// A block matmul reports one output slot before block expansion and an `M*N`
/// range after `LowerMatmulCompute` has replaced tile operands with tensors.
static int64_t getMatmulBlockOutputTileCount(TileMatmulBlockOp op) {
  auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
  if (!lhsType || !rhsType || lhsType.getRank() < 2 || rhsType.getRank() < 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
    return 1;
  }
  return lhsType.getDimSize(0) * rhsType.getDimSize(1);
}

/// Interface defaults require resolved DST operands because callers use this
/// after DST assignment, where unresolved tile residency is invalid IR.
static LogicalResult
appendDstOperandFootprint(SmallVectorImpl<DstFootprint> &footprints,
                          Value operand) {
  if (!isTileValue(operand)) {
    return success();
  }
  FailureOr<DstFootprint> footprint = getDstFootprint(operand);
  if (failed(footprint)) {
    return failure();
  }
  footprints.push_back(*footprint);
  return success();
}

FailureOr<SmallVector<DstFootprint, 2>>
getDefaultDstReadFootprints(Operation *op) {
  SmallVector<DstFootprint, 2> footprints;
  FailureOr<TileExecutionInfo> info = getSelectedTileExecutionInfo(op);
  if (failed(info)) {
    return failure();
  }
  for (OpOperand &operand : op->getOpOperands()) {
    if (info->operandRoutes[operand.getOperandNumber()] !=
        TileOperandRoute::Dst) {
      continue;
    }
    if (failed(appendDstOperandFootprint(footprints, operand.get()))) {
      return failure();
    }
  }
  return footprints;
}

/// Most tile ops write one explicit `dst_index`; block matmul is the current
/// multi-slot writer and stores only read DST for packing.
SmallVector<DstFootprint, 2> getDefaultDstWriteFootprints(Operation *op) {
  if (isa<TileStoreOp, DstIndexOp>(op)) {
    return {};
  }
  if (auto matmul = dyn_cast<TileMatmulBlockOp>(op)) {
    return {{matmul.getDstIndex(), getMatmulBlockOutputTileCount(matmul)}};
  }
  if (auto dstIndex = getTileOpDstIndex(op)) {
    return {{*dstIndex, 1}};
  }
  return {};
}

/// Result residency is separate from writes so index-like ops can name a DST
/// slot without emitting a write.
FailureOr<DstFootprint> getDefaultResultDstFootprint(Operation *op,
                                                     Value result) {
  if (!llvm::is_contained(op->getResults(), result) || !isTileValue(result)) {
    return failure();
  }
  if (auto index = dyn_cast<DstIndexOp>(op)) {
    return DstFootprint{index.getDstIndex(), 1};
  }
  if (auto matmul = dyn_cast<TileMatmulBlockOp>(op)) {
    return DstFootprint{matmul.getDstIndex(),
                        getMatmulBlockOutputTileCount(matmul)};
  }
  if (auto dstIndex = getTileOpDstIndex(op)) {
    return DstFootprint{*dstIndex, 1};
  }
  return failure();
}

/// Resolve a tile SSA value through its defining op's DST access interface.
FailureOr<DstFootprint> getDstFootprint(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return failure();
  }
  auto dstAccess = dyn_cast<DstAccessOpInterface>(definingOp);
  if (!dstAccess) {
    return failure();
  }
  return dstAccess.getResultDstFootprint(value);
}

/// Consumers that lower to TTKernel source operands require exactly one
/// concrete DST slot.
FailureOr<int64_t> getSingleConstantDstIndex(Value value) {
  FailureOr<DstFootprint> footprint = getDstFootprint(value);
  if (failed(footprint) || footprint->tileCount != 1) {
    return failure();
  }
  std::optional<int64_t> index = foldIndexToConstant(footprint->baseIndex);
  if (!index) {
    return failure();
  }
  return *index;
}

/// Scheduler hazards operate on concrete slots after DST assignment.
FailureOr<SmallVector<int64_t>> getConstantDstIndices(DstFootprint footprint) {
  std::optional<int64_t> base = foldIndexToConstant(footprint.baseIndex);
  if (!base || footprint.tileCount < 0) {
    return failure();
  }
  SmallVector<int64_t> indices;
  indices.reserve(footprint.tileCount);
  for (int64_t offset = 0; offset < footprint.tileCount; ++offset) {
    indices.push_back(*base + offset);
  }
  return indices;
}

static FailureOr<SmallVector<int64_t>>
getConstantDstIndices(ArrayRef<DstFootprint> footprints) {
  SmallVector<int64_t> indices;
  for (DstFootprint footprint : footprints) {
    FailureOr<SmallVector<int64_t>> expanded = getConstantDstIndices(footprint);
    if (failed(expanded)) {
      return failure();
    }
    llvm::append_range(indices, *expanded);
  }
  return indices;
}

FailureOr<SmallVector<int64_t>> getConstantDstReadIndices(Operation *op) {
  auto dstAccess = dyn_cast<DstAccessOpInterface>(op);
  if (!dstAccess) {
    return SmallVector<int64_t>{};
  }
  FailureOr<SmallVector<DstFootprint, 2>> footprints =
      dstAccess.getDstReadFootprints();
  if (failed(footprints)) {
    return failure();
  }
  return getConstantDstIndices(*footprints);
}

FailureOr<SmallVector<int64_t>> getConstantDstWriteIndices(Operation *op) {
  auto dstAccess = dyn_cast<DstAccessOpInterface>(op);
  if (!dstAccess) {
    return SmallVector<int64_t>{};
  }
  return getConstantDstIndices(dstAccess.getDstWriteFootprints());
}

//===----------------------------------------------------------------------===//
// Tile operation classification
//===----------------------------------------------------------------------===//

TileOpCategory classifyTileOp(Operation *op) {
  if (isa<DstIndexOp>(op)) {
    return TileOpCategory::DstIndex;
  }
  if (isa<CopyTileOp>(op)) {
    return TileOpCategory::CopyTile;
  }
  if (isa<CopyDstOp>(op)) {
    return TileOpCategory::CopyDst;
  }
  if (isa<TileBcastOp>(op)) {
    return TileOpCategory::Bcast;
  }
  if (isa<TileMatmulBlockOp>(op)) {
    return TileOpCategory::FPUBinary;
  }
  if (auto accumulate = dyn_cast<TileAccumulateOp>(op)) {
    return hasDstBackedTileProducer(accumulate.getContribution())
               ? TileOpCategory::SFPUBinary
               : TileOpCategory::FPUBinary;
  }
  if (isa<TileTransposeOp>(op)) {
    return TileOpCategory::Transpose;
  }
  if (isa<TileFillOp>(op)) {
    return TileOpCategory::Fill;
  }

  if (op->hasTrait<TTLStrategyDependentBinaryOpTrait>()) {
    FailureOr<TileExecutionStrategy> strategy =
        getSelectedTileExecutionStrategy(op);
    assert(succeeded(strategy) && "tile execution strategy must be resolved");
    return *strategy == TileExecutionStrategy::FPU ? TileOpCategory::FPUBinary
                                                   : TileOpCategory::SFPUBinary;
  }
  // SFPU unary: tile unary ops that operate in-place on DST.
  if (op->hasTrait<TTLTileUnaryOpTrait>()) {
    return TileOpCategory::SFPUUnary;
  }
  // SFPU binary: tile binary ops that read both operands from DST.
  if (op->hasTrait<TTLTileBinaryOpTrait>()) {
    return TileOpCategory::SFPUBinary;
  }
  return TileOpCategory::Unknown;
}

FusionTraceResult traceFusionToRoots(
    mlir::Value value,
    llvm::function_ref<bool(mlir::OpOperand &)> isMaterializationPlanned) {
  FusionTraceResult result;

  // A DFB-attached value is an available input to the fused computation.
  if (getAttachedCB(value)) {
    result.rootInputs.insert(value);
    result.lifetimeRootInputs.insert(value);
    return result;
  }

  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = value;
    return result;
  }

  // BlockBroadcastOp is a fusion leaf because its input must be DFB-attached.
  if (auto bcastOp = llvm::dyn_cast<BlockBroadcastOp>(defOp)) {
    mlir::OpOperand &inputOperand = bcastOp->getOpOperand(0);
    mlir::Value bcastInput = inputOperand.get();
    bool isInputMaterialized = isMaterializationPlanned(inputOperand);
    if (isInputMaterialized || getAttachedCB(bcastInput)) {
      result.rootInputs.insert(bcastInput);
      if (!isInputMaterialized) {
        result.lifetimeRootInputs.insert(bcastInput);
      }
      result.opsInOrder.insert(defOp);
      return result;
    }
    // The broadcast cannot be formed until its input is materialized.
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = bcastInput;
    result.failedOperand = &inputOperand;
    return result;
  }

  // MatmulOp is a fusion leaf because both inputs must be DFB-attached.
  if (auto matmulOp = llvm::dyn_cast<MatmulOp>(defOp)) {
    mlir::OpOperand &lhsOperand = matmulOp->getOpOperand(0);
    mlir::OpOperand &rhsOperand = matmulOp->getOpOperand(1);
    mlir::Value lhs = lhsOperand.get();
    mlir::Value rhs = rhsOperand.get();
    bool isLhsMaterialized = isMaterializationPlanned(lhsOperand);
    bool isRhsMaterialized = isMaterializationPlanned(rhsOperand);
    bool lhsAvailable = isLhsMaterialized || getAttachedCB(lhs);
    bool rhsAvailable = isRhsMaterialized || getAttachedCB(rhs);
    if (lhsAvailable && rhsAvailable) {
      result.rootInputs.insert(lhs);
      result.rootInputs.insert(rhs);
      if (!isLhsMaterialized) {
        result.lifetimeRootInputs.insert(lhs);
      }
      if (!isRhsMaterialized) {
        result.lifetimeRootInputs.insert(rhs);
      }
      result.opsInOrder.insert(defOp);
      return result;
    }
    // The matmul cannot be formed until both inputs are materialized.
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = lhsAvailable ? rhs : lhs;
    result.failedOperand = lhsAvailable ? &rhsOperand : &lhsOperand;
    return result;
  }

  // FillOp is a fusable leaf: it produces a value with no input operands.
  if (isa<FillOp>(defOp)) {
    result.opsInOrder.insert(defOp);
    return result;
  }

  if (!isElementwiseOp(defOp)) {
    result.failureReason = TraceFailureReason::NotFusableOp;
    result.failedValue = value;
    return result;
  }

  // Recursively trace every elementwise operand not replaced by a planned
  // materialization.
  unsigned numElementwiseOperands = getElementwiseOperands(defOp).size();
  for (unsigned operandIndex = 0; operandIndex < numElementwiseOperands;
       ++operandIndex) {
    mlir::OpOperand &operand = defOp->getOpOperand(operandIndex);
    if (isMaterializationPlanned(operand)) {
      result.rootInputs.insert(operand.get());
      continue;
    }
    auto operandTrace =
        traceFusionToRoots(operand.get(), isMaterializationPlanned);
    if (operandTrace.failureReason != TraceFailureReason::Success) {
      if (!operandTrace.failedOperand) {
        operandTrace.failedOperand = &operand;
      }
      return operandTrace;
    }
    // Merge roots and ops (SmallSetVector handles deduplication)
    for (mlir::Value root : operandTrace.rootInputs) {
      result.rootInputs.insert(root);
    }
    for (mlir::Value root : operandTrace.lifetimeRootInputs) {
      result.lifetimeRootInputs.insert(root);
    }
    for (mlir::Operation *op : operandTrace.opsInOrder) {
      result.opsInOrder.insert(op);
    }
  }

  // Add this op at the end (after all its dependencies)
  result.opsInOrder.insert(defOp);

  return result;
}

FusionTraceResult traceFusionToRoots(mlir::Value value) {
  return traceFusionToRoots(value, [](mlir::OpOperand &) { return false; });
}

llvm::StringRef describeTraceFailure(TraceFailureReason reason) {
  switch (reason) {
  case TraceFailureReason::Success:
    return "success";
  case TraceFailureReason::NotCBAttached:
    return "value is not attached to a circular buffer";
  case TraceFailureReason::NotFusableOp:
    return "cannot trace through non-fusable op";
  }
  llvm_unreachable("unhandled TraceFailureReason");
}

//===----------------------------------------------------------------------===//
// Loop grouping for L1 accumulation and init selection
//===----------------------------------------------------------------------===//

namespace ttk = mlir::tt::ttkernel;

llvm::SmallDenseSet<Value, 2> getPackTileCBs(scf::ForOp loop) {
  llvm::SmallDenseSet<Value, 2> cbs;
  loop->walk([&](ttk::PackTileOp packOp) { cbs.insert(packOp.getOutCb()); });
  loop->walk(
      [&](ttk::PackTileBlockOp packOp) { cbs.insert(packOp.getOutCb()); });
  return cbs;
}

bool sharePackCB(scf::ForOp loopA, scf::ForOp loopB) {
  auto cbsA = getPackTileCBs(loopA);
  auto cbsB = getPackTileCBs(loopB);
  for (auto cb : cbsA) {
    if (cbsB.contains(cb)) {
      return true;
    }
  }
  return false;
}

} // namespace mlir::tt::ttl
