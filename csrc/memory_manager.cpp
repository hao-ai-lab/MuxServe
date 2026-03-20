#include "memory_manager.h"
#include <torch/extension.h>

KVStorage::KVStorage(int numBlocks)
    : numBlocks_(numBlocks), freeBlocks_(numBlocks), newBlocksAllocated_(0),
      numBlocksAllocated_(0) {
  for (int i = 0; i < numBlocks_; i++) {
    freeBlocks_[i] = i;
  }
}

py::array_t<int> KVStorage::allocate(int size) {
  // create a numpy array with the allocated blocks
  auto allocatedBlocks = py::array_t<int>({size});
  auto blockAccessor = allocatedBlocks.mutable_unchecked<1>();

  for (int i = 0; i < size; i++) {
    int blockId = freeBlocks_.back();
    blockAccessor(i) = blockId;
    freeBlocks_.pop_back();
  }
  return allocatedBlocks;
}

py::array_t<int> KVStorage::allocateBatch(py::array_t<int> &batchAllocInfo,
                                          int numLayers, int numHeads) {
  int numRequests = batchAllocInfo.size() / 3;
  int numFreeBlocks = freeBlocks_.size();

  int totalNewBlocksNeeded = 0;
  std::vector<int> numBlocksToAllocate;
  std::vector<int> totalBlocksRequest(numRequests + 1, {0});
  std::vector<int> newBlocksNeededList, oldBlocksAllocatedList;
  auto blockAllocInfoPtr = batchAllocInfo.data();
  for (int i = 0; i < numRequests; i++) {
    int requestId = blockAllocInfoPtr[i * 3];
    int numBlocksRequest = blockAllocInfoPtr[i * 3 + 1] * numLayers * numHeads;
    int promptAllocated = blockAllocInfoPtr[i * 3 + 2];

    int numNewBlocksNeeded = numBlocksRequest;
    if (!promptAllocated) {
      if (allocatedBlocks_.find(requestId) != allocatedBlocks_.end()) {
        numNewBlocksNeeded -= allocatedBlocks_[requestId].size();
        oldBlocksAllocatedList.push_back(allocatedBlocks_[requestId].size());
      } else {
        oldBlocksAllocatedList.push_back(0);
      }
    } else {
      oldBlocksAllocatedList.push_back(allocatedBlocks_[requestId].size());
    }
    totalNewBlocksNeeded += numNewBlocksNeeded;
    newBlocksNeededList.push_back(totalNewBlocksNeeded);
    numBlocksToAllocate.push_back(numNewBlocksNeeded);
    totalBlocksRequest[i + 1] = totalBlocksRequest[i] + numBlocksRequest;
  }

  // calculate the suffix sum of allocated blocks
  std::vector<int> suffixSum(numRequests + 1);
  suffixSum[numRequests] = 0;
  for (int i = numRequests - 1; i >= 0; i--) {
    suffixSum[i] = suffixSum[i + 1] + oldBlocksAllocatedList[i];
  }

  // try to allocate the blocks with preemption enabled
  int preemptIdx;
  for (preemptIdx = numRequests; preemptIdx >= 0; preemptIdx--) {
    int freeBlocksExpected = numFreeBlocks + suffixSum[preemptIdx];
    if (preemptIdx == 0 ||
        newBlocksNeededList[preemptIdx - 1] <= freeBlocksExpected) {
      break;
    }
  }

  // preempted requests
  int freedBlocks = 0;
  for (int i = preemptIdx; i < numRequests; i++) {
    int requestId = blockAllocInfoPtr[i * 3];
    if (oldBlocksAllocatedList[i] > 0) {
      freedBlocks += oldBlocksAllocatedList[i];
      freeBlocks_.insert(freeBlocks_.end(), allocatedBlocks_[requestId].begin(),
                         allocatedBlocks_[requestId].end());
      allocatedBlocks_.erase(requestId);
    }
  }
  numBlocksAllocated_ -= freedBlocks;
  newBlocksAllocated_ = -freedBlocks;

  // allocate the blocks
  if (preemptIdx <= 0) {
    auto batchBlocks = py::array_t<int>({1});
    auto batchBlocksPtr = batchBlocks.mutable_data();
    batchBlocksPtr[0] = preemptIdx;
    return batchBlocks;
  }

  auto batchBlocks = py::array_t<int>({totalBlocksRequest[preemptIdx] + 1});
  auto batchBlocksPtr = batchBlocks.mutable_data();
  // Encode batchBlocks:
  //  batchBlocks[0]: number of requests
  //  batchBlocks[1:]: block ids allocated
  batchBlocksPtr[0] = preemptIdx;
  assert(freeBlocks_.size() >= newBlocksNeededList[preemptIdx - 1]);

  int offset = 0;
  int batchOffset = 1;
  auto freeBlocksPtr = freeBlocks_.end() - newBlocksNeededList[preemptIdx - 1];
  for (int i = 0; i < preemptIdx; i++) {
    int numNewBlocksNeeded = numBlocksToAllocate[i];
    int requestId = blockAllocInfoPtr[i * 3];
    int numBlocksRequest = blockAllocInfoPtr[i * 3 + 1] * numLayers * numHeads;
    if (numNewBlocksNeeded > 0) {
      if (blockAllocInfoPtr[i * 3 + 2]) {
        allocatedBlocks_[requestId].insert(
            allocatedBlocks_[requestId].end(), freeBlocksPtr + offset,
            freeBlocksPtr + offset + numNewBlocksNeeded);
      } else {
        if (allocatedBlocks_.find(requestId) == allocatedBlocks_.end()) {
          allocatedBlocks_[requestId] =
              std::vector<int>(freeBlocksPtr + offset,
                               freeBlocksPtr + offset + numNewBlocksNeeded);
        } else {
          allocatedBlocks_[requestId].insert(
              allocatedBlocks_[requestId].end(), freeBlocksPtr + offset,
              freeBlocksPtr + offset + numNewBlocksNeeded);
        }
      }
      offset += numNewBlocksNeeded;
    }
    if (numBlocksRequest > 0) {
      assert(allocatedBlocks_[requestId].size() >= numBlocksRequest);
      std::memcpy(batchBlocksPtr + batchOffset,
                  &*(allocatedBlocks_[requestId].end() - numBlocksRequest),
                  numBlocksRequest * sizeof(int));
      batchOffset += numBlocksRequest;
    }
  }
  newBlocksAllocated_ += newBlocksNeededList[preemptIdx - 1];
  numBlocksAllocated_ += newBlocksAllocated_;
  freeBlocks_.erase(freeBlocks_.end() - newBlocksNeededList[preemptIdx - 1],
                    freeBlocks_.end());
  return batchBlocks;
}

int KVStorage::freeBatch(std::vector<int> &batchRequestIds) {
  int freedBlocks = 0;
  for (int i = 0; i < batchRequestIds.size(); i++) {
    int requestId = batchRequestIds[i];
    if (allocatedBlocks_.find(requestId) != allocatedBlocks_.end()) {
      freedBlocks += allocatedBlocks_[requestId].size();
      freeBlocks_.insert(freeBlocks_.end(), allocatedBlocks_[requestId].begin(),
                         allocatedBlocks_[requestId].end());
      allocatedBlocks_.erase(requestId);
    }
  }
  numBlocksAllocated_ -= freedBlocks;
  return freedBlocks;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<KVStorage>(m, "KVStorage")
      .def(py::init<int>())
      .def("allocate", &KVStorage::allocate)
      .def("allocate_batch", &KVStorage::allocateBatch)
      .def("free_batch", &KVStorage::freeBatch)
      .def("get_new_blocks_allocated", &KVStorage::newBlocksAllocated);
}
