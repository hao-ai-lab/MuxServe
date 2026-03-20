#include "batch_scheduler.h"
#include <algorithm>
#include <torch/extension.h>

BatchScheduler::BatchScheduler(int numLayers, int numHeads, int maxSeqLen,
                               int blockSize)
    : numLayers_(numLayers), numHeads_(numHeads), maxSeqLen_(maxSeqLen),
      KVBlockSize_(blockSize) {
  maxBocksPerSeq_ = (maxSeqLen_ + KVBlockSize_ - 1) / KVBlockSize_;
}

void BatchScheduler::addRequest(py::array_t<int> &prompt, int reqId,
                                int maxTokens) {
  auto request =
      std::make_shared<Request>(reqId, maxTokens, prompt, KVBlockSize_);
  request->allocatedBlocks_ =
      py::array_t<int>({maxBocksPerSeq_, numLayers_, numHeads_});
  requests_[reqId] = request;
}

py::array_t<int>
BatchScheduler::tryBatch(std::vector<int> &batchRequestIds,
                         std::vector<int> &batchLastOutputTokens) {
  bool inDecoding = !batchLastOutputTokens.empty();
  int size = batchRequestIds.size() * 3 + 1;
  int totalBlocksRequest = 0;
  /* Encode the batch information into a numpy array
   * Each row contains the following information:
   *   - request id
   *   - number of blocks needed
   *   - prompt blocks allocated(1) or not(0)
   */
  auto blockInfo = py::array_t<int>({size});
  auto blockInfoPtr = blockInfo.mutable_data();
  for (int i = 0; i < batchRequestIds.size(); i++) {
    auto request = requests_[batchRequestIds[i]];
    int numBlocksNeeded = request->logicalBlockIdx_ + 1;
    if (inDecoding) {
      numBlocksNeeded = request->appendLogicalToken(batchLastOutputTokens[i]);
    }
    blockInfoPtr[i * 3] = request->idx_;
    blockInfoPtr[i * 3 + 1] = numBlocksNeeded;
    blockInfoPtr[i * 3 + 2] = std::min(request->numPhysicalBlocks_, 1);
    totalBlocksRequest += numBlocksNeeded;
  }
  blockInfoPtr[size - 1] = totalBlocksRequest;
  return blockInfo;
}

std::vector<int>
BatchScheduler::getBatchInfo(py::array_t<int> &batchBlockRequest,
                             py::array_t<int> &blockInfo) {
  std::vector<int> preemptRequests, promptInfo;
  int numTokens = 0;
  int numContexts = 0;
  int maxContextLen = 0;
  int maxNumBlocksPerSeq = 0;

  auto batchBlockRequestPtr = batchBlockRequest.data();
  auto blockInfoPtr = blockInfo.data();

  int numRequestsInBatch = batchBlockRequest.size() / 3;
  int preemptIdx = blockInfoPtr[0];
  int blockOffset = 1;
  for (int i = 0; i < preemptIdx; i++) {
    int requestId = batchBlockRequestPtr[i * 3];
    int numBlocksNeeded =
        batchBlockRequestPtr[i * 3 + 1] * numLayers_ * numHeads_;
    auto request = requests_[requestId];
    if (numBlocksNeeded > 0) {
      auto reqBlockPtr = request->allocatedBlocks_.mutable_data();
      int offset = request->numPhysicalBlocks_ * numLayers_ * numHeads_;
      std::memcpy(reqBlockPtr + offset, blockInfoPtr + blockOffset,
                  numBlocksNeeded * sizeof(int));
      blockOffset += numBlocksNeeded;
      request->numPhysicalBlocks_ += batchBlockRequestPtr[i * 3 + 1];
    }

    if (request->outputs_.size() > 0) {
      maxNumBlocksPerSeq =
          std::max(maxNumBlocksPerSeq, request->numPhysicalBlocks_);
      maxContextLen = std::max(maxContextLen, request->contextLen_);
      numContexts += 1;
      numTokens += 1;
    } else {
      promptInfo.push_back(request->prompts_.size());
      numTokens += request->prompts_.size();
    }
  }

  std::vector<int> batchInfo({numTokens, numContexts, maxContextLen,
                              maxNumBlocksPerSeq, numLayers_, numHeads_});
  for (int i = 0; i < promptInfo.size(); i++) {
    batchInfo.push_back(promptInfo[i]);
  }
  return batchInfo;
}

void BatchScheduler::getBatch(py::array_t<int> &batchInfo,
                              py::array_t<int> &blockInfo,
                              torch::Tensor &tokenTensor,
                              torch::Tensor &tokenPositionTensor,
                              torch::Tensor &contextLenTensor,
                              torch::Tensor &blockTableTensor,
                              torch::Tensor &slotMappingTensor) {
  int numTokens = 0;
  int maxContextLen = 0;
  int maxNumBlocksPerSeq = 0;
  std::vector<int> tokenArray, tokenPositionArray, contextLenArray;
  std::vector<int64_t> slotMappingArray;
  preemptRequests_.clear();
  promptInfo_.clear();
  batchReqs_.clear();

  auto batchInfoPtr = batchInfo.data();
  auto blockInfoPtr = blockInfo.data();

  int numRequestsInBatch = batchInfo.size() / 3;
  int preemptIdx = blockInfoPtr[0];
  for (int i = 0; i < numRequestsInBatch; i++) {
    int requestId = batchInfoPtr[i * 3];
    int numBlocksNeeded = batchInfoPtr[i * 3 + 1] * numLayers_ * numHeads_;
    if (i >= preemptIdx) {
      preemptRequests_.push_back(requestId);
      if (requests_.find(requestId) != requests_.end()) {
        requests_.erase(requestId);
      }
      continue;
    }

    auto request = requests_[requestId];
    if (request->outputs_.size() > 0) {
      maxNumBlocksPerSeq =
          std::max(maxNumBlocksPerSeq, request->numPhysicalBlocks_);
      tokenArray.push_back(request->outputs_[request->outputs_.size() - 1]);
      tokenPositionArray.push_back(request->contextLen_ - 1);
      contextLenArray.push_back(request->contextLen_);
      maxContextLen = std::max(maxContextLen, request->contextLen_);
      request->getLastEmptySlot(numLayers_, numHeads_, slotMappingArray);
      numTokens += 1;
    } else {
      tokenArray.resize(tokenArray.size() + request->prompts_.size());
      std::memcpy(tokenArray.data() + numTokens, request->prompts_.data(),
                  request->prompts_.size() * sizeof(int));
      tokenPositionArray.resize(tokenArray.size());
      std::iota(tokenPositionArray.begin() + numTokens,
                tokenPositionArray.end(), 0);
      promptInfo_.push_back(request->prompts_.size());
      request->getPromptSlots(numLayers_, numHeads_, slotMappingArray);
      numTokens += request->prompts_.size();
    }
    batchReqs_.push_back({requestId, request->maxTokens_});
  }
  promptInfo_.push_back(maxContextLen);

  py::array_t<int> blockTableArray(
      {numTokens, maxNumBlocksPerSeq, numLayers_, numHeads_});
  auto blockTablePtr = blockTableArray.mutable_data();
  if (maxNumBlocksPerSeq > 0) {
    for (int i = 0; i < preemptIdx; i++) {
      int requestId = batchInfoPtr[i * 3];
      auto request = requests_[requestId];

      auto reqBlockPtr = request->allocatedBlocks_.data();
      int offset = i * maxNumBlocksPerSeq * numLayers_ * numHeads_;
      int numBlocks = request->numPhysicalBlocks_ * numLayers_ * numHeads_;
      std::memcpy(blockTablePtr + offset, reqBlockPtr, numBlocks * sizeof(int));
    }
  }

  // copy to tensor
  int padLen = tokenTensor.numel() - tokenArray.size();
  auto tokenTensorPtr = tokenTensor.data_ptr<int32_t>();
  auto tokenPositionTensorPtr = tokenPositionTensor.data_ptr<int32_t>();
  auto contextLenTensorPtr = contextLenTensor.data_ptr<int32_t>();
  auto slotMappingTensorPtr = slotMappingTensor.data_ptr<int64_t>();
  auto blockTableTensorPtr = blockTableTensor.data_ptr<int32_t>();
  std::memcpy(tokenTensorPtr, tokenArray.data(),
              tokenArray.size() * sizeof(int));
  std::memcpy(tokenPositionTensorPtr, tokenPositionArray.data(),
              tokenPositionArray.size() * sizeof(int));
  if (padLen > 0) {
    std::memset(tokenTensorPtr + tokenArray.size(), 0, padLen * sizeof(int));
    std::memset(tokenPositionTensorPtr + tokenPositionArray.size(), 0,
                padLen * sizeof(int));
  }
  std::memcpy(contextLenTensorPtr, contextLenArray.data(),
              contextLenArray.size() * sizeof(int));
  std::memcpy(slotMappingTensorPtr, slotMappingArray.data(),
              slotMappingArray.size() * sizeof(int64_t));
  std::memcpy(blockTableTensorPtr, blockTablePtr,
              blockTableArray.size() * sizeof(int));
}

void BatchScheduler::releaseRequests(std::vector<int> &requestIds) {
  for (int i = 0; i < requestIds.size(); i++) {
    requests_.erase(requestIds[i]);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<BatchScheduler>(m, "BatchScheduler")
      .def(py::init<int, int, int, int>())
      .def("add_request", &BatchScheduler::addRequest)
      .def("try_batch", &BatchScheduler::tryBatch)
      .def("get_batch_info", &BatchScheduler::getBatchInfo)
      .def("get_batch", &BatchScheduler::getBatch)
      .def("release_requests", &BatchScheduler::releaseRequests)
      .def("get_preempt_requests", &BatchScheduler::getPreemptRequests)
      .def("get_prompt_info", &BatchScheduler::getPromptInfo)
      .def("get_batch_reqs", &BatchScheduler::getBatchReqs);
}
