#include <map>
#include <memory>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;

struct Request {
  int idx_;
  int maxTokens_;
  int contextLen_;
  int blockSize_;
  int numPhysicalBlocks_;
  int logicalBlockIdx_;
  int logicalBlockOffset_;
  py::array_t<int> prompts_;
  std::vector<int> outputs_;
  // shape: [maxBocksPerSeq, numLayers, numHeads]
  py::array_t<int> allocatedBlocks_;

  Request(int reqId, int maxTokens, py::array_t<int> &prompts, int blockSize)
      : idx_(reqId), maxTokens_(maxTokens), blockSize_(blockSize),
        numPhysicalBlocks_(0), prompts_(prompts), outputs_({}) {
    int promptLen = prompts_.size();
    logicalBlockIdx_ = (promptLen + blockSize_ - 1) / blockSize_ - 1;
    logicalBlockOffset_ = (promptLen - 1) % blockSize_;
    contextLen_ = promptLen;
  }

  int appendLogicalToken(int tokenId) {
    outputs_.push_back(tokenId);
    contextLen_++;
    logicalBlockOffset_++;
    if (logicalBlockOffset_ == blockSize_) {
      logicalBlockIdx_++;
      logicalBlockOffset_ = 0;
    }
    return logicalBlockIdx_ + 1 - numPhysicalBlocks_;
  }

  void getPromptSlots(int numLayers, int numHeads,
                      std::vector<int64_t> &slotMapping) {
    int blocksPerToken = numLayers * numHeads;
    auto allocatedBlocksPtr = allocatedBlocks_.mutable_data();
    for (int i = 0; i < prompts_.size(); i++) {
      int blockIdx = i / blockSize_;
      int blockOffset = i % blockSize_;
      for (int j = 0; j < blocksPerToken; j++) {
        slotMapping.push_back(
            int64_t(allocatedBlocksPtr[blockIdx * blocksPerToken + j]) *
                blockSize_ +
            blockOffset);
      }
    }
  }

  void getLastEmptySlot(int numLayers, int numHeads,
                        std::vector<int64_t> &slotMapping) {
    auto allocatedBlocksPtr = allocatedBlocks_.mutable_data();
    int offset = (numPhysicalBlocks_ - 1) * numLayers * numHeads;
    for (int i = 0; i < numLayers * numHeads; i++) {
      slotMapping.push_back(int64_t(allocatedBlocksPtr[offset + i]) *
                                blockSize_ +
                            logicalBlockOffset_);
    }
  }
};

class BatchScheduler {
public:
  BatchScheduler(int numLayers, int numHeads, int maxSeqLen, int blockSize);

  void addRequest(py::array_t<int> &prompt, int reqId, int maxTokens);

  py::array_t<int> tryBatch(std::vector<int> &batchRequestIds,
                            std::vector<int> &batchLastOutputTokens);

  std::vector<int> getBatchInfo(py::array_t<int> &batchBlockRequest,
                                py::array_t<int> &blockInfo);

  void getBatch(py::array_t<int> &batchInfo, py::array_t<int> &blockInfo,
                torch::Tensor &tokenTensor, torch::Tensor &tokenPositionTensor,
                torch::Tensor &contextLenTensor,
                torch::Tensor &blockTableTensor,
                torch::Tensor &slotMappingTensor);

  void releaseRequests(std::vector<int> &requestIds);

  std::vector<int> getPreemptRequests() { return preemptRequests_; }

  std::vector<int> getPromptInfo() { return promptInfo_; }

  std::vector<std::vector<int>> getBatchReqs() { return batchReqs_; }

  int numLayers_;
  int numHeads_;
  int maxSeqLen_;
  int KVBlockSize_;
  int maxBocksPerSeq_;
  std::map<int, std::shared_ptr<Request>> requests_;
  // placeholder for batch info
  std::vector<int> preemptRequests_;
  std::vector<int> promptInfo_;
  std::vector<std::vector<int>> batchReqs_;
};
