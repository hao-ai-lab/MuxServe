#include <map>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;

class KVStorage {
public:
  KVStorage(int numBlocks);

  py::array_t<int> allocate(int size);

  py::array_t<int> allocateBatch(py::array_t<int> &batchAllocInfo,
                                 int numLayers, int numHeads);

  int newBlocksAllocated() { return newBlocksAllocated_; }

  int freeBatch(std::vector<int> &batchRequestIds);

  int numBlocks_;
  std::vector<int> freeBlocks_;
  std::map<int, std::vector<int>> allocatedBlocks_;

  int numBlocksAllocated_;
  // temporary record for logging
  int newBlocksAllocated_;
};
