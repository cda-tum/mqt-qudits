/*
 * This file is part of the MQT DD Package which is released under the MIT
 * license. See file README.md or go to
 * https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_COMPLEXCACHE_HPP
#define DD_PACKAGE_COMPLEXCACHE_HPP

#include "Complex.hpp"
#include "ComplexTable.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

namespace dd {

template <std::size_t INITIAL_ALLOCATION_SIZE = 2048,
          std::size_t GROWTH_FACTOR = 2>
class ComplexCache {
  using Entry = ComplexTable<>::Entry;

public:
  ComplexCache() : allocationSize(INITIAL_ALLOCATION_SIZE) {
    // allocate first chunk of cache entries
    chunks.emplace_back(allocationSize);
    allocations += allocationSize;
    allocationSize *= GROWTH_FACTOR;
    chunkIt = chunks[0].begin();
    chunkEndIt = chunks[0].end();
  }

  ~ComplexCache() = default;

  // access functions
  [[nodiscard]] std::size_t getCount() const { return count; }
  [[nodiscard]] std::size_t getPeakCount() const { return peakCount; }
  [[nodiscard]] std::size_t getAllocations() const { return allocations; }
  [[nodiscard]] std::size_t getGrowthFactor() const { return GROWTH_FACTOR; }

  [[nodiscard]] Complex getCachedComplex() {
    // an entry is available on the stack
    if (available != nullptr) {
      assert(available->next != nullptr);
      auto entry = Complex{available, available->next};
      available = entry.img->next;
      count += 2;
      return entry;
    }

    // new chunk has to be allocated
    if (chunkIt == chunkEndIt) {
      chunks.emplace_back(allocationSize);
      allocations += allocationSize;
      allocationSize *= GROWTH_FACTOR;
      chunkID++;
      chunkIt = chunks[chunkID].begin();
      chunkEndIt = chunks[chunkID].end();
    }

    Complex c{};
    c.real = &(*chunkIt);
    ++chunkIt;
    c.img = &(*chunkIt);
    ++chunkIt;
    count += 2;
    return c;
  }

  [[nodiscard]] Complex getTemporaryComplex() {
    // an entry is available on the stack
    if (available != nullptr) {
      assert(available->next != nullptr);
      return {available, available->next};
    }

    // new chunk has to be allocated
    if (chunkIt == chunkEndIt) {
      chunks.emplace_back(allocationSize);
      allocations += allocationSize;
      allocationSize *= GROWTH_FACTOR;
      chunkID++;
      chunkIt = chunks[chunkID].begin();
      chunkEndIt = chunks[chunkID].end();
    }
    return {&(*chunkIt), &(*(chunkIt + 1))};
  }

  void returnToCache(Complex& c) {
    assert(count >= 2);
    assert(c != Complex::zero);
    assert(c != Complex::one);
    assert(c.real->refCount == 0);
    assert(c.img->refCount == 0);
    c.img->next = available;
    c.real->next = c.img;
    available = c.real;
    count -= 2;
  }

  void clear() {
    // clear available stack
    available = nullptr;

    // release memory of all but the first chunk TODO: it could be desirable to
    // keep the memory
    while (chunkID > 0) {
      chunks.pop_back();
      chunkID--;
    }
    // restore initial chunk setting
    chunkIt = chunks[0].begin();
    chunkEndIt = chunks[0].end();
    allocationSize = INITIAL_ALLOCATION_SIZE * GROWTH_FACTOR;
    allocations = INITIAL_ALLOCATION_SIZE;

    count = 0;
    peakCount = 0;
  };

private:
  Entry* available{};
  std::vector<std::vector<Entry>> chunks{};
  std::size_t chunkID{0};
  typename std::vector<Entry>::iterator chunkIt;
  typename std::vector<Entry>::iterator chunkEndIt;
  std::size_t allocationSize;

  std::size_t allocations = 0;
  std::size_t count = 0;
  std::size_t peakCount = 0;
};
} // namespace dd

#endif // DD_PACKAGE_COMPLEXCACHE_HPP
