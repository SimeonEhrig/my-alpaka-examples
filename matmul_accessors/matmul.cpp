#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <cstdlib>
#include <iostream>

namespace alpakaex = alpaka::experimental;

template <typename TIdx> struct InitKernel {
  template <typename TAcc, typename TData>
  ALPAKA_FN_ACC auto
  operator()(TAcc const &acc,
             alpakaex::BufferAccessor<TAcc, TData, 2, alpakaex::WriteAccess>
                 data) const {
    auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    auto const gridSize =
        alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

    for (TIdx y = idx[0]; y < data.extents[0]; y += gridSize[0]) {
      for (TIdx x = idx[1]; x < data.extents[1]; x += gridSize[1]) {
        data(y, x) = y * data.extents[0] + x;
      }
    }
  }
};

template <typename TIdx> struct MatmulKernel {
  template <typename TAcc, typename TData>
  ALPAKA_FN_ACC auto operator()(
      TAcc const &acc,
      alpakaex::BufferAccessor<TAcc, TData, 2, alpakaex::ReadAccess> A,
      alpakaex::BufferAccessor<TAcc, TData, 2, alpakaex::ReadAccess> B,
      alpakaex::BufferAccessor<TAcc, TData, 2, alpakaex::WriteAccess> C) const {
    auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    auto const gridSize =
        alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

    TData sum;
    for (TIdx y = idx[0]; y < A.extents[0]; y += gridSize[0]) {
      for (TIdx x = idx[1]; x < A.extents[1]; x += gridSize[1]) {
        sum = static_cast<TData>(0);
        for (TIdx k = 0; k < A.extents[0]; ++k) {
          sum += A(y, k) * B(k, x);
        }
        C(y, x) = sum;
      }
    }
  }
};

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "please set a size\n";
    return 1;
  }

  using Dim = alpaka::DimInt<2>;
  using Idx = int;
  using Data = float;
  const Idx size = static_cast<Idx>(std::atoi(argv[1]));

  using Host = alpaka::AccCpuSerial<Dim, Idx>;
  using HostQueueProperty = alpaka::Blocking;
  using HostQueue = alpaka::Queue<Host, HostQueueProperty>;

  using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
  using AccQueueProperty = alpaka::Blocking;
  using DevQueue = alpaka::Queue<Acc, AccQueueProperty>;

  auto const devHost = alpaka::getDevByIdx<Host>(0);
  HostQueue hostQueue(devHost);

  auto const devAcc = alpaka::getDevByIdx<Acc>(0);
  DevQueue devQueue(devAcc);

  using Vec = alpaka::Vec<Dim, Idx>;
  const Vec extents(Vec::all(static_cast<Idx>(size)));

  using BufHost = alpaka::Buf<Host, Data, Dim, Idx>;

  BufHost hostC(alpaka::allocBuf<Data, Idx>(devHost, extents));

  using BufAcc = alpaka::Buf<Acc, Data, Dim, Idx>;

  BufAcc accA(alpaka::allocBuf<Data, Idx>(devAcc, extents));
  BufAcc accB(alpaka::allocBuf<Data, Idx>(devAcc, extents));
  BufAcc accC(alpaka::allocBuf<Data, Idx>(devAcc, extents));

  Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
  Vec const threadsPerGrid(Vec::all(static_cast<Idx>(10)));
  using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

  WorkDiv const hostWorkDiv = alpaka::getValidWorkDiv<Host>(
      devHost, threadsPerGrid, elementsPerThread, false,
      alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

  WorkDiv const accWorkDiv = alpaka::getValidWorkDiv<Acc>(
      devAcc, threadsPerGrid, elementsPerThread, false,
      alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

  InitKernel<Idx> initKernel;

  alpaka::exec<Acc>(devQueue, accWorkDiv, initKernel,
                    alpakaex::writeAccess(accA));
  alpaka::exec<Acc>(devQueue, accWorkDiv, initKernel,
                    alpakaex::writeAccess(accB));

  if (size <= 10) {
    std::cout << "Init A:\n";

    alpaka::memcpy(devQueue, hostC, accA, extents);
    auto hostCAccessor = alpakaex::access(hostC);
    for (Idx y = 0; y < hostCAccessor.extents[0]; ++y) {
      for (Idx x = 0; x < hostCAccessor.extents[1]; ++x) {
        std::cout << hostCAccessor(y, x) << " ";
      }
      std::cout << "\n";
    }

    std::cout << "\n";
  }

  MatmulKernel<Idx> matmulKernel;
  alpaka::exec<Acc>(devQueue, accWorkDiv, matmulKernel,
                    alpakaex::readAccess(accA), alpakaex::readAccess(accB),
                    alpakaex::writeAccess(accC));

  if (size <= 10) {
    std::cout << "Result C:\n";

    alpaka::memcpy(devQueue, hostC, accC, extents);
    auto hostCAccessor = alpakaex::access(hostC);
    for (Idx y = 0; y < hostCAccessor.extents[0]; ++y) {
      for (Idx x = 0; x < hostCAccessor.extents[1]; ++x) {
        std::cout << hostCAccessor(y, x) << " ";
      }
      std::cout << "\n";
    }

    std::cout << "\n";
  }

  return 0;
}
