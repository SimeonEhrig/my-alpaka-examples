#include <alpaka/alpaka.hpp>
#include <iostream>

struct HelloAlpakaKernel {

  template<typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc) const{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    using Vec1 = alpaka::vec::Vec<alpaka::dim::DimInt<1u>, Idx>;

    Vec const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    Vec const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

    Vec1 const linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(
								   globalThreadIdx,
								   globalThreadExtent);

    printf("[z:%u, y:%u, x;%u][linear:%u] Hello Alpaka\n",
	   static_cast<unsigned>(globalThreadIdx[0u]),
	   static_cast<unsigned>(globalThreadIdx[1u]),
	   static_cast<unsigned>(globalThreadIdx[2u]),
	   static_cast<unsigned>(linearizedGlobalThreadIdx[0u]));
  }

};

int main(){
  using Dim = alpaka::dim::DimInt<3>;
  using Idx = std::size_t;

  using Acc = alpaka::acc::AccCpuSerial<Dim, Idx>;
  using QueueProperty = alpaka::queue::Blocking;
  using Queue = alpaka::queue::Queue<Acc, QueueProperty>;
  using Dev = alpaka::dev::Dev<Acc>;
  using Pltf = alpaka::pltf::Pltf<Dev>;

  Dev const devAcc(alpaka::pltf::getDevByIdx<Pltf>(0u));

  Queue queue(devAcc);

  using Vec = alpaka::vec::Vec<Dim, Idx>;
  Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
  Vec const threadPerBlock(Vec::all(static_cast<Idx>(1)));
  Vec const blockPerGrid(
			 static_cast<Idx>(4),
			 static_cast<Idx>(8),
			 static_cast<Idx>(16));

  using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
  WorkDiv const workDiv(
			blockPerGrid,
			threadPerBlock,
			elementsPerThread);

  HelloAlpakaKernel helloAlpakaKernel;

  alpaka::kernel::exec<Acc>(queue, workDiv, helloAlpakaKernel);
  alpaka::wait::wait(queue);
  return 0;
}
