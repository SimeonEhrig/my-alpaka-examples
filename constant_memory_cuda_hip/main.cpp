#include <alpaka/alpaka.hpp>
#include <iostream>

constexpr int num_const_elem = 5;

ALPAKA_STATIC_ACC_MEM_CONSTANT float constant_maxval[num_const_elem];

struct CopyKernel
{
  template<typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, float * data) const{
    for(int i = 0; i < num_const_elem; i++){
      data[i] = constant_maxval[i];
    }
  }
};

struct ZeroKernel
{
  template<typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, float * data) const{
    for(int i = 0; i < num_const_elem; i++){
      data[i] = 0.f;
    }
  }
};


int main(){
#if( ALPAKA_ACC_GPU_HIP_ENABLED == 1 || ALPAKA_ACC_GPU_CUDA_ENABLED == 1 )
  // **************************
  // setup devices
  // **************************
  using Dim = alpaka::DimInt<1u>;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  using DevAcc = alpaka::AccGpuCudaRt<Dim, std::size_t>;
#else
  using DevAcc = alpaka::AccGpuHipRt<Dim, std::size_t>;
#endif

  using DevQueue = alpaka::Queue<DevAcc, alpaka::Blocking>;
  auto const devAcc = alpaka::getDevByIdx<DevAcc>(0u);
  DevQueue devQueue(devAcc);

  using DevHost = alpaka::AccCpuSerial<Dim, std::size_t>;
  auto const devHost = alpaka::getDevByIdx<DevHost>(0u);

  alpaka::WorkDivMembers<Dim, std::size_t> const workdiv{
    static_cast<std::size_t>(1),
    static_cast<std::size_t>(1),
    static_cast<std::size_t>(1)};

  // **************************
  // setup memory
  // **************************
  using Vec = alpaka::Vec<Dim, std::size_t>;
  const Vec extents(Vec::all(static_cast<std::size_t>(num_const_elem)));

  using BufHost = alpaka::Buf<DevHost, float, Dim, std::size_t>;
  BufHost hostBufferInput(alpaka::allocBuf<float, std::size_t>(devHost, extents));
  float * hostBufferInputPtr = alpaka::getPtrNative(hostBufferInput);

  BufHost hostBufferOutput(alpaka::allocBuf<float, std::size_t>(devHost, extents));
  float * hostBufferOutputPtr = alpaka::getPtrNative(hostBufferOutput);

  using BufDevice = alpaka::Buf<DevAcc, float, Dim, std::size_t>;
  BufDevice deviceBuffer(alpaka::allocBuf<float, std::size_t>(devAcc, extents));
  float * deviceBufferPtr = alpaka::getPtrNative(deviceBuffer);

  // **************************
  // setup constant memory
  // **************************

  for(int i = 0; i < num_const_elem; ++i){
    hostBufferInputPtr[i] = static_cast<float>(i*2);
  }

  auto viewConstantMemory = alpaka::createStaticDevMemView(constant_maxval, devAcc, extents);
  alpaka::memcpy(devQueue, viewConstantMemory, hostBufferInput, extents);

  // **************************
  // execute kernel
  // **************************

  CopyKernel copyKernel;
  ZeroKernel zeroKernel;

  alpaka::exec<DevAcc>(devQueue, workdiv, zeroKernel, deviceBufferPtr);
  alpaka::exec<DevAcc>(devQueue, workdiv, copyKernel, deviceBufferPtr);
  alpaka::memcpy(devQueue, hostBufferOutput, deviceBuffer, extents);

  // **************************
  // print result
  // **************************

  std::cout << "         Result: ";
  for(int i = 0; i < num_const_elem; ++i){
    std::cout << hostBufferOutputPtr[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Expected Result: ";
  for(int i = 0; i < num_const_elem; ++i){
    std::cout << i*2 << " ";
  }
  std::cout << std::endl;

  return 0;
#else
  std::cout << "Only CUDA or HIP backend supported" << std::endl;
  return 1;
#endif
}
