/* Host-only driver launcher for Nsight. No GPU kernel is implemented here;
 * the frozen Triton cubin supplies every GPU instruction. */
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { CUresult e=(call); if(e!=CUDA_SUCCESS) { \
  const char *message=NULL; cuGetErrorString(e,&message); \
  fprintf(stderr,"%s: %d %s\n",#call,e,message?message:""); exit(2); } } while(0)

int main(int argc,char **argv) {
  if(argc!=4) { fprintf(stderr,"usage: target cubin baseline|baseline_padded|shared profile0or1\n"); return 2; }
  puts("HOST_TARGET_ENTERED"); fflush(stdout);
  const int shared=!strcmp(argv[2],"shared");
  const unsigned dynamic=79232+(!strcmp(argv[2],"baseline_padded")?4352:0);
  CHECK(cuInit(0));
  CUcontext context; CHECK(cuCtxCreate(&context,0,0));
  puts("CONTEXT_CREATED"); fflush(stdout);
  CUmodule module; CUfunction function;
  CHECK(cuModuleLoad(&module,argv[1]));
  CHECK(cuModuleGetFunction(&function,module,shared?"_fragment_partition":"_normalized_partition"));
  CHECK(cuFuncSetCacheConfig(function,CU_FUNC_CACHE_PREFER_SHARED));
  CHECK(cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,(int)dynamic));
  size_t sizes[]={8192,4718592,4718592,36,4,589824,2304};
  CUdeviceptr addresses[8]; void *params[8];
  for(int i=0;i<7;i++) {
    CHECK(cuMemAlloc(&addresses[i],sizes[i]));
    CHECK(cuMemsetD8(addresses[i],0,sizes[i]));
    params[i]=&addresses[i];
  }
  int table[]={2,8,6,0,7,4,3,5,1},length=2064;
  CHECK(cuMemcpyHtoD(addresses[3],table,sizeof(table)));
  CHECK(cuMemcpyHtoD(addresses[4],&length,sizeof(length)));
  addresses[7]=addresses[0]; params[7]=&addresses[7];
  for(int i=0;i<3;i++) CHECK(cuLaunchKernel(function,1,4,36,128,1,1,dynamic,0,params,NULL));
  CHECK(cuCtxSynchronize());
  puts("WARMUP_COMPLETE"); fflush(stdout);
  void *lib=dlopen("libcuda.so.1",RTLD_NOW);
  CUresult (*start)(void)=(CUresult (*)(void))dlsym(lib,"cuProfilerStart");
  CUresult (*stop)(void)=(CUresult (*)(void))dlsym(lib,"cuProfilerStop");
  if(!start || !stop) return 3;
  if(atoi(argv[3])) CHECK(start());
  CHECK(cuLaunchKernel(function,1,4,36,128,1,1,dynamic,0,params,NULL));
  CHECK(cuCtxSynchronize());
  if(atoi(argv[3])) CHECK(stop());
  float output[16]; CHECK(cuMemcpyDtoH(output,addresses[5],sizeof(output)));
  for(int i=0;i<16;i++) if(output[i]!=0) return 4;
  int blocks=0,regs=0,statics=0;
  CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,function,128,dynamic));
  CHECK(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,function));
  CHECK(cuFuncGetAttribute(&statics,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,function));
  printf("{\"variant\":\"%s\",\"max_blocks\":%d,\"regs\":%d,\"static\":%d,\"dynamic\":%u,\"zero_partial_check\":true}\n",argv[2],blocks,regs,statics,dynamic);
  CHECK(cuCtxDestroy(context));
  return 0;
}
