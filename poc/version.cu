#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[])
{
    int version = 0;
    NCCLCHECK(ncclGetVersion(&version));
    printf("NCCL version: %d\n", version);

    ncclUniqueId ncclId;
    NCCLCHECK(ncclGetUniqueId(&ncclId));
    printf("NCCL UniqueId: \n");
    for ( int i=0; i < 128; ++i) {
        if (i > 0) printf(":");
        printf("%02X", ncclId.internal[i]);
    }
    printf("\n");
    return 0;
}
