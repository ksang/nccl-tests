#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <vector>
#include "cuda_runtime.h"
#include "nccl.h"

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

static void usage(const char *argv0)
{
	printf("Usage: %s       vgpu nccl proof of concept.\n", argv0);
	printf("\n");
	printf("Options:\n");
	printf("  -n        number of physical GPU (default 1)\n");
	printf("  -m        number of virtual GPU (default 2)\n");
	printf("  %s -n 2 -m 4\n", argv0);
}

int vgpu_to_pgpu(const int vgpu_id, const int num_pgpu)
{
    return vgpu_id % num_pgpu;
}

int vgpu_test(const int num_pgpu, const int num_vgpu)
{
    ncclComm_t *comms = (ncclComm_t *)malloc(sizeof(ncclComm_t)*num_vgpu);

    int nDev = num_vgpu;
    int size = 32*1024*1024;
    int *devs = (int *)malloc(sizeof(int)*num_vgpu);
    for (int i=0; i<num_vgpu; ++i) {
        devs[i] = i;
    }

    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(vgpu_to_pgpu(i, num_pgpu)));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s+i));
    }


    //initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));


     //calling NCCL communication API. Group API is required when using
     //multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
        NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
            comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());


    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }


    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }


    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);


    printf("Success \n");
    return 0;
}

int main(int argc, char* argv[])
{
    int num_pgpu = 1 ;
    int num_vgpu = 2 ;
    while (1) {
		int c;
		c = getopt(argc, argv, "n:m:");
		if (c == -1)
			break;

		switch (c) {
		case 'n':
			num_pgpu = strtol(optarg, NULL, 0);
            if(num_pgpu < 0) {
                usage(argv[0]);
                return 1;
            }
			break;
        case 'm':
        	num_vgpu = strtol(optarg, NULL, 0);
        	if (num_vgpu < 0 || num_vgpu < num_pgpu) {
        		usage(argv[0]);
        		return 1;
        	}
        	break;
		default:
			usage(argv[0]);
			return -1;
		}
	}
    return vgpu_test(num_pgpu, num_vgpu);
}
