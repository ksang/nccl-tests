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
    //ncclUniqueId ncclId;
    //NCCLCHECK(ncclGetUniqueId(&ncclId));

    ncclComm_t *comms = (ncclComm_t *)malloc(sizeof(ncclComm_t)*num_vgpu);

    int nDev = num_vgpu;
    int size = 32*1024*1024;
    int *devs = (int *)malloc(sizeof(int)*num_vgpu);
    for (int i=0; i<num_vgpu; ++i) {
        devs[i] = i;
    }
    int *vdevs = (int *)malloc(sizeof(int)*num_vgpu);
    for (int i=0; i<num_vgpu; ++i) {
        vdevs[i] = vgpu_to_pgpu(i, num_pgpu);
    }

    //allocating and initializing device buffers
    int** sendbuff = (int**)malloc(nDev * sizeof(int*));
    int** recvbuff = (int**)malloc(nDev * sizeof(int*));
    int  *hostbuff = (int *)malloc(size * sizeof(int));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


    for (int i = 0; i < nDev; ++i) {
        printf("Malloc and preparing buffer on vGPU: %d - pGPU: %d\n",
            i, vgpu_to_pgpu(i, num_pgpu));
        CUDACHECK(cudaSetDevice(vgpu_to_pgpu(i, num_pgpu)));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(int)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(int)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(int)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(int)));
        printf("    sendbuf: %p\n    recvbuf: %p\n", sendbuff[i], recvbuff[i]);
        CUDACHECK(cudaStreamCreate(s+i));
    }

    // Check memory contents
    for (int i = 0; i < nDev; ++i) {
        printf("Buffer content on vGPU: %d - pGPU: %d\n",
            i, vgpu_to_pgpu(i, num_pgpu));
        CUDACHECK(cudaSetDevice(vgpu_to_pgpu(i, num_pgpu)));
        CUDACHECK(cudaMemcpy(hostbuff, sendbuff[i], size * sizeof(int), cudaMemcpyDeviceToHost));
        printf("    send: 0x%x\n", hostbuff[0]);
        CUDACHECK(cudaMemcpy(hostbuff, recvbuff[i], size * sizeof(int), cudaMemcpyDeviceToHost));
        printf("    recv: 0x%x\n", hostbuff[1]);
    }

    //initializing NCCL
    printf("NCCL comm init all, ndev: %d, devlist: { ", nDev);
    for (int i = 0; i < nDev; ++i) {
        printf("%d", vdevs[i]);
        if (i < nDev-1)
            printf(", ");
    }
    printf(" }\n");
    NCCLCHECK(ncclCommInitAll(comms, nDev, vdevs));
    //NCCLCHECK(ncclGroupStart());
    //for (int i = 0; i < nDev; ++i) {
    //    printf("Init comms on vGPU: %d - pGPU: %d, rank: %d\n",
    //        i, vgpu_to_pgpu(i, num_pgpu), i);
    //    CUDACHECK(cudaSetDevice(vgpu_to_pgpu(i, num_pgpu)));
    //    NCCLCHECK(ncclCommInitRank(comms+i, nDev, ncclId, i));
    //}
    //NCCLCHECK(ncclGroupEnd());

    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        printf("NCCL all reduce on vGPU: %d - pGPU: %d\n",
            i, vgpu_to_pgpu(i, num_pgpu));
        NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclInt, ncclSum,
            comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        printf("Synchronize streams on vGPU: %d - pGPU: %d\n",
            i, vgpu_to_pgpu(i, num_pgpu));
        CUDACHECK(cudaSetDevice(vgpu_to_pgpu(i, num_pgpu)));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // Check memory contents
    for (int i = 0; i < nDev; ++i) {
        printf("Buffer content on vGPU: %d - pGPU: %d\n",
            i, vgpu_to_pgpu(i, num_pgpu));
        CUDACHECK(cudaSetDevice(vgpu_to_pgpu(i, num_pgpu)));
        CUDACHECK(cudaMemcpy(hostbuff, sendbuff[i], size * sizeof(int), cudaMemcpyDeviceToHost));
        printf("    send: 0x%x\n", hostbuff[0]);
        CUDACHECK(cudaMemcpy(hostbuff, recvbuff[i], size * sizeof(int), cudaMemcpyDeviceToHost));
        printf("    recv: 0x%x\n", hostbuff[1]);
    }

    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        printf("Free memories on vGPU: %d - pGPU: %d\n",
            i, vgpu_to_pgpu(i, num_pgpu));
        CUDACHECK(cudaSetDevice(vgpu_to_pgpu(i, num_pgpu)));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    //finalizing NCCL
    for(int i = 0; i < nDev; ++i) {
        printf("NCCL comm destroy on vGPU: %d - pGPU: %d\n",
            i, vgpu_to_pgpu(i, num_pgpu));
        ncclCommDestroy(comms[i]);
    }

    printf("Success \n");

    free(devs);
    free(vdevs);
    free(sendbuff);
    free(recvbuff);
    free(hostbuff);
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
