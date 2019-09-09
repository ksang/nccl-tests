#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <vector>
#include "cuda_runtime.h"
#include "nccl.h"
#include <pthread.h>

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
	printf("Usage: %s       sync nccl proof of concept.\n", argv0);
	printf("                create threads per gpu and exec nccl calls synchronously.\n");
	printf("Options:\n");
	printf("  -n        number of GPUs and threads(default 2)\n");
	printf("  %s -n 4\n", argv0);
}

struct  parameter
{
    int rank;
    int nranks;
};

void worker(void *arg)
{
    struct parameter *pstru;
    pstru = ( struct parameter *) arg;
    int rank = *pstru.rank;
    int nranks = *pstru.nranks;
    tid = pthread_self();

    ncclUniqueId ncclId;
    NCCLCHECK(ncclGetUniqueId(&ncclId));

    ncclComm_t *comm = (ncclComm_t *)malloc(sizeof(ncclComm_t));

    int size = 32*1024*1024;

    //allocating and initializing device buffers
    int** sendbuff = (int**)malloc(sizeof(int*));
    int** recvbuff = (int**)malloc(sizeof(int*));
    int  *hostbuff = (int *)malloc(sizeof(int));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t));


    printf("%d: Malloc and preparing buffer on GPU: %d\n", tid, rank);
    CUDACHECK(cudaMalloc(sendbuff, size * sizeof(int)));
    CUDACHECK(cudaMalloc(recvbuff, size * sizeof(int)));
    CUDACHECK(cudaMemset(sendbuff[0], 1, size * sizeof(int)));
    CUDACHECK(cudaMemset(recvbuff[0], 0, size * sizeof(int)));
    printf("%d:    sendbuf: %p\n    recvbuf: %p\n", tid, sendbuff[0], recvbuff[0]);
    CUDACHECK(cudaStreamCreate(s));

    // Check memory contents
    printf("%d: Buffer content on GPU: %d\n", tid, rank);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMemcpy(hostbuff, sendbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d:    send: 0x%x\n", tid, hostbuff[0]);
    CUDACHECK(cudaMemcpy(hostbuff, recvbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d:    recv: 0x%x\n", tid, hostbuff[0]);

    //initializing NCCL
    printf("%d: Init comms on vGPU: %d - GPU: %d, rank: %d\n", tid, rank, rank);
    CUDACHECK(cudaSetDevice(rank));
    NCCLCHECK(ncclCommInitRank(comms, nranks, ncclId, rank));

    //calling NCCL communication API. Group API is required when using
    printf("%d: NCCL all reduce on GPU: %d\n", tid, rank);
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[0], (void*)recvbuff[0], size, ncclInt, ncclSum,
        comms, s[0]));

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    printf("%d: Synchronize streams on GPU: %d\n", tid, rank);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaStreamSynchronize(s[0]));

    // Check memory contents
    printf("%d: Buffer content on GPU: %d\n", tid, rank);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMemcpy(hostbuff, sendbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d:    send: 0x%x\n", tid, hostbuff[0]);
    CUDACHECK(cudaMemcpy(hostbuff, recvbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d:    recv: 0x%x\n", tid, hostbuff[0]);

    //free device buffers
    printf("%d: Free memories on GPU: %d\n", tid, rank);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaFree(sendbuff[0]));
    CUDACHECK(cudaFree(recvbuff[0]));
    NCCLCHECK(ncclCommDestroy(comms[0]));

    printf("%d: Success \n", tid);

    free(devs);
    free(vdevs);
    free(sendbuff);
    free(recvbuff);
    free(hostbuff);
    return 0;
}

int sync_test(const int num_gpu)
{
    pthread_t threads[num_gpu];
    for (int i=0; i<num_gpu; ++i){
        struct paramager arg;
        arg.rank = i;
        arg.nranks = num_gpu;
        pthread_create(&threads[i], NULL, worker, &arg);
    }
    for (int i=0; i<num_gpu; ++i)
        pthread_join(thread[i], NULL);
    printf("Main thread done.\n");
    return 0;
}

int main(int argc, char* argv[])
{
    int num_gpu = 2;
    while (1) {
		int c;
		c = getopt(argc, argv, "n:");
		if (c == -1)
			break;

		switch (c) {
		case 'n':
			num_gpu = strtol(optarg, NULL, 0);
            if(num_pgpu < 0) {
                usage(argv[0]);
                return 1;
            }
			break;
		default:
			usage(argv[0]);
			return -1;
		}
	}
    return sync_test(num_gpu);
}
