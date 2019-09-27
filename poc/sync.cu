#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <vector>
#include "cuda_runtime.h"
#include "nccl.h"
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>

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
    printf("  -n        number of GPUs and threads (default 2)\n");
	printf("  -i        number of iterations (default 1)\n");
	printf("  %s -n 4 -i 100\n", argv0);
}

int NRANKS;
int ITER_NUM;

void *worker(void *arg)
{
    int rank = *(int*)arg;
    int nranks = NRANKS;
    pthread_t tid = pthread_self();

    ncclUniqueId ncclId;
    NCCLCHECK(ncclGetUniqueId(&ncclId));

    ncclComm_t comm;

    int size = 32*1024*1024;

    //allocating and initializing device buffers
    int** sendbuff = (int**)malloc(sizeof(int*));
    int** recvbuff = (int**)malloc(sizeof(int*));
    int  *hostbuff = (int *)malloc(size * sizeof(int));
    cudaStream_t stream;


    printf("%ld: Malloc and preparing buffer on GPU: %d\n", tid, rank);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMalloc(sendbuff+0, size * sizeof(int)));
    CUDACHECK(cudaMalloc(recvbuff+0, size * sizeof(int)));
    CUDACHECK(cudaMemset(sendbuff[0], 1, size * sizeof(int)));
    CUDACHECK(cudaMemset(recvbuff[0], 0, size * sizeof(int)));
    printf("%ld:    sendbuf: %p\n%ld:    recvbuf: %p\n", tid, sendbuff[0], tid, recvbuff[0]);
    CUDACHECK(cudaStreamCreate(&stream));

    // Check memory contents
    printf("%ld: Buffer content on GPU: %d\n", tid, rank);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMemcpy(hostbuff, sendbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("%ld:    send: 0x%x\n", tid, hostbuff[0]);
    CUDACHECK(cudaMemcpy(hostbuff, recvbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("%ld:    recv: 0x%x\n", tid, hostbuff[0]);

    //initializing NCCL
    printf("%ld: Init comms on GPU: %d, rank: %d nranks: %d\n", tid, rank, rank, nranks);
    CUDACHECK(cudaSetDevice(rank));
    NCCLCHECK(ncclCommInitRank(&comm, nranks, ncclId, rank));

    for (int i=0; i<ITER_NUM; ++i) {

        printf("%ld: NCCL all reduce on GPU: %d, iter: %d\n", tid, rank, i+1);
        CUDACHECK(cudaSetDevice(rank));
        NCCLCHECK(ncclAllReduce((const void*)sendbuff[0], (void*)recvbuff[0], size, ncclInt, ncclSum,
            comm, stream));

        //synchronizing on CUDA streams to wait for completion of NCCL operation
        printf("%ld: Synchronize streams on GPU: %d\n", tid, rank);
        CUDACHECK(cudaSetDevice(rank));
        CUDACHECK(cudaStreamSynchronize(stream));

        // Check memory contents
        printf("%ld: Buffer content on GPU: %d\n", tid, rank);
        CUDACHECK(cudaSetDevice(rank));
        CUDACHECK(cudaMemcpy(hostbuff, sendbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
        printf("%ld:    send: 0x%x\n", tid, hostbuff[0]);
        CUDACHECK(cudaMemcpy(hostbuff, recvbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
        printf("%ld:    recv: 0x%x\n", tid, hostbuff[0]);

        printf("%ld: NCCL reduce on GPU: %d\n", tid, rank);
        CUDACHECK(cudaSetDevice(rank));
        NCCLCHECK(ncclReduce((const void*)sendbuff[0], (void*)recvbuff[0], size, ncclInt, ncclSum,
            0, comm, stream));

        //synchronizing on CUDA streams to wait for completion of NCCL operation
        printf("%ld: Synchronize streams on GPU: %d\n", tid, rank);
        CUDACHECK(cudaSetDevice(rank));
        CUDACHECK(cudaStreamSynchronize(stream));

        // Check memory contents
        printf("%ld: Buffer content on GPU: %d\n", tid, rank);
        CUDACHECK(cudaSetDevice(rank));
        CUDACHECK(cudaMemcpy(hostbuff, sendbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
        printf("%ld:    send: 0x%x\n", tid, hostbuff[0]);
        CUDACHECK(cudaMemcpy(hostbuff, recvbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
        printf("%ld:    recv: 0x%x\n", tid, hostbuff[0]);

        printf("%ld: NCCL BroadCase on GPU: %d, iter: %d\n", tid, rank, i+1);
        CUDACHECK(cudaSetDevice(rank));
        NCCLCHECK(ncclBroadcast((const void*)sendbuff[0], (void*)recvbuff[0], size, ncclInt, 0,
            comm, stream));

        //synchronizing on CUDA streams to wait for completion of NCCL operation
        printf("%ld: Synchronize streams on GPU: %d\n", tid, rank);
        CUDACHECK(cudaSetDevice(rank));
        CUDACHECK(cudaStreamSynchronize(stream));

        // Check memory contents
        printf("%ld: Buffer content on GPU: %d\n", tid, rank);
        CUDACHECK(cudaSetDevice(rank));
        CUDACHECK(cudaMemcpy(hostbuff, sendbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
        printf("%ld:    send: 0x%x\n", tid, hostbuff[0]);
        CUDACHECK(cudaMemcpy(hostbuff, recvbuff[0], size * sizeof(int), cudaMemcpyDeviceToHost));
        printf("%ld:    recv: 0x%x\n", tid, hostbuff[0]);
    }

    //free device buffers
    printf("%ld: Free memories on GPU: %d\n", tid, rank);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaFree(sendbuff[0]));
    CUDACHECK(cudaFree(recvbuff[0]));
    NCCLCHECK(ncclCommDestroy(comm));

    printf("%ld: Success \n", tid);

    free(sendbuff);
    free(recvbuff);
    free(hostbuff);
    pthread_exit(NULL);
}

int sync_test(const int num_gpu)
{
    NRANKS = num_gpu;
    int i, ranks[num_gpu];
    pthread_t threads[num_gpu];
    for (i=0; i<num_gpu; ++i){
        ranks[i] = i;
        printf("Creating thread #%d nranks: %d\n", i, NRANKS);
        pthread_create(&(threads[i]), NULL, worker, (void*)(ranks+i));
    }
    for (i=0; i<num_gpu; ++i)
        pthread_join(threads[i], NULL);
    printf("Main thread done.\n");
    return 0;
}

int main(int argc, char* argv[])
{
    int num_gpu = 2;
    ITER_NUM = 1;
    while (1) {
		int c;
		c = getopt(argc, argv, "n:i:");
		if (c == -1)
			break;

		switch (c) {
		case 'n':
			num_gpu = strtol(optarg, NULL, 0);
            if(num_gpu < 0) {
                usage(argv[0]);
                return 1;
            }
            break;
        case 'i':
            ITER_NUM = strtol(optarg, NULL, 0);
            if(ITER_NUM < 0) {
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
