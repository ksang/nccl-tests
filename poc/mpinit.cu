#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
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
    printf("Usage: %s       multi-process nccl comm init proof of concept.\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  -n        number of total ranks (default 2)\n");
    printf("  %s -n 2\n", argv0);
}

int mpinit_fork(int rank, int nranks)
{
    pid_t pid;
    pid = fork();
    if (pid == 0) {
        ncclUniqueId ncclId;
        NCCLCHECK(ncclGetUniqueId(&ncclId));
        ncclComm_t *comms = (ncclComm_t *)malloc(sizeof(ncclComm_t)*1);
        printf("Forked PID: %d rank: %d nranks: %d\n", getpid(), rank, nranks);
        NCCLCHECK(ncclGroupStart());
        printf("Init comms on rank: %d, device: %d\n", rank, rank);
        CUDACHECK(cudaSetDevice(rank));
        NCCLCHECK(ncclCommInitRank(comms, nranks, ncclId, rank));
        printf("ncclCommInitRank comm: %p\n", comms[0]);
        NCCLCHECK(ncclGroupEnd());
        printf("NCCL comm init complete, destorying...\n");
        ncclCommDestroy(comms[0]);
        printf("PID: %d exiting\n", getpid());
        exit(0);
    } else if (pid < 0) {
        fprintf(stderr, "Fork failed\n");
        return 1;
    }
    return 0;
}

int mpinit_test(int nranks)
{
    for (int i=0; i<nranks; ++i) {
        mpinit_fork(i, nranks);
    }
    printf("Main process waiting\n");
    sleep(30);
    return 0;
}

int main(int argc, char* argv[])
{
    int nranks = 2;
    while (1) {
		int c;
		c = getopt(argc, argv, "n:");
		if (c == -1)
			break;

		switch (c) {
		case 'n':
			nranks = strtol(optarg, NULL, 0);
            if(nranks <= 0) {
                usage(argv[0]);
                return 1;
            }
			break;
		default:
			usage(argv[0]);
			return -1;
		}
	}
    return mpinit_test(nranks);
}