/// My Query 24
/// select l_linenumber --> l_linenumber is the 4th attribute in lineitem table
/// from lineitem
/// group by l_linenumber
#include <unordered_set>
#include <cassert>
//#define COLLISION_PRINT

#include <list>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <ctime>
#include <limits.h>
#include <float.h>
#include "../dogqc/include/csv.h"
#include "../dogqc/include/util.h"
#include "../dogqc/include/mappedmalloc.h"
#include "../dogqc/include/util.cuh"
#include "../dogqc/include/hashing.cuh"
struct apayl2 {
    int att4_llinenum;
};

constexpr int SHARED_MEMORY_SIZE = 49152;  /// Total amount of shared memory per block:       49152 bytes
constexpr int SHARED_MEMORY_HT_SIZE = 1024;  /// In shared memory
constexpr int LINEITEM_SIZE = 6001215;       /// SF1
//constexpr int LINEITEM_SIZE = 59986052;      /// SF10, change the folder name to sf10
constexpr int GLOBAL_HT_SIZE = LINEITEM_SIZE * 2;  /// In global memory
//constexpr int GLOBAL_HT_SIZE = 8192;  /// In global memory

__device__ void sm_to_gm(agg_ht_sm<apayl2>* aht2, int SHARED_MEMORY_HT_SIZE, agg_ht<apayl2>* g_aht2) {
    /// Copy the shared memory hash table (pre-aggreagation) into the global hash table.
    {
        /// <-- START: first half of the kernel 2
        int att4_llinenum;
        int tid_aggregation2 = 0;
        unsigned loopVar = threadIdx.x;  ///
        unsigned step = blockDim.x;  ///
        unsigned flushPipeline = 0;
        int active = 0;
        while(!(flushPipeline)) {
            tid_aggregation2 = loopVar;
            active = (loopVar < SHARED_MEMORY_HT_SIZE);  ///
            // flush pipeline if no new elements
            flushPipeline = !(__ballot_sync(ALL_LANES,active));
            if(active) {
            }
            // -------- scan aggregation ht (opId: 2) --------
            if(active) {
                active &= ((aht2[tid_aggregation2].lock.lock == OnceLock::LOCK_DONE));
            }
            if(active) {
                apayl2 payl = aht2[tid_aggregation2].payload;
                att4_llinenum = payl.att4_llinenum;
            }
            if(active) {
            }
            /// <-- END: first half of the kernel 2


            /// <-- START: second half of the kernel 1
            /// Insert to global hash table.
            int bucket = 0;
            if(active) {
                uint64_t hash2 = 0;
                hash2 = 0;
                if(active) {
                    hash2 = hash ( (hash2 + ((uint64_t)att4_llinenum)));
                }
                apayl2 payl;
                payl.att4_llinenum = att4_llinenum;
                int bucketFound = 0;
                int numLookups = 0;
                while(!(bucketFound)) {
                    bucket = hashAggregateGetBucket ( g_aht2, GLOBAL_HT_SIZE, hash2, numLookups, &(payl));  ////
                    apayl2 probepayl = g_aht2[bucket].payload;  ////
                    bucketFound = 1;
                    bucketFound &= ((payl.att4_llinenum == probepayl.att4_llinenum));
                }
            }
            if(active) {
            }
            /// <-- END: second half of the kernel 1
            loopVar += step;
        }
    }
}

__global__ void krnl_lineitem1(
    int* iatt4_llinenum, int* nout_result, int* oatt4_llinenum, agg_ht<apayl2>* g_aht2) {  ///

    /// local block memory cache : ONLY FOR A BLOCK'S THREADS!!!
    __shared__ agg_ht_sm<apayl2> aht2[SHARED_MEMORY_HT_SIZE];  ///
    volatile __shared__ int HT_FULL_FLAG; HT_FULL_FLAG = 0;  ////
#ifdef COLLISION_PRINT
    __shared__ int num_collision; num_collision = 0;
#endif
    const int shared_memory_usage = sizeof(aht2);
    assert(shared_memory_usage <= SHARED_MEMORY_SIZE);  /// Check stuff fits into shared memory in a SM.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        /// Allow only one print here.
        printf("Shared memory usage: %d / %d bytes.\n", shared_memory_usage, SHARED_MEMORY_SIZE);
    }

    initSMAggHT(aht2,SHARED_MEMORY_HT_SIZE);
    __syncthreads();

    {
        /// The first old kenrel
        int att4_llinenum;

        int tid_lineitem1 = 0;
        unsigned loopVar__ = ((blockIdx.x * blockDim.x) + threadIdx.x);
        unsigned step = (blockDim.x * gridDim.x);
        unsigned flushPipeline__ = 0;
        int active = 0;
        while(!(flushPipeline__)) {
            tid_lineitem1 = loopVar__;
            active = (loopVar__ < LINEITEM_SIZE);
            // flush pipeline if no new elements
            flushPipeline__ = !(__ballot_sync(ALL_LANES,active));
            if(active) {
                att4_llinenum = iatt4_llinenum[tid_lineitem1];
            }
            // -------- aggregation (opId: 2) --------
            int bucket = 0;
            if(active) {
                uint64_t hash2 = 0;
                hash2 = 0;
                if(active) {
                    hash2 = hash ( (hash2 + ((uint64_t)att4_llinenum)));
                }
                apayl2 payl;
                payl.att4_llinenum = att4_llinenum;
                int bucketFound = 0;
                int numLookups = 0;
                while(!(bucketFound)) {   ////
                    bucket = hashAggregateGetBucket ( aht2, SHARED_MEMORY_HT_SIZE, hash2, numLookups, &(payl));  ///
                    if (bucket != -1) {  ////
                        apayl2 probepayl = aht2[bucket].payload;
                        bucketFound = 1;
                        bucketFound &= ((payl.att4_llinenum == probepayl.att4_llinenum));
                    }
                    else {
                        assert(bucketFound == 0);  ////
                        loopVar__ -= step;
                        atomicAdd((int *)&HT_FULL_FLAG, 1);  ////
                        break;  ////
                    }
                }
#ifdef COLLISION_PRINT
                atomicAdd(&num_collision, numLookups - 1);
#endif
            }
            if(active && bucket != -1) {  ////
            }

            /// Implication and Disjunction: P->Q <=>  ^P OR Q
            /// bucket==-1 -> HT_FULL_FLAG!=0
            assert(bucket != -1 || HT_FULL_FLAG != 0);

            //// insert the tuple into the global memory hash table.
            __syncthreads();  ////
            if (HT_FULL_FLAG != 0) {
                sm_to_gm(aht2, SHARED_MEMORY_HT_SIZE, g_aht2);
                __threadfence_block(); /// Ensure the ordering:
                initSMAggHT(aht2,SHARED_MEMORY_HT_SIZE);
                atomicExch((int*)&HT_FULL_FLAG, 0);
                __syncthreads();  ////
            }
            ////
            loopVar__ += step;
        }
    }

    __syncthreads();  ///
#ifdef COLLISION_PRINT
    if (threadIdx.x == 0) {
        /// Allow only one print per block here.
        printf("In Block %d: num_collision: %d\n", blockIdx.x, num_collision);
    }
#endif

#ifdef HT_CHECKER
    if (threadIdx.x == 0) {
        if (HT_FULL_FLAG != 0) {
            printf("FUll.\n");
        } else {
            printf("Not FULL.\n");
        }
    }
#endif
    sm_to_gm(aht2, SHARED_MEMORY_HT_SIZE, g_aht2);
}

__global__ void krnl_reduce(
        agg_ht<apayl2>* aht2, int* n_out_result, int* oatt4_llinenum) {  ///
    int att4_llinenum;  ///

    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));
    int tid_aggregation2 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation2 = loopVar;
        active = (loopVar < GLOBAL_HT_SIZE);  ///
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 2) --------
        if(active) {
            active &= ((aht2[tid_aggregation2].lock.lock == OnceLock::LOCK_DONE));
        }
        if(active) {
            apayl2 payl = aht2[tid_aggregation2].payload;
            att4_llinenum = payl.att4_llinenum;
        }
        if(active) {
        }
        // -------- projection (no code) (opId: 3) --------
        // -------- materialize (opId: 4) --------
        int wp;
        int writeMask;
        int numProj;
        writeMask = __ballot_sync(ALL_LANES,active);
        numProj = __popc(writeMask);
        if((warplane == 0)) {
            wp = atomicAdd(n_out_result, numProj);  ///
        }
        wp = __shfl_sync(ALL_LANES,wp,0);
        wp = (wp + __popc((writeMask & prefixlanes)));
        if(active) {
            oatt4_llinenum[wp] = att4_llinenum;
        }
        loopVar += step;
    }
}

int main() {
    int* iatt4_llinenum;
    iatt4_llinenum = ( int*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_linenumber" );

    int nout_result;
    std::vector < int > oatt4_llinenum(LINEITEM_SIZE);

    // wake up gpu
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in wake up gpu! " << cudaGetErrorString( err ) << std::endl;
            ERROR("wake up gpu")
        }
    }

    int* d_iatt4_llinenum;
    cudaMalloc((void**) &d_iatt4_llinenum, LINEITEM_SIZE* sizeof(int) );
    int* d_nout_result;
    cudaMalloc((void**) &d_nout_result, 1* sizeof(int) );
    int* d_oatt4_llinenum;
    cudaMalloc((void**) &d_oatt4_llinenum, LINEITEM_SIZE* sizeof(int) );
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda malloc! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda malloc")
        }
    }


    // show memory usage of GPU
    {   size_t free_byte ;
        size_t total_byte ;
        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
        if ( cudaSuccess != cuda_status ) {
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
            exit(1);
        }
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        fprintf(stderr, "Memory %.1f / %.1f GB\n",
                used_db/(1024*1024*1024), total_db/(1024*1024*1024) );
        fflush(stdout);
    }

    agg_ht<apayl2>* d_aht2;
    cudaMalloc((void**) &d_aht2, GLOBAL_HT_SIZE* sizeof(agg_ht<apayl2>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_aht2, GLOBAL_HT_SIZE);
    }
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_nout_result, 0, 1);
    }
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda mallocHT! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda mallocHT")
        }
    }


    // show memory usage of GPU
    {   size_t free_byte ;
        size_t total_byte ;
        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
        if ( cudaSuccess != cuda_status ) {
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
            exit(1);
        }
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        fprintf(stderr, "Memory %.1f / %.1f GB\n",
                used_db/(1024*1024*1024), total_db/(1024*1024*1024) );
        fflush(stdout);
    }

    cudaMemcpy( d_iatt4_llinenum, iatt4_llinenum, LINEITEM_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy in! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy in")
        }
    }

    std::clock_t start_totalKernelTime0 = std::clock();
    std::clock_t start_krnl_lineitem11 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem1<<<gridsize, blocksize>>>(d_iatt4_llinenum, d_nout_result, d_oatt4_llinenum, d_aht2);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem11 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem1! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem1")
        }
    }

    std::clock_t start_krnl_reduce2 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_reduce<<<gridsize, blocksize>>>(d_aht2, d_nout_result, d_oatt4_llinenum);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_reduce2 = std::clock();
    std::clock_t stop_totalKernelTime0 = std::clock();

    cudaMemcpy( &nout_result, d_nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt4_llinenum.data(), d_oatt4_llinenum, LINEITEM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy out! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy out")
        }
    }

    cudaFree( d_iatt4_llinenum);
    cudaFree( d_nout_result);
    cudaFree( d_oatt4_llinenum);
    cudaFree( d_aht2);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda free! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda free")
        }
    }

    std::clock_t start_finish3 = std::clock();
    printf("\nResult: %i tuples\n", nout_result);
    if((nout_result > LINEITEM_SIZE)) {
        ERROR("Index out of range. Output size larger than allocated with expected result number.")
    }
    for ( int pv = 0; ((pv < 10) && (pv < nout_result)); pv += 1) {
        printf("l_linenumber: ");
        printf("%8i", oatt4_llinenum[pv]);
        printf("  ");
        printf("\n");
    }
    if((nout_result > 10)) {
        printf("[...]\n");
    }
    printf("\n");
    std::clock_t stop_finish3 = std::clock();


    printf("<timing>\n");
    printf ( "%32s: %6.1f ms\n", "finish", (stop_finish3 - start_finish3) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem1", (stop_krnl_lineitem11 - start_krnl_lineitem11) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_reduce", (stop_krnl_reduce2 - start_krnl_reduce2) / (double) (CLOCKS_PER_SEC / 1000) );  ///
    printf ( "%32s: %6.1f ms\n", "totalKernelTime", (stop_totalKernelTime0 - start_totalKernelTime0) / (double) (CLOCKS_PER_SEC / 1000) );
    printf("</timing>\n");
}
