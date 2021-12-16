/// select l_suppkey, count(*)
/// from lineitem
/// group by l_suppkey
/// order by l_suppkey
#include <map>
#include <cassert>
//#define COLLISION_PRINT
//#define HT_CHECKER

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
    int att4_lsuppkey;
};

constexpr int SHARED_MEMORY_SIZE = 49152;  /// Total amount of shared memory per block:       49152 bytes
const int HT_SIZE = 1024;  /// In shared memory
constexpr int GLOBAL_HT_SIZE = 12002430;  /// In global memory
//constexpr int GLOBAL_HT_SIZE = 65536;  /// In global memory

__global__ void krnl_lineitem1(
    int* iatt4_lsuppkey, int* nout_result, int* oatt4_lsuppkey, int* oatt1_countlsu, agg_ht<apayl2>* g_aht2, int* g_agg1) {  ///

    /// local block memory cache : ONLY FOR A BLOCK'S THREADS!!!
    __shared__ agg_ht_sm<apayl2> aht2[HT_SIZE];  ///
    __shared__ int agg1[HT_SIZE];  ///
    volatile __shared__ int migration_to_global_memory; migration_to_global_memory = 0;  ////
#ifdef COLLISION_PRINT
    __shared__ int num_collision; num_collision = 0;
#endif
    const int shared_memory_usage = sizeof(aht2) + sizeof(agg1);
    assert(shared_memory_usage <= SHARED_MEMORY_SIZE);  /// Check stuff fits into shared memory in a SM.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        /// Allow only one print here.
        printf("Shared memory usage: %d / %d bytes.\n", shared_memory_usage, SHARED_MEMORY_SIZE);
    }

    {
        /// Init hash table in shared memory.
        int ht_index;
        unsigned loopVar = threadIdx.x;  ///
        unsigned step = blockDim.x;  ///
        while(loopVar < HT_SIZE) {
            ht_index = loopVar;
            aht2[ht_index].lock.init();
            aht2[ht_index].hash = HASH_EMPTY;
            loopVar += step;
        }
    }

    {
        /// Init array in shared memory.
        int index;
        unsigned loopVar = threadIdx.x;  ///
        unsigned step = blockDim.x;  ///
        while(loopVar < HT_SIZE) {
            index = loopVar;
            agg1[index] = 0;
            loopVar += step;
        }
    }

    __syncthreads();

    {
        /// The first old kenrel
        int att4_lsuppkey;

        int tid_lineitem1 = 0;
        unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
        unsigned step = (blockDim.x * gridDim.x);
        unsigned flushPipeline = 0;
        int active = 0;
        while(!(flushPipeline)) {
            tid_lineitem1 = loopVar;
            active = (loopVar < 6001215);
            // flush pipeline if no new elements
            flushPipeline = !(__ballot_sync(ALL_LANES,active));
            if(active) {
                att4_lsuppkey = iatt4_lsuppkey[tid_lineitem1];
            }
            // -------- aggregation (opId: 2) --------
            int bucket = 0;
            if(active) {
                uint64_t hash2 = 0;
                hash2 = 0;
                if(active) {
                    hash2 = hash ( (hash2 + ((uint64_t)att4_lsuppkey)));
                }
                apayl2 payl;
                payl.att4_lsuppkey = att4_lsuppkey;
                int bucketFound = 0;
                int numLookups = 0;
                while(!(bucketFound)) {   ////
                    bucket = hashAggregateGetBucket ( aht2, HT_SIZE, hash2, numLookups, &(payl));  ///
                    if (bucket != -1) {  ////
                        apayl2 probepayl = aht2[bucket].payload;
                        bucketFound = 1;
                        bucketFound &= ((payl.att4_lsuppkey == probepayl.att4_lsuppkey));
                    } else {  ////
                        assert(bucketFound == 0);  ////
//                        loopVar -= step; TODO use this in other branch: copy ht first, then ...
                        atomicAdd((int *)&migration_to_global_memory, 1);  ////
                        bucket = -1;
                        break;  ////
                    }  ////
                }
#ifdef COLLISION_PRINT
                atomicAdd(&num_collision, numLookups - 1);
#endif
            }
            if(active && bucket != -1) {  ////
                atomicAdd(&(agg1[bucket]), ((int)1));
            }

            /// Implication and Disjunction: P->Q <=>  ^P OR Q
            /// bucket==-1 -> migration_to_global_memory!=0
            assert(bucket != -1 || migration_to_global_memory!=0);

            //// insert the tuple into the global memory hash table.
//            if (bucket == -1 || migration_to_global_memory != 0) {  ////
            if (bucket == -1) {  ////
//            if (migration_to_global_memory != 0) {  ////
                /// <-- START: second half of the kernel 1
                if(active) {
                    uint64_t hash2 = 0;
                    hash2 = 0;
                    if(active) {
                        hash2 = hash ( (hash2 + ((uint64_t)att4_lsuppkey)));
                    }
                    apayl2 payl;
                    payl.att4_lsuppkey = att4_lsuppkey;
                    int bucketFound = 0;
                    int numLookups = 0;
                    while(!(bucketFound)) {
                        bucket = hashAggregateGetBucket ( g_aht2, GLOBAL_HT_SIZE, hash2, numLookups, &(payl));  ////
                        apayl2 probepayl = g_aht2[bucket].payload;  ////
                        bucketFound = 1;
                        bucketFound &= ((payl.att4_lsuppkey == probepayl.att4_lsuppkey));
                    }
                }
                if(active) {
                    atomicAdd(&(g_agg1[bucket]), ((int)1));  ////
                }
                /// <-- END: second half of the kernel 1
            }
            ////
            loopVar += step;
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
        if (migration_to_global_memory != 0) {
            printf("FUll.\n");
        } else {
            printf("Not FULL.\n");
        }
    }
#endif

    /// Copy the shared memory hash table (pre-aggreagation) into the global hash table.
    {
        /// <-- START: first half of the kernel 2
        int att4_lsuppkey;
        int att1_countlsu;
        int tid_aggregation2 = 0;
        unsigned loopVar = threadIdx.x;  ///
        unsigned step = blockDim.x;  ///
        unsigned flushPipeline = 0;
        int active = 0;
        while(!(flushPipeline)) {
            tid_aggregation2 = loopVar;
            active = (loopVar < HT_SIZE);  ///
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
                att4_lsuppkey = payl.att4_lsuppkey;
            }
            if(active) {
                att1_countlsu = agg1[tid_aggregation2];
            }
            /// <-- END: first half of the kernel 2

            /// <-- START: second half of the kernel 1
            /// Insert to global hash table.
            int bucket = 0;
            if(active) {
                uint64_t hash2 = 0;
                hash2 = 0;
                if(active) {
                    hash2 = hash ( (hash2 + ((uint64_t)att4_lsuppkey)));
                }
                apayl2 payl;
                payl.att4_lsuppkey = att4_lsuppkey;
                int bucketFound = 0;
                int numLookups = 0;
                while(!(bucketFound)) {
                    bucket = hashAggregateGetBucket ( g_aht2, GLOBAL_HT_SIZE, hash2, numLookups, &(payl));  ////
                    apayl2 probepayl = g_aht2[bucket].payload;  ////
                    bucketFound = 1;
                    bucketFound &= ((payl.att4_lsuppkey == probepayl.att4_lsuppkey));
                }
            }
            if(active) {
                atomicAdd(&(g_agg1[bucket]), ((int)att1_countlsu));  ////
            }
            /// <-- END: second half of the kernel 1
            loopVar += step;
        }
    }
}

__global__ void krnl_aggregation2(
    agg_ht<apayl2>* aht2, int* agg1, int* nout_result, int* oatt4_lsuppkey, int* oatt1_countlsu) {
    int att4_lsuppkey;
    int att1_countlsu;
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
            att4_lsuppkey = payl.att4_lsuppkey;
        }
        if(active) {
            att1_countlsu = agg1[tid_aggregation2];
        }
        // -------- projection (no code) (opId: 3) --------
        // -------- materialize (opId: 4) --------
        int wp;
        int writeMask;
        int numProj;
        writeMask = __ballot_sync(ALL_LANES,active);
        numProj = __popc(writeMask);
        if((warplane == 0)) {
            wp = atomicAdd(nout_result, numProj);
        }
        wp = __shfl_sync(ALL_LANES,wp,0);
        wp = (wp + __popc((writeMask & prefixlanes)));
        if(active) {
            oatt4_lsuppkey[wp] = att4_lsuppkey;
            oatt1_countlsu[wp] = att1_countlsu;
        }
        loopVar += step;
    }

}

int main() {
    int* iatt4_lsuppkey;
    iatt4_lsuppkey = ( int*) map_memory_file ( "mmdb/lineitem_l_suppkey" );

    int nout_result;
    std::vector < int > oatt4_lsuppkey(6001215);
    std::vector < int > oatt1_countlsu(6001215);

    // wake up gpu
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in wake up gpu! " << cudaGetErrorString( err ) << std::endl;
            ERROR("wake up gpu")
        }
    }

    int* d_iatt4_lsuppkey;
    cudaMalloc((void**) &d_iatt4_lsuppkey, 6001215* sizeof(int) );
    int* d_nout_result;
    cudaMalloc((void**) &d_nout_result, 1* sizeof(int) );
    int* d_oatt4_lsuppkey;
    cudaMalloc((void**) &d_oatt4_lsuppkey, 6001215* sizeof(int) );
    int* d_oatt1_countlsu;
    cudaMalloc((void**) &d_oatt1_countlsu, 6001215* sizeof(int) );
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
    int* d_agg1;
    cudaMalloc((void**) &d_agg1, GLOBAL_HT_SIZE* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg1, 0, GLOBAL_HT_SIZE);
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

    cudaMemcpy( d_iatt4_lsuppkey, iatt4_lsuppkey, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
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
        krnl_lineitem1<<<gridsize, blocksize>>>(d_iatt4_lsuppkey, d_nout_result, d_oatt4_lsuppkey, d_oatt1_countlsu, d_aht2, d_agg1);
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

    std::clock_t start_krnl_aggregation22 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation2<<<gridsize, blocksize>>>(d_aht2, d_agg1, d_nout_result, d_oatt4_lsuppkey, d_oatt1_countlsu);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation22 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation2! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation2")
        }
    }
    std::clock_t stop_totalKernelTime0 = std::clock();

    cudaMemcpy( &nout_result, d_nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt4_lsuppkey.data(), d_oatt4_lsuppkey, 6001215 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt1_countlsu.data(), d_oatt1_countlsu, 6001215 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy out! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy out")
        }
    }

    cudaFree( d_iatt4_lsuppkey);
    cudaFree( d_aht2);
    cudaFree( d_agg1);
    cudaFree( d_nout_result);
    cudaFree( d_oatt4_lsuppkey);
    cudaFree( d_oatt1_countlsu);
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
    if((nout_result > 6001215)) {
        ERROR("Index out of range. Output size larger than allocated with expected result number.")
    }
    for ( int pv = 0; ((pv < 10) && (pv < nout_result)); pv += 1) {
        printf("l_suppkey: ");
        printf("%8i", oatt4_lsuppkey[pv]);
        printf("  ");
        printf("count_l_suppkey: ");
        printf("%8i", oatt1_countlsu[pv]);
        printf("  ");
        printf("\n");
    }
    if((nout_result > 10)) {  ///
        printf("[...]\n");
    }
    printf("\n");

    /// 40 sorted output
    std::map<int, int> ht;
    for ( int pv = 0; pv < nout_result; pv += 1) {
        ht.emplace(oatt4_lsuppkey[pv], oatt1_countlsu[pv]);
    }
    auto it = ht.begin();
    printf("\nSorted Result: %ld tuples\n", ht.size());
    for ( int pv = 0; ((pv < 25) && (pv < ht.size())); pv += 1) {
        printf("l_orderkey: ");
        printf("%8i", it->first);
        printf("  ");
        printf("count_l_orderkey: ");
        printf("%8i", it->second);
        printf("  ");
        printf("\n");
        std::advance(it, 1);
    }
    if((ht.size() > 25)) {
        printf("[...sorted...]\n");
    }
    printf("\n");
    std::clock_t stop_finish3 = std::clock();


    printf("<timing>\n");
    printf ( "%32s: %6.1f ms\n", "finish", (stop_finish3 - start_finish3) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem1", (stop_krnl_lineitem11 - start_krnl_lineitem11) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation2", (stop_krnl_aggregation22 - start_krnl_aggregation22) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "totalKernelTime", (stop_totalKernelTime0 - start_totalKernelTime0) / (double) (CLOCKS_PER_SEC / 1000) );
    printf("</timing>\n");
}
