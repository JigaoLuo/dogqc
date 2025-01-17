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
    int att6_lsuppkey;
    int att4_lorderke;
};


__device__ bool operator==(const apayl2& lhs, const apayl2& rhs) {
    return lhs.att6_lsuppkey == rhs.att6_lsuppkey && lhs.att4_lorderke == rhs.att4_lorderke;
}

__global__ void krnl_lineitem1(
    int* iatt4_lorderke, int* iatt5_lpartkey, int* iatt6_lsuppkey, int* iatt7_llinenum, agg_ht<apayl2>* aht2, int* agg1, int* agg2, int* agg3) {
    int att4_lorderke;
    int att5_lpartkey;
    int att6_lsuppkey;
    int att7_llinenum;

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
            att4_lorderke = iatt4_lorderke[tid_lineitem1];
            att5_lpartkey = iatt5_lpartkey[tid_lineitem1];
            att6_lsuppkey = iatt6_lsuppkey[tid_lineitem1];
            att7_llinenum = iatt7_llinenum[tid_lineitem1];
        }
        // -------- aggregation (opId: 2) --------
        int bucket = 0;
        if(active) {
            uint64_t hash2 = 0;
            hash2 = 0;
            if(active) {
                hash2 = hash ( (hash2 + ((uint64_t)att6_lsuppkey)));
            }
            if(active) {
                hash2 = hash ( (hash2 + ((uint64_t)att4_lorderke)));
            }
            apayl2 payl;
            payl.att6_lsuppkey = att6_lsuppkey;
            payl.att4_lorderke = att4_lorderke;
            int bucketFound = 0;
            int numLookups = 0;
            while(!(bucketFound)) {
                bucket = hashAggregateGetBucket ( aht2, 12002430, hash2, numLookups, &(payl));
                apayl2 probepayl = aht2[bucket].payload;
                bucketFound = 1;
                bucketFound &= ((payl.att6_lsuppkey == probepayl.att6_lsuppkey));
                bucketFound &= ((payl.att4_lorderke == probepayl.att4_lorderke));
            }
        }
        if(active) {
            atomicAdd(&(agg1[bucket]), ((int)1));
            atomicAdd(&(agg2[bucket]), ((int)1));
            atomicAdd(&(agg3[bucket]), ((int)1));
        }
        loopVar += step;
    }

}

__global__ void krnl_aggregation2(
    agg_ht<apayl2>* aht2, int* agg1, int* agg2, int* agg3, int* nout_result, int* oatt6_lsuppkey, int* oatt4_lorderke, int* oatt1_countlpa, int* oatt2_countlli, int* oatt3_countlsu) {
    int att6_lsuppkey;
    int att4_lorderke;
    int att1_countlpa;
    int att2_countlli;
    int att3_countlsu;
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));

    int tid_aggregation2 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation2 = loopVar;
        active = (loopVar < 12002430);
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
            att6_lsuppkey = payl.att6_lsuppkey;
            att4_lorderke = payl.att4_lorderke;
        }
        if(active) {
            att1_countlpa = agg1[tid_aggregation2];
            att2_countlli = agg2[tid_aggregation2];
            att3_countlsu = agg3[tid_aggregation2];
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
            oatt6_lsuppkey[wp] = att6_lsuppkey;
            oatt4_lorderke[wp] = att4_lorderke;
            oatt1_countlpa[wp] = att1_countlpa;
            oatt2_countlli[wp] = att2_countlli;
            oatt3_countlsu[wp] = att3_countlsu;
        }
        loopVar += step;
    }

}

int main() {
    int* iatt4_lorderke;
    iatt4_lorderke = ( int*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_orderkey" );
    int* iatt5_lpartkey;
    iatt5_lpartkey = ( int*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_partkey" );
    int* iatt6_lsuppkey;
    iatt6_lsuppkey = ( int*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_suppkey" );
    int* iatt7_llinenum;
    iatt7_llinenum = ( int*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_linenumber" );

    int nout_result;
    std::vector < int > oatt6_lsuppkey(6001215);
    std::vector < int > oatt4_lorderke(6001215);
    std::vector < int > oatt1_countlpa(6001215);
    std::vector < int > oatt2_countlli(6001215);
    std::vector < int > oatt3_countlsu(6001215);

    // wake up gpu
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in wake up gpu! " << cudaGetErrorString( err ) << std::endl;
            ERROR("wake up gpu")
        }
    }

    int* d_iatt4_lorderke;
    cudaMalloc((void**) &d_iatt4_lorderke, 6001215* sizeof(int) );
    int* d_iatt5_lpartkey;
    cudaMalloc((void**) &d_iatt5_lpartkey, 6001215* sizeof(int) );
    int* d_iatt6_lsuppkey;
    cudaMalloc((void**) &d_iatt6_lsuppkey, 6001215* sizeof(int) );
    int* d_iatt7_llinenum;
    cudaMalloc((void**) &d_iatt7_llinenum, 6001215* sizeof(int) );
    int* d_nout_result;
    cudaMalloc((void**) &d_nout_result, 1* sizeof(int) );
    int* d_oatt6_lsuppkey;
    cudaMalloc((void**) &d_oatt6_lsuppkey, 6001215* sizeof(int) );
    int* d_oatt4_lorderke;
    cudaMalloc((void**) &d_oatt4_lorderke, 6001215* sizeof(int) );
    int* d_oatt1_countlpa;
    cudaMalloc((void**) &d_oatt1_countlpa, 6001215* sizeof(int) );
    int* d_oatt2_countlli;
    cudaMalloc((void**) &d_oatt2_countlli, 6001215* sizeof(int) );
    int* d_oatt3_countlsu;
    cudaMalloc((void**) &d_oatt3_countlsu, 6001215* sizeof(int) );
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
    cudaMalloc((void**) &d_aht2, 12002430* sizeof(agg_ht<apayl2>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_aht2, 12002430);
    }
    int* d_agg1;
    cudaMalloc((void**) &d_agg1, 12002430* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg1, 0, 12002430);
    }
    int* d_agg2;
    cudaMalloc((void**) &d_agg2, 12002430* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg2, 0, 12002430);
    }
    int* d_agg3;
    cudaMalloc((void**) &d_agg3, 12002430* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg3, 0, 12002430);
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

    cudaMemcpy( d_iatt4_lorderke, iatt4_lorderke, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt5_lpartkey, iatt5_lpartkey, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt6_lsuppkey, iatt6_lsuppkey, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt7_llinenum, iatt7_llinenum, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
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
        krnl_lineitem1<<<gridsize, blocksize>>>(d_iatt4_lorderke, d_iatt5_lpartkey, d_iatt6_lsuppkey, d_iatt7_llinenum, d_aht2, d_agg1, d_agg2, d_agg3);
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
        krnl_aggregation2<<<gridsize, blocksize>>>(d_aht2, d_agg1, d_agg2, d_agg3, d_nout_result, d_oatt6_lsuppkey, d_oatt4_lorderke, d_oatt1_countlpa, d_oatt2_countlli, d_oatt3_countlsu);
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
    cudaMemcpy( oatt6_lsuppkey.data(), d_oatt6_lsuppkey, 6001215 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt4_lorderke.data(), d_oatt4_lorderke, 6001215 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt1_countlpa.data(), d_oatt1_countlpa, 6001215 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt2_countlli.data(), d_oatt2_countlli, 6001215 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt3_countlsu.data(), d_oatt3_countlsu, 6001215 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy out! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy out")
        }
    }

    cudaFree( d_iatt4_lorderke);
    cudaFree( d_iatt5_lpartkey);
    cudaFree( d_iatt6_lsuppkey);
    cudaFree( d_iatt7_llinenum);
    cudaFree( d_aht2);
    cudaFree( d_agg1);
    cudaFree( d_agg2);
    cudaFree( d_agg3);
    cudaFree( d_nout_result);
    cudaFree( d_oatt6_lsuppkey);
    cudaFree( d_oatt4_lorderke);
    cudaFree( d_oatt1_countlpa);
    cudaFree( d_oatt2_countlli);
    cudaFree( d_oatt3_countlsu);
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
        printf("%8i", oatt6_lsuppkey[pv]);
        printf("  ");
        printf("l_orderkey: ");
        printf("%8i", oatt4_lorderke[pv]);
        printf("  ");
        printf("count_l_partkey: ");
        printf("%8i", oatt1_countlpa[pv]);
        printf("  ");
        printf("count_l_linenumber: ");
        printf("%8i", oatt2_countlli[pv]);
        printf("  ");
        printf("count_l_suppkey_orderkey: ");
        printf("%8i", oatt3_countlsu[pv]);
        printf("  ");
        printf("\n");
    }
    if((nout_result > 10)) {
        printf("[...]\n");
    }
    printf("\n");
    std::clock_t stop_finish3 = std::clock();

    printf("<timing>\n");
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem1", (stop_krnl_lineitem11 - start_krnl_lineitem11) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation2", (stop_krnl_aggregation22 - start_krnl_aggregation22) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "finish", (stop_finish3 - start_finish3) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "totalKernelTime", (stop_totalKernelTime0 - start_totalKernelTime0) / (double) (CLOCKS_PER_SEC / 1000) );
    printf("</timing>\n");
}
