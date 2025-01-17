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
struct jpayl4 {
    int att5_ppartkey;
};
struct apayl8 {
    int att32_lpartkey;
};
struct jpayl10 {
    int att32_lpartkey;
    float att1_avgquan;
};

__global__ void krnl_part1(
    int* iatt5_ppartkey, size_t* iatt8_pbrand_offset, char* iatt8_pbrand_char, size_t* iatt11_pcontain_offset, char* iatt11_pcontain_char, unique_ht<jpayl4>* jht4) {
    int att5_ppartkey;
    str_t att8_pbrand;
    str_t att11_pcontain;
    str_t c1 = stringConstant ( "Brand#23", 8);
    str_t c2 = stringConstant ( "MED BOX", 7);

    int tid_part1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_part1 = loopVar;
        active = (loopVar < 200000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att5_ppartkey = iatt5_ppartkey[tid_part1];
            att8_pbrand = stringScan ( iatt8_pbrand_offset, iatt8_pbrand_char, tid_part1);
            att11_pcontain = stringScan ( iatt11_pcontain_offset, iatt11_pcontain_char, tid_part1);
        }
        // -------- selection (opId: 2) --------
        if(active) {
            active = (stringEquals ( att8_pbrand, c1) && stringEquals ( att11_pcontain, c2));
        }
        // -------- hash join build (opId: 4) --------
        if(active) {
            jpayl4 payl4;
            payl4.att5_ppartkey = att5_ppartkey;
            uint64_t hash4;
            hash4 = 0;
            if(active) {
                hash4 = hash ( (hash4 + ((uint64_t)att5_ppartkey)));
            }
            hashBuildUnique ( jht4, 400, hash4, &(payl4));
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem3(
    int* iatt15_lpartkey, int* iatt18_lquantit, float* iatt19_lextende, unique_ht<jpayl4>* jht4, int* nout_lineitem_filtered, int* itm_lineitem_filtered_l_quantity, float* itm_lineitem_filtered_l_extendedprice, int* itm_lineitem_filtered_l_partkey) {
    int att15_lpartkey;
    int att18_lquantit;
    float att19_lextende;
    int att5_ppartkey;
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));

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
            att15_lpartkey = iatt15_lpartkey[tid_lineitem1];
            att18_lquantit = iatt18_lquantit[tid_lineitem1];
            att19_lextende = iatt19_lextende[tid_lineitem1];
        }
        // -------- hash join probe (opId: 4) --------
        uint64_t hash4 = 0;
        if(active) {
            hash4 = 0;
            if(active) {
                hash4 = hash ( (hash4 + ((uint64_t)att15_lpartkey)));
            }
        }
        jpayl4* probepayl4;
        int numLookups4 = 0;
        if(active) {
            active = hashProbeUnique ( jht4, 400, hash4, numLookups4, &(probepayl4));
        }
        int bucketFound4 = 0;
        int probeActive4 = active;
        while((probeActive4 && !(bucketFound4))) {
            jpayl4 jprobepayl4 = *(probepayl4);
            att5_ppartkey = jprobepayl4.att5_ppartkey;
            bucketFound4 = 1;
            bucketFound4 &= ((att5_ppartkey == att15_lpartkey));
            if(!(bucketFound4)) {
                probeActive4 = hashProbeUnique ( jht4, 400, hash4, numLookups4, &(probepayl4));
            }
        }
        active = bucketFound4;
        // -------- projection (no code) (opId: 5) --------
        // -------- materialize (opId: 6) --------
        int wp;
        int writeMask;
        int numProj;
        writeMask = __ballot_sync(ALL_LANES,active);
        numProj = __popc(writeMask);
        if((warplane == 0)) {
            wp = atomicAdd(nout_lineitem_filtered, numProj);
        }
        wp = __shfl_sync(ALL_LANES,wp,0);
        wp = (wp + __popc((writeMask & prefixlanes)));
        if(active) {
            itm_lineitem_filtered_l_quantity[wp] = att18_lquantit;
            itm_lineitem_filtered_l_extendedprice[wp] = att19_lextende;
            itm_lineitem_filtered_l_partkey[wp] = att15_lpartkey;
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem_filtered7(
    int* itm_lineitem_filtered_l_quantity, int* itm_lineitem_filtered_l_partkey, int* nout_lineitem_filtered, agg_ht<apayl8>* aht8, float* agg1, int* agg2) {
    int att30_lquantit;
    int att32_lpartkey;

    int tid_lineitem_filtered1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_lineitem_filtered1 = loopVar;
        active = (loopVar < *(nout_lineitem_filtered));
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att30_lquantit = itm_lineitem_filtered_l_quantity[tid_lineitem_filtered1];
            att32_lpartkey = itm_lineitem_filtered_l_partkey[tid_lineitem_filtered1];
        }
        // -------- aggregation (opId: 8) --------
        int bucket = 0;
        if(active) {
            uint64_t hash8 = 0;
            hash8 = 0;
            if(active) {
                hash8 = hash ( (hash8 + ((uint64_t)att32_lpartkey)));
            }
            apayl8 payl;
            payl.att32_lpartkey = att32_lpartkey;
            int bucketFound = 0;
            int numLookups = 0;
            while(!(bucketFound)) {
                bucket = hashAggregateGetBucket ( aht8, 120024, hash8, numLookups, &(payl));
                apayl8 probepayl = aht8[bucket].payload;
                bucketFound = 1;
                bucketFound &= ((payl.att32_lpartkey == probepayl.att32_lpartkey));
            }
        }
        if(active) {
            atomicAdd(&(agg1[bucket]), ((float)att30_lquantit));
            atomicAdd(&(agg2[bucket]), ((int)1));
        }
        loopVar += step;
    }

}

__global__ void krnl_aggregation8(
    agg_ht<apayl8>* aht8, float* agg1, int* agg2, unique_ht<jpayl10>* jht10) {
    int att32_lpartkey;
    float att1_avgquan;
    int att2_countagg;

    int tid_aggregation8 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation8 = loopVar;
        active = (loopVar < 120024);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 8) --------
        if(active) {
            active &= ((aht8[tid_aggregation8].lock.lock == OnceLock::LOCK_DONE));
        }
        if(active) {
            apayl8 payl = aht8[tid_aggregation8].payload;
            att32_lpartkey = payl.att32_lpartkey;
        }
        if(active) {
            att1_avgquan = agg1[tid_aggregation8];
            att2_countagg = agg2[tid_aggregation8];
            att1_avgquan = (att1_avgquan / ((float)att2_countagg));
        }
        // -------- hash join build (opId: 10) --------
        if(active) {
            jpayl10 payl10;
            payl10.att32_lpartkey = att32_lpartkey;
            payl10.att1_avgquan = att1_avgquan;
            uint64_t hash10;
            hash10 = 0;
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att32_lpartkey)));
            }
            hashBuildUnique ( jht10, 120024, hash10, &(payl10));
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem_filtered29(
    int* itm_lineitem_filtered_l_quantity, float* itm_lineitem_filtered_l_extendedprice, int* itm_lineitem_filtered_l_partkey, int* nout_lineitem_filtered, unique_ht<jpayl10>* jht10, float* agg3, int* agg4) {
    int att33_lquantit;
    float att34_lextende;
    int att35_lpartkey;
    int att32_lpartkey;
    float att1_avgquan;
    float att36_limquan;

    int tid_lineitem_filtered2 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_lineitem_filtered2 = loopVar;
        active = (loopVar < *(nout_lineitem_filtered));
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att33_lquantit = itm_lineitem_filtered_l_quantity[tid_lineitem_filtered2];
            att34_lextende = itm_lineitem_filtered_l_extendedprice[tid_lineitem_filtered2];
            att35_lpartkey = itm_lineitem_filtered_l_partkey[tid_lineitem_filtered2];
        }
        // -------- hash join probe (opId: 10) --------
        uint64_t hash10 = 0;
        if(active) {
            hash10 = 0;
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att35_lpartkey)));
            }
        }
        jpayl10* probepayl10;
        int numLookups10 = 0;
        if(active) {
            active = hashProbeUnique ( jht10, 120024, hash10, numLookups10, &(probepayl10));
        }
        int bucketFound10 = 0;
        int probeActive10 = active;
        while((probeActive10 && !(bucketFound10))) {
            jpayl10 jprobepayl10 = *(probepayl10);
            att32_lpartkey = jprobepayl10.att32_lpartkey;
            att1_avgquan = jprobepayl10.att1_avgquan;
            bucketFound10 = 1;
            bucketFound10 &= ((att32_lpartkey == att35_lpartkey));
            if(!(bucketFound10)) {
                probeActive10 = hashProbeUnique ( jht10, 120024, hash10, numLookups10, &(probepayl10));
            }
        }
        active = bucketFound10;
        // -------- map (opId: 11) --------
        if(active) {
            att36_limquan = (att1_avgquan * 0.2f);
        }
        // -------- selection (opId: 12) --------
        if(active) {
            active = (att33_lquantit < att36_limquan);
        }
        // -------- aggregation (opId: 13) --------
        int bucket = 0;
        if(active) {
            atomicAdd(&(agg3[bucket]), ((float)att34_lextende));
            atomicAdd(&(agg4[bucket]), ((int)1));
        }
        loopVar += step;
    }

}

__global__ void krnl_aggregation13(
    float* agg3, int* agg4, int* nout_result, float* oatt37_avgyearl, int* oatt4_countpri) {
    float att3_sumprice;
    int att4_countpri;
    float att37_avgyearl;
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));

    int tid_aggregation13 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation13 = loopVar;
        active = (loopVar < 1);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 13) --------
        if(active) {
            att3_sumprice = agg3[tid_aggregation13];
            att4_countpri = agg4[tid_aggregation13];
        }
        // -------- map (opId: 14) --------
        if(active) {
            att37_avgyearl = (att3_sumprice / 7.0f);
        }
        // -------- projection (no code) (opId: 15) --------
        // -------- materialize (opId: 16) --------
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
            oatt37_avgyearl[wp] = att37_avgyearl;
            oatt4_countpri[wp] = att4_countpri;
        }
        loopVar += step;
    }

}

int main() {
    int* iatt5_ppartkey;
    iatt5_ppartkey = ( int*) map_memory_file ( "mmdb/part_p_partkey" );
    size_t* iatt8_pbrand_offset;
    iatt8_pbrand_offset = ( size_t*) map_memory_file ( "mmdb/part_p_brand_offset" );
    char* iatt8_pbrand_char;
    iatt8_pbrand_char = ( char*) map_memory_file ( "mmdb/part_p_brand_char" );
    size_t* iatt11_pcontain_offset;
    iatt11_pcontain_offset = ( size_t*) map_memory_file ( "mmdb/part_p_container_offset" );
    char* iatt11_pcontain_char;
    iatt11_pcontain_char = ( char*) map_memory_file ( "mmdb/part_p_container_char" );
    int* iatt15_lpartkey;
    iatt15_lpartkey = ( int*) map_memory_file ( "mmdb/lineitem_l_partkey" );
    int* iatt18_lquantit;
    iatt18_lquantit = ( int*) map_memory_file ( "mmdb/lineitem_l_quantity" );
    float* iatt19_lextende;
    iatt19_lextende = ( float*) map_memory_file ( "mmdb/lineitem_l_extendedprice" );

    int nout_lineitem_filtered;
    int nout_result;
    std::vector < float > oatt37_avgyearl(1);
    std::vector < int > oatt4_countpri(1);

    // wake up gpu
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in wake up gpu! " << cudaGetErrorString( err ) << std::endl;
            ERROR("wake up gpu")
        }
    }

    int* d_iatt5_ppartkey;
    cudaMalloc((void**) &d_iatt5_ppartkey, 200000* sizeof(int) );
    size_t* d_iatt8_pbrand_offset;
    cudaMalloc((void**) &d_iatt8_pbrand_offset, (200000 + 1)* sizeof(size_t) );
    char* d_iatt8_pbrand_char;
    cudaMalloc((void**) &d_iatt8_pbrand_char, 1600009* sizeof(char) );
    size_t* d_iatt11_pcontain_offset;
    cudaMalloc((void**) &d_iatt11_pcontain_offset, (200000 + 1)* sizeof(size_t) );
    char* d_iatt11_pcontain_char;
    cudaMalloc((void**) &d_iatt11_pcontain_char, 1514980* sizeof(char) );
    int* d_iatt15_lpartkey;
    cudaMalloc((void**) &d_iatt15_lpartkey, 6001215* sizeof(int) );
    int* d_iatt18_lquantit;
    cudaMalloc((void**) &d_iatt18_lquantit, 6001215* sizeof(int) );
    float* d_iatt19_lextende;
    cudaMalloc((void**) &d_iatt19_lextende, 6001215* sizeof(float) );
    int* d_nout_lineitem_filtered;
    cudaMalloc((void**) &d_nout_lineitem_filtered, 1* sizeof(int) );
    int* d_itm_lineitem_filtered_l_quantity;
    cudaMalloc((void**) &d_itm_lineitem_filtered_l_quantity, 60012* sizeof(int) );
    float* d_itm_lineitem_filtered_l_extendedprice;
    cudaMalloc((void**) &d_itm_lineitem_filtered_l_extendedprice, 60012* sizeof(float) );
    int* d_itm_lineitem_filtered_l_partkey;
    cudaMalloc((void**) &d_itm_lineitem_filtered_l_partkey, 60012* sizeof(int) );
    int* d_nout_result;
    cudaMalloc((void**) &d_nout_result, 1* sizeof(int) );
    float* d_oatt37_avgyearl;
    cudaMalloc((void**) &d_oatt37_avgyearl, 1* sizeof(float) );
    int* d_oatt4_countpri;
    cudaMalloc((void**) &d_oatt4_countpri, 1* sizeof(int) );
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

    unique_ht<jpayl4>* d_jht4;
    cudaMalloc((void**) &d_jht4, 400* sizeof(unique_ht<jpayl4>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht4, 400);
    }
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_nout_lineitem_filtered, 0, 1);
    }
    agg_ht<apayl8>* d_aht8;
    cudaMalloc((void**) &d_aht8, 120024* sizeof(agg_ht<apayl8>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_aht8, 120024);
    }
    float* d_agg1;
    cudaMalloc((void**) &d_agg1, 120024* sizeof(float) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg1, 0.0f, 120024);
    }
    int* d_agg2;
    cudaMalloc((void**) &d_agg2, 120024* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg2, 0, 120024);
    }
    unique_ht<jpayl10>* d_jht10;
    cudaMalloc((void**) &d_jht10, 120024* sizeof(unique_ht<jpayl10>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht10, 120024);
    }
    float* d_agg3;
    cudaMalloc((void**) &d_agg3, 1* sizeof(float) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg3, 0.0f, 1);
    }
    int* d_agg4;
    cudaMalloc((void**) &d_agg4, 1* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg4, 0, 1);
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

    cudaMemcpy( d_iatt5_ppartkey, iatt5_ppartkey, 200000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt8_pbrand_offset, iatt8_pbrand_offset, (200000 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt8_pbrand_char, iatt8_pbrand_char, 1600009 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt11_pcontain_offset, iatt11_pcontain_offset, (200000 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt11_pcontain_char, iatt11_pcontain_char, 1514980 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt15_lpartkey, iatt15_lpartkey, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt18_lquantit, iatt18_lquantit, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt19_lextende, iatt19_lextende, 6001215 * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy in! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy in")
        }
    }

    std::clock_t start_totalKernelTime145 = std::clock();
    std::clock_t start_krnl_part1146 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_part1<<<gridsize, blocksize>>>(d_iatt5_ppartkey, d_iatt8_pbrand_offset, d_iatt8_pbrand_char, d_iatt11_pcontain_offset, d_iatt11_pcontain_char, d_jht4);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_part1146 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_part1! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_part1")
        }
    }

    std::clock_t start_krnl_lineitem3147 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem3<<<gridsize, blocksize>>>(d_iatt15_lpartkey, d_iatt18_lquantit, d_iatt19_lextende, d_jht4, d_nout_lineitem_filtered, d_itm_lineitem_filtered_l_quantity, d_itm_lineitem_filtered_l_extendedprice, d_itm_lineitem_filtered_l_partkey);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem3147 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem3! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem3")
        }
    }

    std::clock_t start_krnl_lineitem_filtered7148 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem_filtered7<<<gridsize, blocksize>>>(d_itm_lineitem_filtered_l_quantity, d_itm_lineitem_filtered_l_partkey, d_nout_lineitem_filtered, d_aht8, d_agg1, d_agg2);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem_filtered7148 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem_filtered7! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem_filtered7")
        }
    }

    std::clock_t start_krnl_aggregation8149 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation8<<<gridsize, blocksize>>>(d_aht8, d_agg1, d_agg2, d_jht10);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation8149 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation8! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation8")
        }
    }

    std::clock_t start_krnl_lineitem_filtered29150 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem_filtered29<<<gridsize, blocksize>>>(d_itm_lineitem_filtered_l_quantity, d_itm_lineitem_filtered_l_extendedprice, d_itm_lineitem_filtered_l_partkey, d_nout_lineitem_filtered, d_jht10, d_agg3, d_agg4);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem_filtered29150 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem_filtered29! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem_filtered29")
        }
    }

    std::clock_t start_krnl_aggregation13151 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation13<<<gridsize, blocksize>>>(d_agg3, d_agg4, d_nout_result, d_oatt37_avgyearl, d_oatt4_countpri);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation13151 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation13! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation13")
        }
    }

    std::clock_t stop_totalKernelTime145 = std::clock();
    cudaMemcpy( &nout_lineitem_filtered, d_nout_lineitem_filtered, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( &nout_result, d_nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt37_avgyearl.data(), d_oatt37_avgyearl, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt4_countpri.data(), d_oatt4_countpri, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy out! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy out")
        }
    }

    cudaFree( d_iatt5_ppartkey);
    cudaFree( d_iatt8_pbrand_offset);
    cudaFree( d_iatt8_pbrand_char);
    cudaFree( d_iatt11_pcontain_offset);
    cudaFree( d_iatt11_pcontain_char);
    cudaFree( d_jht4);
    cudaFree( d_iatt15_lpartkey);
    cudaFree( d_iatt18_lquantit);
    cudaFree( d_iatt19_lextende);
    cudaFree( d_nout_lineitem_filtered);
    cudaFree( d_itm_lineitem_filtered_l_quantity);
    cudaFree( d_itm_lineitem_filtered_l_extendedprice);
    cudaFree( d_itm_lineitem_filtered_l_partkey);
    cudaFree( d_aht8);
    cudaFree( d_agg1);
    cudaFree( d_agg2);
    cudaFree( d_jht10);
    cudaFree( d_agg3);
    cudaFree( d_agg4);
    cudaFree( d_nout_result);
    cudaFree( d_oatt37_avgyearl);
    cudaFree( d_oatt4_countpri);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda free! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda free")
        }
    }

    std::clock_t start_finish152 = std::clock();
    printf("\nResult: %i tuples\n", nout_result);
    if((nout_result > 1)) {
        ERROR("Index out of range. Output size larger than allocated with expected result number.")
    }
    for ( int pv = 0; ((pv < 10) && (pv < nout_result)); pv += 1) {
        printf("avg_yearly: ");
        printf("%15.2f", oatt37_avgyearl[pv]);
        printf("  ");
        printf("count_price: ");
        printf("%8i", oatt4_countpri[pv]);
        printf("  ");
        printf("\n");
    }
    if((nout_result > 10)) {
        printf("[...]\n");
    }
    printf("\n");
    std::clock_t stop_finish152 = std::clock();

    printf("<timing>\n");
    printf ( "%32s: %6.1f ms\n", "krnl_part1", (stop_krnl_part1146 - start_krnl_part1146) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem3", (stop_krnl_lineitem3147 - start_krnl_lineitem3147) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem_filtered7", (stop_krnl_lineitem_filtered7148 - start_krnl_lineitem_filtered7148) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation8", (stop_krnl_aggregation8149 - start_krnl_aggregation8149) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem_filtered29", (stop_krnl_lineitem_filtered29150 - start_krnl_lineitem_filtered29150) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation13", (stop_krnl_aggregation13151 - start_krnl_aggregation13151) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "finish", (stop_finish152 - start_finish152) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "totalKernelTime", (stop_totalKernelTime145 - start_totalKernelTime145) / (double) (CLOCKS_PER_SEC / 1000) );
    printf("</timing>\n");
}
