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
    int att4_lpartkey;
    float att8_lextende;
    float att9_ldiscoun;
};

__global__ void krnl_lineitem1(
    int* iatt4_lpartkey, float* iatt8_lextende, float* iatt9_ldiscoun, unsigned* iatt13_lshipdat, multi_ht* jht4, jpayl4* jht4_payload) {
    int att4_lpartkey;
    float att8_lextende;
    float att9_ldiscoun;
    unsigned att13_lshipdat;

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
            att4_lpartkey = iatt4_lpartkey[tid_lineitem1];
            att8_lextende = iatt8_lextende[tid_lineitem1];
            att9_ldiscoun = iatt9_ldiscoun[tid_lineitem1];
            att13_lshipdat = iatt13_lshipdat[tid_lineitem1];
        }
        // -------- selection (opId: 2) --------
        if(active) {
            active = ((att13_lshipdat >= 19950901) && (att13_lshipdat < 19951001));
        }
        // -------- hash join build (opId: 4) --------
        if(active) {
            uint64_t hash4 = 0;
            if(active) {
                hash4 = 0;
                if(active) {
                    hash4 = hash ( (hash4 + ((uint64_t)att4_lpartkey)));
                }
            }
            hashCountMulti ( jht4, 240048, hash4);
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem1_ins(
    int* iatt4_lpartkey, float* iatt8_lextende, float* iatt9_ldiscoun, unsigned* iatt13_lshipdat, multi_ht* jht4, jpayl4* jht4_payload, int* offs4) {
    int att4_lpartkey;
    float att8_lextende;
    float att9_ldiscoun;
    unsigned att13_lshipdat;

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
            att4_lpartkey = iatt4_lpartkey[tid_lineitem1];
            att8_lextende = iatt8_lextende[tid_lineitem1];
            att9_ldiscoun = iatt9_ldiscoun[tid_lineitem1];
            att13_lshipdat = iatt13_lshipdat[tid_lineitem1];
        }
        // -------- selection (opId: 2) --------
        if(active) {
            active = ((att13_lshipdat >= 19950901) && (att13_lshipdat < 19951001));
        }
        // -------- hash join build (opId: 4) --------
        if(active) {
            uint64_t hash4 = 0;
            if(active) {
                hash4 = 0;
                if(active) {
                    hash4 = hash ( (hash4 + ((uint64_t)att4_lpartkey)));
                }
            }
            jpayl4 payl;
            payl.att4_lpartkey = att4_lpartkey;
            payl.att8_lextende = att8_lextende;
            payl.att9_ldiscoun = att9_ldiscoun;
            hashInsertMulti ( jht4, jht4_payload, offs4, 240048, hash4, &(payl));
        }
        loopVar += step;
    }

}

__global__ void krnl_part3(
    int* iatt19_ppartkey, size_t* iatt23_ptype_offset, char* iatt23_ptype_char, multi_ht* jht4, jpayl4* jht4_payload, float* agg1, float* agg2) {
    int att19_ppartkey;
    str_t att23_ptype;
    unsigned warplane = (threadIdx.x % 32);
    int att4_lpartkey;
    float att8_lextende;
    float att9_ldiscoun;
    float att28_promo;
    str_t c1 = stringConstant ( "PROMO%", 6);
    float att29_rev;

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
            att19_ppartkey = iatt19_ppartkey[tid_part1];
            att23_ptype = stringScan ( iatt23_ptype_offset, iatt23_ptype_char, tid_part1);
        }
        // -------- hash join probe (opId: 4) --------
        // -------- multiprobe multi broadcast (opId: 4) --------
        int matchEnd4 = 0;
        int matchEndBuf4 = 0;
        int matchOffset4 = 0;
        int matchOffsetBuf4 = 0;
        int probeActive4 = active;
        int att19_ppartkey_bcbuf4;
        str_t att23_ptype_bcbuf4;
        uint64_t hash4 = 0;
        if(probeActive4) {
            hash4 = 0;
            if(active) {
                hash4 = hash ( (hash4 + ((uint64_t)att19_ppartkey)));
            }
            probeActive4 = hashProbeMulti ( jht4, 240048, hash4, matchOffsetBuf4, matchEndBuf4);
        }
        unsigned activeProbes4 = __ballot_sync(ALL_LANES,probeActive4);
        int num4 = 0;
        num4 = (matchEndBuf4 - matchOffsetBuf4);
        unsigned wideProbes4 = __ballot_sync(ALL_LANES,(num4 >= 32));
        att19_ppartkey_bcbuf4 = att19_ppartkey;
        att23_ptype_bcbuf4 = att23_ptype;
        while((activeProbes4 > 0)) {
            unsigned tupleLane;
            unsigned broadcastLane;
            int numFilled = 0;
            int num = 0;
            while(((numFilled < 32) && activeProbes4)) {
                if((wideProbes4 > 0)) {
                    tupleLane = (__ffs(wideProbes4) - 1);
                    wideProbes4 -= (1 << tupleLane);
                }
                else {
                    tupleLane = (__ffs(activeProbes4) - 1);
                }
                num = __shfl_sync(ALL_LANES,num4,tupleLane);
                if((numFilled && ((numFilled + num) > 32))) {
                    break;
                }
                if((warplane >= numFilled)) {
                    broadcastLane = tupleLane;
                    matchOffset4 = (warplane - numFilled);
                }
                numFilled += num;
                activeProbes4 -= (1 << tupleLane);
            }
            matchOffset4 += __shfl_sync(ALL_LANES,matchOffsetBuf4,broadcastLane);
            matchEnd4 = __shfl_sync(ALL_LANES,matchEndBuf4,broadcastLane);
            att19_ppartkey = __shfl_sync(ALL_LANES,att19_ppartkey_bcbuf4,broadcastLane);
            att23_ptype = __shfl_sync(ALL_LANES,att23_ptype_bcbuf4,broadcastLane);
            probeActive4 = (matchOffset4 < matchEnd4);
            while(__any_sync(ALL_LANES,probeActive4)) {
                active = probeActive4;
                active = 0;
                jpayl4 payl;
                if(probeActive4) {
                    payl = jht4_payload[matchOffset4];
                    att4_lpartkey = payl.att4_lpartkey;
                    att8_lextende = payl.att8_lextende;
                    att9_ldiscoun = payl.att9_ldiscoun;
                    active = 1;
                    active &= ((att4_lpartkey == att19_ppartkey));
                    matchOffset4 += 32;
                    probeActive4 &= ((matchOffset4 < matchEnd4));
                }
                // -------- map (opId: 5) --------
                if(active) {
                    float casevar1273;
                    if(stringLikeCheck ( att23_ptype, c1)) {
                        casevar1273 = (att8_lextende * (1.0f - att9_ldiscoun));
                    }
                    else {
                        casevar1273 = 0.0f;
                    }
                    att28_promo = casevar1273;
                }
                // -------- map (opId: 6) --------
                if(active) {
                    att29_rev = (att8_lextende * ((float)1.0f - att9_ldiscoun));
                }
                // -------- aggregation (opId: 7) --------
                int bucket = 0;
                if(active) {
                    atomicAdd(&(agg1[bucket]), ((float)att29_rev));
                    atomicAdd(&(agg2[bucket]), ((float)att28_promo));
                }
            }
        }
        loopVar += step;
    }

}

__global__ void krnl_aggregation7(
    float* agg1, float* agg2, int* nout_result, float* oatt30_promorev) {
    float att1_sumrev;
    float att2_sumpromo;
    float att30_promorev;
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));

    int tid_aggregation7 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation7 = loopVar;
        active = (loopVar < 1);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 7) --------
        if(active) {
            att1_sumrev = agg1[tid_aggregation7];
            att2_sumpromo = agg2[tid_aggregation7];
        }
        // -------- map (opId: 8) --------
        if(active) {
            att30_promorev = ((100.0f * att2_sumpromo) / att1_sumrev);
        }
        // -------- projection (no code) (opId: 9) --------
        // -------- materialize (opId: 10) --------
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
            oatt30_promorev[wp] = att30_promorev;
        }
        loopVar += step;
    }

}

int main() {
    int* iatt4_lpartkey;
    iatt4_lpartkey = ( int*) map_memory_file ( "mmdb/lineitem_l_partkey" );
    float* iatt8_lextende;
    iatt8_lextende = ( float*) map_memory_file ( "mmdb/lineitem_l_extendedprice" );
    float* iatt9_ldiscoun;
    iatt9_ldiscoun = ( float*) map_memory_file ( "mmdb/lineitem_l_discount" );
    unsigned* iatt13_lshipdat;
    iatt13_lshipdat = ( unsigned*) map_memory_file ( "mmdb/lineitem_l_shipdate" );
    int* iatt19_ppartkey;
    iatt19_ppartkey = ( int*) map_memory_file ( "mmdb/part_p_partkey" );
    size_t* iatt23_ptype_offset;
    iatt23_ptype_offset = ( size_t*) map_memory_file ( "mmdb/part_p_type_offset" );
    char* iatt23_ptype_char;
    iatt23_ptype_char = ( char*) map_memory_file ( "mmdb/part_p_type_char" );

    int nout_result;
    std::vector < float > oatt30_promorev(1);

    // wake up gpu
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in wake up gpu! " << cudaGetErrorString( err ) << std::endl;
            ERROR("wake up gpu")
        }
    }

    int* d_iatt4_lpartkey;
    cudaMalloc((void**) &d_iatt4_lpartkey, 6001215* sizeof(int) );
    float* d_iatt8_lextende;
    cudaMalloc((void**) &d_iatt8_lextende, 6001215* sizeof(float) );
    float* d_iatt9_ldiscoun;
    cudaMalloc((void**) &d_iatt9_ldiscoun, 6001215* sizeof(float) );
    unsigned* d_iatt13_lshipdat;
    cudaMalloc((void**) &d_iatt13_lshipdat, 6001215* sizeof(unsigned) );
    int* d_iatt19_ppartkey;
    cudaMalloc((void**) &d_iatt19_ppartkey, 200000* sizeof(int) );
    size_t* d_iatt23_ptype_offset;
    cudaMalloc((void**) &d_iatt23_ptype_offset, (200000 + 1)* sizeof(size_t) );
    char* d_iatt23_ptype_char;
    cudaMalloc((void**) &d_iatt23_ptype_char, 4119955* sizeof(char) );
    int* d_nout_result;
    cudaMalloc((void**) &d_nout_result, 1* sizeof(int) );
    float* d_oatt30_promorev;
    cudaMalloc((void**) &d_oatt30_promorev, 1* sizeof(float) );
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

    multi_ht* d_jht4;
    cudaMalloc((void**) &d_jht4, 240048* sizeof(multi_ht) );
    jpayl4* d_jht4_payload;
    cudaMalloc((void**) &d_jht4_payload, 240048* sizeof(jpayl4) );
    {
        int gridsize=920;
        int blocksize=128;
        initMultiHT<<<gridsize, blocksize>>>(d_jht4, 240048);
    }
    int* d_offs4;
    cudaMalloc((void**) &d_offs4, 1* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_offs4, 0, 1);
    }
    float* d_agg1;
    cudaMalloc((void**) &d_agg1, 1* sizeof(float) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg1, 0.0f, 1);
    }
    float* d_agg2;
    cudaMalloc((void**) &d_agg2, 1* sizeof(float) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg2, 0.0f, 1);
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

    cudaMemcpy( d_iatt4_lpartkey, iatt4_lpartkey, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt8_lextende, iatt8_lextende, 6001215 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt9_ldiscoun, iatt9_ldiscoun, 6001215 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt13_lshipdat, iatt13_lshipdat, 6001215 * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt19_ppartkey, iatt19_ppartkey, 200000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt23_ptype_offset, iatt23_ptype_offset, (200000 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt23_ptype_char, iatt23_ptype_char, 4119955 * sizeof(char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy in! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy in")
        }
    }

    std::clock_t start_totalKernelTime119 = std::clock();
    std::clock_t start_krnl_lineitem1120 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem1<<<gridsize, blocksize>>>(d_iatt4_lpartkey, d_iatt8_lextende, d_iatt9_ldiscoun, d_iatt13_lshipdat, d_jht4, d_jht4_payload);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem1120 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem1! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem1")
        }
    }

    std::clock_t start_scanMultiHT121 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        scanMultiHT<<<gridsize, blocksize>>>(d_jht4, 240048, d_offs4);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_scanMultiHT121 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in scanMultiHT! " << cudaGetErrorString( err ) << std::endl;
            ERROR("scanMultiHT")
        }
    }

    std::clock_t start_krnl_lineitem1_ins122 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem1_ins<<<gridsize, blocksize>>>(d_iatt4_lpartkey, d_iatt8_lextende, d_iatt9_ldiscoun, d_iatt13_lshipdat, d_jht4, d_jht4_payload, d_offs4);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem1_ins122 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem1_ins! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem1_ins")
        }
    }

    std::clock_t start_krnl_part3123 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_part3<<<gridsize, blocksize>>>(d_iatt19_ppartkey, d_iatt23_ptype_offset, d_iatt23_ptype_char, d_jht4, d_jht4_payload, d_agg1, d_agg2);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_part3123 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_part3! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_part3")
        }
    }

    std::clock_t start_krnl_aggregation7124 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation7<<<gridsize, blocksize>>>(d_agg1, d_agg2, d_nout_result, d_oatt30_promorev);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation7124 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation7! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation7")
        }
    }

    std::clock_t stop_totalKernelTime119 = std::clock();
    cudaMemcpy( &nout_result, d_nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt30_promorev.data(), d_oatt30_promorev, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy out! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy out")
        }
    }

    cudaFree( d_iatt4_lpartkey);
    cudaFree( d_iatt8_lextende);
    cudaFree( d_iatt9_ldiscoun);
    cudaFree( d_iatt13_lshipdat);
    cudaFree( d_jht4);
    cudaFree( d_jht4_payload);
    cudaFree( d_offs4);
    cudaFree( d_iatt19_ppartkey);
    cudaFree( d_iatt23_ptype_offset);
    cudaFree( d_iatt23_ptype_char);
    cudaFree( d_agg1);
    cudaFree( d_agg2);
    cudaFree( d_nout_result);
    cudaFree( d_oatt30_promorev);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda free! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda free")
        }
    }

    std::clock_t start_finish125 = std::clock();
    printf("\nResult: %i tuples\n", nout_result);
    if((nout_result > 1)) {
        ERROR("Index out of range. Output size larger than allocated with expected result number.")
    }
    for ( int pv = 0; ((pv < 10) && (pv < nout_result)); pv += 1) {
        printf("promo_revenue: ");
        printf("%15.2f", oatt30_promorev[pv]);
        printf("  ");
        printf("\n");
    }
    if((nout_result > 10)) {
        printf("[...]\n");
    }
    printf("\n");
    std::clock_t stop_finish125 = std::clock();

    printf("<timing>\n");
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem1", (stop_krnl_lineitem1120 - start_krnl_lineitem1120) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "scanMultiHT", (stop_scanMultiHT121 - start_scanMultiHT121) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem1_ins", (stop_krnl_lineitem1_ins122 - start_krnl_lineitem1_ins122) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_part3", (stop_krnl_part3123 - start_krnl_part3123) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation7", (stop_krnl_aggregation7124 - start_krnl_aggregation7124) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "finish", (stop_finish125 - start_finish125) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "totalKernelTime", (stop_totalKernelTime119 - start_totalKernelTime119) / (double) (CLOCKS_PER_SEC / 1000) );
    printf("</timing>\n");
}
