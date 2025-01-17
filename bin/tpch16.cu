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
struct jpayl7 {
    int att3_ssuppkey;
};
struct jpayl6 {
    int att10_ppartkey;
    str_t att13_pbrand;
    str_t att14_ptype;
    int att15_psize;
};
struct apayl8 {
    str_t att13_pbrand;
    str_t att14_ptype;
    int att15_psize;
    int att20_pssuppke;
};
struct apayl9 {
    str_t att13_pbrand;
    str_t att14_ptype;
    int att15_psize;
};

__global__ void krnl_supplier1(
    int* iatt3_ssuppkey, size_t* iatt9_scomment_offset, char* iatt9_scomment_char, agg_ht<jpayl7>* jht7) {
    int att3_ssuppkey;
    str_t att9_scomment;
    str_t c1 = stringConstant ( "%Customer%Complaints%", 21);

    int tid_supplier1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_supplier1 = loopVar;
        active = (loopVar < 10000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att3_ssuppkey = iatt3_ssuppkey[tid_supplier1];
            att9_scomment = stringScan ( iatt9_scomment_offset, iatt9_scomment_char, tid_supplier1);
        }
        // -------- selection (opId: 2) --------
        if(active) {
            active = stringLikeCheck ( att9_scomment, c1);
        }
        // -------- hash join build (opId: 7) --------
        if(active) {
            uint64_t hash7;
            hash7 = 0;
            if(active) {
                hash7 = hash ( (hash7 + ((uint64_t)att3_ssuppkey)));
            }
            int bucket = 0;
            jpayl7 payl7;
            payl7.att3_ssuppkey = att3_ssuppkey;
            int bucketFound = 0;
            int numLookups = 0;
            while(!(bucketFound)) {
                bucket = hashAggregateGetBucket ( jht7, 20000, hash7, numLookups, &(payl7));
                jpayl7 probepayl = jht7[bucket].payload;
                bucketFound = 1;
                bucketFound &= ((payl7.att3_ssuppkey == probepayl.att3_ssuppkey));
            }
        }
        loopVar += step;
    }

}

__global__ void krnl_part3(
    int* iatt10_ppartkey, size_t* iatt13_pbrand_offset, char* iatt13_pbrand_char, size_t* iatt14_ptype_offset, char* iatt14_ptype_char, int* iatt15_psize, unique_ht<jpayl6>* jht6) {
    int att10_ppartkey;
    str_t att13_pbrand;
    str_t att14_ptype;
    int att15_psize;
    str_t c2 = stringConstant ( "Brand#45", 8);
    str_t c3 = stringConstant ( "MEDIUM POLISHED%", 16);

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
            att10_ppartkey = iatt10_ppartkey[tid_part1];
            att13_pbrand = stringScan ( iatt13_pbrand_offset, iatt13_pbrand_char, tid_part1);
            att14_ptype = stringScan ( iatt14_ptype_offset, iatt14_ptype_char, tid_part1);
            att15_psize = iatt15_psize[tid_part1];
        }
        // -------- selection (opId: 4) --------
        if(active) {
            active = ((!(stringEquals ( att13_pbrand, c2)) && !(stringLikeCheck ( att14_ptype, c3))) && ((att15_psize == 9) || ((att15_psize == 36) || ((att15_psize == 3) || ((att15_psize == 19) || ((att15_psize == 45) || ((att15_psize == 23) || ((att15_psize == 14) || (att15_psize == 49)))))))));
        }
        // -------- hash join build (opId: 6) --------
        if(active) {
            jpayl6 payl6;
            payl6.att10_ppartkey = att10_ppartkey;
            payl6.att13_pbrand = att13_pbrand;
            payl6.att14_ptype = att14_ptype;
            payl6.att15_psize = att15_psize;
            uint64_t hash6;
            hash6 = 0;
            if(active) {
                hash6 = hash ( (hash6 + ((uint64_t)att10_ppartkey)));
            }
            hashBuildUnique ( jht6, 400000, hash6, &(payl6));
        }
        loopVar += step;
    }

}

__global__ void krnl_partsupp5(
    int* iatt19_pspartke, int* iatt20_pssuppke, unique_ht<jpayl6>* jht6, agg_ht<jpayl7>* jht7, agg_ht<apayl8>* aht8, int* agg1) {
    int att19_pspartke;
    int att20_pssuppke;
    int att10_ppartkey;
    str_t att13_pbrand;
    str_t att14_ptype;
    int att15_psize;
    int att3_ssuppkey;

    int tid_partsupp1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_partsupp1 = loopVar;
        active = (loopVar < 800000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att19_pspartke = iatt19_pspartke[tid_partsupp1];
            att20_pssuppke = iatt20_pssuppke[tid_partsupp1];
        }
        // -------- hash join probe (opId: 6) --------
        uint64_t hash6 = 0;
        if(active) {
            hash6 = 0;
            if(active) {
                hash6 = hash ( (hash6 + ((uint64_t)att19_pspartke)));
            }
        }
        jpayl6* probepayl6;
        int numLookups6 = 0;
        if(active) {
            active = hashProbeUnique ( jht6, 400000, hash6, numLookups6, &(probepayl6));
        }
        int bucketFound6 = 0;
        int probeActive6 = active;
        while((probeActive6 && !(bucketFound6))) {
            jpayl6 jprobepayl6 = *(probepayl6);
            att10_ppartkey = jprobepayl6.att10_ppartkey;
            att13_pbrand = jprobepayl6.att13_pbrand;
            att14_ptype = jprobepayl6.att14_ptype;
            att15_psize = jprobepayl6.att15_psize;
            bucketFound6 = 1;
            bucketFound6 &= ((att10_ppartkey == att19_pspartke));
            if(!(bucketFound6)) {
                probeActive6 = hashProbeUnique ( jht6, 400000, hash6, numLookups6, &(probepayl6));
            }
        }
        active = bucketFound6;
        // -------- hash join probe (opId: 7) --------
        if(active) {
            uint64_t hash7 = 0;
            hash7 = 0;
            if(active) {
                hash7 = hash ( (hash7 + ((uint64_t)att20_pssuppke)));
            }
            int numLookups7 = 0;
            int location7 = 0;
            int filterMatch7 = 0;
            int activeProbe7 = 1;
            while((!(filterMatch7) && activeProbe7)) {
                activeProbe7 = hashAggregateFindBucket ( jht7, 20000, hash7, numLookups7, location7);
                if(activeProbe7) {
                    jpayl7 probepayl = jht7[location7].payload;
                    att3_ssuppkey = probepayl.att3_ssuppkey;
                    filterMatch7 = 1;
                    filterMatch7 &= ((att3_ssuppkey == att20_pssuppke));
                }
            }
            active &= (!(filterMatch7));
        }
        // -------- aggregation (opId: 8) --------
        int bucket = 0;
        if(active) {
            uint64_t hash8 = 0;
            hash8 = 0;
            hash8 = hash ( (hash8 + stringHash ( att13_pbrand)));
            hash8 = hash ( (hash8 + stringHash ( att14_ptype)));
            if(active) {
                hash8 = hash ( (hash8 + ((uint64_t)att15_psize)));
            }
            if(active) {
                hash8 = hash ( (hash8 + ((uint64_t)att20_pssuppke)));
            }
            apayl8 payl;
            payl.att13_pbrand = att13_pbrand;
            payl.att14_ptype = att14_ptype;
            payl.att15_psize = att15_psize;
            payl.att20_pssuppke = att20_pssuppke;
            int bucketFound = 0;
            int numLookups = 0;
            while(!(bucketFound)) {
                bucket = hashAggregateGetBucket ( aht8, 1600000, hash8, numLookups, &(payl));
                apayl8 probepayl = aht8[bucket].payload;
                bucketFound = 1;
                bucketFound &= (stringEquals ( payl.att13_pbrand, probepayl.att13_pbrand));
                bucketFound &= (stringEquals ( payl.att14_ptype, probepayl.att14_ptype));
                bucketFound &= ((payl.att15_psize == probepayl.att15_psize));
                bucketFound &= ((payl.att20_pssuppke == probepayl.att20_pssuppke));
            }
        }
        if(active) {
            atomicAdd(&(agg1[bucket]), ((int)1));
        }
        loopVar += step;
    }

}

__global__ void krnl_aggregation8(
    agg_ht<apayl8>* aht8, int* agg1, agg_ht<apayl9>* aht9, int* agg2) {
    str_t att13_pbrand;
    str_t att14_ptype;
    int att15_psize;
    int att20_pssuppke;
    int att1_suppcoun;

    int tid_aggregation8 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation8 = loopVar;
        active = (loopVar < 1600000);
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
            att13_pbrand = payl.att13_pbrand;
            att14_ptype = payl.att14_ptype;
            att15_psize = payl.att15_psize;
            att20_pssuppke = payl.att20_pssuppke;
        }
        if(active) {
            att1_suppcoun = agg1[tid_aggregation8];
        }
        // -------- aggregation (opId: 9) --------
        int bucket = 0;
        if(active) {
            uint64_t hash9 = 0;
            hash9 = 0;
            hash9 = hash ( (hash9 + stringHash ( att13_pbrand)));
            hash9 = hash ( (hash9 + stringHash ( att14_ptype)));
            if(active) {
                hash9 = hash ( (hash9 + ((uint64_t)att15_psize)));
            }
            apayl9 payl;
            payl.att13_pbrand = att13_pbrand;
            payl.att14_ptype = att14_ptype;
            payl.att15_psize = att15_psize;
            int bucketFound = 0;
            int numLookups = 0;
            while(!(bucketFound)) {
                bucket = hashAggregateGetBucket ( aht9, 1600000, hash9, numLookups, &(payl));
                apayl9 probepayl = aht9[bucket].payload;
                bucketFound = 1;
                bucketFound &= (stringEquals ( payl.att13_pbrand, probepayl.att13_pbrand));
                bucketFound &= (stringEquals ( payl.att14_ptype, probepayl.att14_ptype));
                bucketFound &= ((payl.att15_psize == probepayl.att15_psize));
            }
        }
        if(active) {
            atomicAdd(&(agg2[bucket]), ((int)1));
        }
        loopVar += step;
    }

}

__global__ void krnl_aggregation9(
    agg_ht<apayl9>* aht9, int* agg2, int* nout_result, str_offs* oatt13_pbrand_offset, char* iatt13_pbrand_char, str_offs* oatt14_ptype_offset, char* iatt14_ptype_char, int* oatt15_psize, int* oatt2_suppcoun) {
    str_t att13_pbrand;
    str_t att14_ptype;
    int att15_psize;
    int att2_suppcoun;
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));

    int tid_aggregation9 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation9 = loopVar;
        active = (loopVar < 1600000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 9) --------
        if(active) {
            active &= ((aht9[tid_aggregation9].lock.lock == OnceLock::LOCK_DONE));
        }
        if(active) {
            apayl9 payl = aht9[tid_aggregation9].payload;
            att13_pbrand = payl.att13_pbrand;
            att14_ptype = payl.att14_ptype;
            att15_psize = payl.att15_psize;
        }
        if(active) {
            att2_suppcoun = agg2[tid_aggregation9];
        }
        // -------- projection (no code) (opId: 10) --------
        // -------- materialize (opId: 11) --------
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
            oatt13_pbrand_offset[wp] = toStringOffset ( iatt13_pbrand_char, att13_pbrand);
            oatt14_ptype_offset[wp] = toStringOffset ( iatt14_ptype_char, att14_ptype);
            oatt15_psize[wp] = att15_psize;
            oatt2_suppcoun[wp] = att2_suppcoun;
        }
        loopVar += step;
    }

}

int main() {
    int* iatt3_ssuppkey;
    iatt3_ssuppkey = ( int*) map_memory_file ( "mmdb/supplier_s_suppkey" );
    size_t* iatt9_scomment_offset;
    iatt9_scomment_offset = ( size_t*) map_memory_file ( "mmdb/supplier_s_comment_offset" );
    char* iatt9_scomment_char;
    iatt9_scomment_char = ( char*) map_memory_file ( "mmdb/supplier_s_comment_char" );
    int* iatt10_ppartkey;
    iatt10_ppartkey = ( int*) map_memory_file ( "mmdb/part_p_partkey" );
    size_t* iatt13_pbrand_offset;
    iatt13_pbrand_offset = ( size_t*) map_memory_file ( "mmdb/part_p_brand_offset" );
    char* iatt13_pbrand_char;
    iatt13_pbrand_char = ( char*) map_memory_file ( "mmdb/part_p_brand_char" );
    size_t* iatt14_ptype_offset;
    iatt14_ptype_offset = ( size_t*) map_memory_file ( "mmdb/part_p_type_offset" );
    char* iatt14_ptype_char;
    iatt14_ptype_char = ( char*) map_memory_file ( "mmdb/part_p_type_char" );
    int* iatt15_psize;
    iatt15_psize = ( int*) map_memory_file ( "mmdb/part_p_size" );
    int* iatt19_pspartke;
    iatt19_pspartke = ( int*) map_memory_file ( "mmdb/partsupp_ps_partkey" );
    int* iatt20_pssuppke;
    iatt20_pssuppke = ( int*) map_memory_file ( "mmdb/partsupp_ps_suppkey" );

    int nout_result;
    std::vector < str_offs > oatt13_pbrand_offset(800000);
    std::vector < str_offs > oatt14_ptype_offset(800000);
    std::vector < int > oatt15_psize(800000);
    std::vector < int > oatt2_suppcoun(800000);

    // wake up gpu
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in wake up gpu! " << cudaGetErrorString( err ) << std::endl;
            ERROR("wake up gpu")
        }
    }

    int* d_iatt3_ssuppkey;
    cudaMalloc((void**) &d_iatt3_ssuppkey, 10000* sizeof(int) );
    size_t* d_iatt9_scomment_offset;
    cudaMalloc((void**) &d_iatt9_scomment_offset, (10000 + 1)* sizeof(size_t) );
    char* d_iatt9_scomment_char;
    cudaMalloc((void**) &d_iatt9_scomment_char, 623073* sizeof(char) );
    int* d_iatt10_ppartkey;
    cudaMalloc((void**) &d_iatt10_ppartkey, 200000* sizeof(int) );
    size_t* d_iatt13_pbrand_offset;
    cudaMalloc((void**) &d_iatt13_pbrand_offset, (200000 + 1)* sizeof(size_t) );
    char* d_iatt13_pbrand_char;
    cudaMalloc((void**) &d_iatt13_pbrand_char, 1600009* sizeof(char) );
    size_t* d_iatt14_ptype_offset;
    cudaMalloc((void**) &d_iatt14_ptype_offset, (200000 + 1)* sizeof(size_t) );
    char* d_iatt14_ptype_char;
    cudaMalloc((void**) &d_iatt14_ptype_char, 4119955* sizeof(char) );
    int* d_iatt15_psize;
    cudaMalloc((void**) &d_iatt15_psize, 200000* sizeof(int) );
    int* d_iatt19_pspartke;
    cudaMalloc((void**) &d_iatt19_pspartke, 800000* sizeof(int) );
    int* d_iatt20_pssuppke;
    cudaMalloc((void**) &d_iatt20_pssuppke, 800000* sizeof(int) );
    int* d_nout_result;
    cudaMalloc((void**) &d_nout_result, 1* sizeof(int) );
    str_offs* d_oatt13_pbrand_offset;
    cudaMalloc((void**) &d_oatt13_pbrand_offset, 800000* sizeof(str_offs) );
    str_offs* d_oatt14_ptype_offset;
    cudaMalloc((void**) &d_oatt14_ptype_offset, 800000* sizeof(str_offs) );
    int* d_oatt15_psize;
    cudaMalloc((void**) &d_oatt15_psize, 800000* sizeof(int) );
    int* d_oatt2_suppcoun;
    cudaMalloc((void**) &d_oatt2_suppcoun, 800000* sizeof(int) );
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

    agg_ht<jpayl7>* d_jht7;
    cudaMalloc((void**) &d_jht7, 20000* sizeof(agg_ht<jpayl7>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_jht7, 20000);
    }
    unique_ht<jpayl6>* d_jht6;
    cudaMalloc((void**) &d_jht6, 400000* sizeof(unique_ht<jpayl6>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht6, 400000);
    }
    agg_ht<apayl8>* d_aht8;
    cudaMalloc((void**) &d_aht8, 1600000* sizeof(agg_ht<apayl8>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_aht8, 1600000);
    }
    int* d_agg1;
    cudaMalloc((void**) &d_agg1, 1600000* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg1, 0, 1600000);
    }
    agg_ht<apayl9>* d_aht9;
    cudaMalloc((void**) &d_aht9, 1600000* sizeof(agg_ht<apayl9>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_aht9, 1600000);
    }
    int* d_agg2;
    cudaMalloc((void**) &d_agg2, 1600000* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg2, 0, 1600000);
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

    cudaMemcpy( d_iatt3_ssuppkey, iatt3_ssuppkey, 10000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt9_scomment_offset, iatt9_scomment_offset, (10000 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt9_scomment_char, iatt9_scomment_char, 623073 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt10_ppartkey, iatt10_ppartkey, 200000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt13_pbrand_offset, iatt13_pbrand_offset, (200000 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt13_pbrand_char, iatt13_pbrand_char, 1600009 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt14_ptype_offset, iatt14_ptype_offset, (200000 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt14_ptype_char, iatt14_ptype_char, 4119955 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt15_psize, iatt15_psize, 200000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt19_pspartke, iatt19_pspartke, 800000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt20_pssuppke, iatt20_pssuppke, 800000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy in! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy in")
        }
    }

    std::clock_t start_totalKernelTime138 = std::clock();
    std::clock_t start_krnl_supplier1139 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_supplier1<<<gridsize, blocksize>>>(d_iatt3_ssuppkey, d_iatt9_scomment_offset, d_iatt9_scomment_char, d_jht7);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_supplier1139 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_supplier1! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_supplier1")
        }
    }

    std::clock_t start_krnl_part3140 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_part3<<<gridsize, blocksize>>>(d_iatt10_ppartkey, d_iatt13_pbrand_offset, d_iatt13_pbrand_char, d_iatt14_ptype_offset, d_iatt14_ptype_char, d_iatt15_psize, d_jht6);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_part3140 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_part3! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_part3")
        }
    }

    std::clock_t start_krnl_partsupp5141 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_partsupp5<<<gridsize, blocksize>>>(d_iatt19_pspartke, d_iatt20_pssuppke, d_jht6, d_jht7, d_aht8, d_agg1);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_partsupp5141 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_partsupp5! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_partsupp5")
        }
    }

    std::clock_t start_krnl_aggregation8142 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation8<<<gridsize, blocksize>>>(d_aht8, d_agg1, d_aht9, d_agg2);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation8142 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation8! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation8")
        }
    }

    std::clock_t start_krnl_aggregation9143 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation9<<<gridsize, blocksize>>>(d_aht9, d_agg2, d_nout_result, d_oatt13_pbrand_offset, d_iatt13_pbrand_char, d_oatt14_ptype_offset, d_iatt14_ptype_char, d_oatt15_psize, d_oatt2_suppcoun);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation9143 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation9! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation9")
        }
    }

    std::clock_t stop_totalKernelTime138 = std::clock();
    cudaMemcpy( &nout_result, d_nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt13_pbrand_offset.data(), d_oatt13_pbrand_offset, 800000 * sizeof(str_offs), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt14_ptype_offset.data(), d_oatt14_ptype_offset, 800000 * sizeof(str_offs), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt15_psize.data(), d_oatt15_psize, 800000 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt2_suppcoun.data(), d_oatt2_suppcoun, 800000 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy out! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy out")
        }
    }

    cudaFree( d_iatt3_ssuppkey);
    cudaFree( d_iatt9_scomment_offset);
    cudaFree( d_iatt9_scomment_char);
    cudaFree( d_jht7);
    cudaFree( d_iatt10_ppartkey);
    cudaFree( d_iatt13_pbrand_offset);
    cudaFree( d_iatt13_pbrand_char);
    cudaFree( d_iatt14_ptype_offset);
    cudaFree( d_iatt14_ptype_char);
    cudaFree( d_iatt15_psize);
    cudaFree( d_jht6);
    cudaFree( d_iatt19_pspartke);
    cudaFree( d_iatt20_pssuppke);
    cudaFree( d_aht8);
    cudaFree( d_agg1);
    cudaFree( d_aht9);
    cudaFree( d_agg2);
    cudaFree( d_nout_result);
    cudaFree( d_oatt13_pbrand_offset);
    cudaFree( d_oatt14_ptype_offset);
    cudaFree( d_oatt15_psize);
    cudaFree( d_oatt2_suppcoun);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda free! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda free")
        }
    }

    std::clock_t start_finish144 = std::clock();
    printf("\nResult: %i tuples\n", nout_result);
    if((nout_result > 800000)) {
        ERROR("Index out of range. Output size larger than allocated with expected result number.")
    }
    for ( int pv = 0; ((pv < 10) && (pv < nout_result)); pv += 1) {
        printf("p_brand: ");
        stringPrint ( iatt13_pbrand_char, oatt13_pbrand_offset[pv]);
        printf("  ");
        printf("p_type: ");
        stringPrint ( iatt14_ptype_char, oatt14_ptype_offset[pv]);
        printf("  ");
        printf("p_size: ");
        printf("%8i", oatt15_psize[pv]);
        printf("  ");
        printf("supp_count: ");
        printf("%8i", oatt2_suppcoun[pv]);
        printf("  ");
        printf("\n");
    }
    if((nout_result > 10)) {
        printf("[...]\n");
    }
    printf("\n");
    std::clock_t stop_finish144 = std::clock();

    printf("<timing>\n");
    printf ( "%32s: %6.1f ms\n", "krnl_supplier1", (stop_krnl_supplier1139 - start_krnl_supplier1139) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_part3", (stop_krnl_part3140 - start_krnl_part3140) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_partsupp5", (stop_krnl_partsupp5141 - start_krnl_partsupp5141) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation8", (stop_krnl_aggregation8142 - start_krnl_aggregation8142) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation9", (stop_krnl_aggregation9143 - start_krnl_aggregation9143) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "finish", (stop_finish144 - start_finish144) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "totalKernelTime", (stop_totalKernelTime138 - start_totalKernelTime138) / (double) (CLOCKS_PER_SEC / 1000) );
    printf("</timing>\n");
}
