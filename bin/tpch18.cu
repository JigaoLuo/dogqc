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
    int att3_ccustkey;
    str_t att4_cname;
};
struct apayl3 {
    int att11_lorderke;
};
struct jpayl6 {
    int att11_lorderke;
};
struct jpayl9 {
    int att3_ccustkey;
    str_t att4_cname;
    int att27_oorderke;
    float att30_ototalpr;
    unsigned att31_oorderda;
};
struct apayl10 {
    str_t att4_cname;
    int att3_ccustkey;
    int att27_oorderke;
    unsigned att31_oorderda;
    float att30_ototalpr;
};

__global__ void krnl_customer1(
    int* iatt3_ccustkey, size_t* iatt4_cname_offset, char* iatt4_cname_char, unique_ht<jpayl7>* jht7) {
    int att3_ccustkey;
    str_t att4_cname;

    int tid_customer1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_customer1 = loopVar;
        active = (loopVar < 150000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att3_ccustkey = iatt3_ccustkey[tid_customer1];
            att4_cname = stringScan ( iatt4_cname_offset, iatt4_cname_char, tid_customer1);
        }
        // -------- hash join build (opId: 7) --------
        if(active) {
            jpayl7 payl7;
            payl7.att3_ccustkey = att3_ccustkey;
            payl7.att4_cname = att4_cname;
            uint64_t hash7;
            hash7 = 0;
            if(active) {
                hash7 = hash ( (hash7 + ((uint64_t)att3_ccustkey)));
            }
            hashBuildUnique ( jht7, 300000, hash7, &(payl7));
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem2(
    int* iatt11_lorderke, int* iatt15_lquantit, agg_ht<apayl3>* aht3, float* agg1) {
    int att11_lorderke;
    int att15_lquantit;

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
            att11_lorderke = iatt11_lorderke[tid_lineitem1];
            att15_lquantit = iatt15_lquantit[tid_lineitem1];
        }
        // -------- aggregation (opId: 3) --------
        int bucket = 0;
        if(active) {
            uint64_t hash3 = 0;
            hash3 = 0;
            if(active) {
                hash3 = hash ( (hash3 + ((uint64_t)att11_lorderke)));
            }
            apayl3 payl;
            payl.att11_lorderke = att11_lorderke;
            int bucketFound = 0;
            int numLookups = 0;
            while(!(bucketFound)) {
                bucket = hashAggregateGetBucket ( aht3, 50000000, hash3, numLookups, &(payl));
                apayl3 probepayl = aht3[bucket].payload;
                bucketFound = 1;
                bucketFound &= ((payl.att11_lorderke == probepayl.att11_lorderke));
            }
        }
        if(active) {
            atomicAdd(&(agg1[bucket]), ((float)att15_lquantit));
        }
        loopVar += step;
    }

}

__global__ void krnl_aggregation3(
    agg_ht<apayl3>* aht3, float* agg1, agg_ht<jpayl6>* jht6) {
    int att11_lorderke;
    float att1_sumqty;

    int tid_aggregation3 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation3 = loopVar;
        active = (loopVar < 50000000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 3) --------
        if(active) {
            active &= ((aht3[tid_aggregation3].lock.lock == OnceLock::LOCK_DONE));
        }
        if(active) {
            apayl3 payl = aht3[tid_aggregation3].payload;
            att11_lorderke = payl.att11_lorderke;
        }
        if(active) {
            att1_sumqty = agg1[tid_aggregation3];
        }
        // -------- selection (opId: 4) --------
        if(active) {
            active = (att1_sumqty > 300.0f);
        }
        // -------- hash join build (opId: 6) --------
        if(active) {
            uint64_t hash6;
            hash6 = 0;
            if(active) {
                hash6 = hash ( (hash6 + ((uint64_t)att11_lorderke)));
            }
            int bucket = 0;
            jpayl6 payl6;
            payl6.att11_lorderke = att11_lorderke;
            int bucketFound = 0;
            int numLookups = 0;
            while(!(bucketFound)) {
                bucket = hashAggregateGetBucket ( jht6, 25000000, hash6, numLookups, &(payl6));
                jpayl6 probepayl = jht6[bucket].payload;
                bucketFound = 1;
                bucketFound &= ((payl6.att11_lorderke == probepayl.att11_lorderke));
            }
        }
        loopVar += step;
    }

}

__global__ void krnl_orders5(
    int* iatt27_oorderke, int* iatt28_ocustkey, float* iatt30_ototalpr, unsigned* iatt31_oorderda, agg_ht<jpayl6>* jht6, unique_ht<jpayl7>* jht7, unique_ht<jpayl9>* jht9) {
    int att27_oorderke;
    int att28_ocustkey;
    float att30_ototalpr;
    unsigned att31_oorderda;
    int att11_lorderke;
    int att3_ccustkey;
    str_t att4_cname;

    int tid_orders1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_orders1 = loopVar;
        active = (loopVar < 1500000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att27_oorderke = iatt27_oorderke[tid_orders1];
            att28_ocustkey = iatt28_ocustkey[tid_orders1];
            att30_ototalpr = iatt30_ototalpr[tid_orders1];
            att31_oorderda = iatt31_oorderda[tid_orders1];
        }
        // -------- hash join probe (opId: 6) --------
        if(active) {
            uint64_t hash6 = 0;
            hash6 = 0;
            if(active) {
                hash6 = hash ( (hash6 + ((uint64_t)att27_oorderke)));
            }
            int numLookups6 = 0;
            int location6 = 0;
            int filterMatch6 = 0;
            int activeProbe6 = 1;
            while((!(filterMatch6) && activeProbe6)) {
                activeProbe6 = hashAggregateFindBucket ( jht6, 25000000, hash6, numLookups6, location6);
                if(activeProbe6) {
                    jpayl6 probepayl = jht6[location6].payload;
                    att11_lorderke = probepayl.att11_lorderke;
                    filterMatch6 = 1;
                    filterMatch6 &= ((att11_lorderke == att27_oorderke));
                }
            }
            active &= (filterMatch6);
        }
        // -------- hash join probe (opId: 7) --------
        uint64_t hash7 = 0;
        if(active) {
            hash7 = 0;
            if(active) {
                hash7 = hash ( (hash7 + ((uint64_t)att28_ocustkey)));
            }
        }
        jpayl7* probepayl7;
        int numLookups7 = 0;
        if(active) {
            active = hashProbeUnique ( jht7, 300000, hash7, numLookups7, &(probepayl7));
        }
        int bucketFound7 = 0;
        int probeActive7 = active;
        while((probeActive7 && !(bucketFound7))) {
            jpayl7 jprobepayl7 = *(probepayl7);
            att3_ccustkey = jprobepayl7.att3_ccustkey;
            att4_cname = jprobepayl7.att4_cname;
            bucketFound7 = 1;
            bucketFound7 &= ((att3_ccustkey == att28_ocustkey));
            if(!(bucketFound7)) {
                probeActive7 = hashProbeUnique ( jht7, 300000, hash7, numLookups7, &(probepayl7));
            }
        }
        active = bucketFound7;
        // -------- hash join build (opId: 9) --------
        if(active) {
            jpayl9 payl9;
            payl9.att3_ccustkey = att3_ccustkey;
            payl9.att4_cname = att4_cname;
            payl9.att27_oorderke = att27_oorderke;
            payl9.att30_ototalpr = att30_ototalpr;
            payl9.att31_oorderda = att31_oorderda;
            uint64_t hash9;
            hash9 = 0;
            if(active) {
                hash9 = hash ( (hash9 + ((uint64_t)att27_oorderke)));
            }
            hashBuildUnique ( jht9, 6250000, hash9, &(payl9));
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem28(
    int* iatt36_lorderke, int* iatt40_lquantit, unique_ht<jpayl9>* jht9, agg_ht<apayl10>* aht10, float* agg2) {
    int att36_lorderke;
    int att40_lquantit;
    int att3_ccustkey;
    str_t att4_cname;
    int att27_oorderke;
    float att30_ototalpr;
    unsigned att31_oorderda;

    int tid_lineitem2 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_lineitem2 = loopVar;
        active = (loopVar < 6001215);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att36_lorderke = iatt36_lorderke[tid_lineitem2];
            att40_lquantit = iatt40_lquantit[tid_lineitem2];
        }
        // -------- hash join probe (opId: 9) --------
        uint64_t hash9 = 0;
        if(active) {
            hash9 = 0;
            if(active) {
                hash9 = hash ( (hash9 + ((uint64_t)att36_lorderke)));
            }
        }
        jpayl9* probepayl9;
        int numLookups9 = 0;
        if(active) {
            active = hashProbeUnique ( jht9, 6250000, hash9, numLookups9, &(probepayl9));
        }
        int bucketFound9 = 0;
        int probeActive9 = active;
        while((probeActive9 && !(bucketFound9))) {
            jpayl9 jprobepayl9 = *(probepayl9);
            att3_ccustkey = jprobepayl9.att3_ccustkey;
            att4_cname = jprobepayl9.att4_cname;
            att27_oorderke = jprobepayl9.att27_oorderke;
            att30_ototalpr = jprobepayl9.att30_ototalpr;
            att31_oorderda = jprobepayl9.att31_oorderda;
            bucketFound9 = 1;
            bucketFound9 &= ((att27_oorderke == att36_lorderke));
            if(!(bucketFound9)) {
                probeActive9 = hashProbeUnique ( jht9, 6250000, hash9, numLookups9, &(probepayl9));
            }
        }
        active = bucketFound9;
        // -------- aggregation (opId: 10) --------
        int bucket = 0;
        if(active) {
            uint64_t hash10 = 0;
            hash10 = 0;
            hash10 = hash ( (hash10 + stringHash ( att4_cname)));
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att3_ccustkey)));
            }
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att27_oorderke)));
            }
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att31_oorderda)));
            }
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att30_ototalpr)));
            }
            apayl10 payl;
            payl.att4_cname = att4_cname;
            payl.att3_ccustkey = att3_ccustkey;
            payl.att27_oorderke = att27_oorderke;
            payl.att31_oorderda = att31_oorderda;
            payl.att30_ototalpr = att30_ototalpr;
            int bucketFound = 0;
            int numLookups = 0;
            while(!(bucketFound)) {
                bucket = hashAggregateGetBucket ( aht10, 5000000, hash10, numLookups, &(payl));
                apayl10 probepayl = aht10[bucket].payload;
                bucketFound = 1;
                bucketFound &= (stringEquals ( payl.att4_cname, probepayl.att4_cname));
                bucketFound &= ((payl.att3_ccustkey == probepayl.att3_ccustkey));
                bucketFound &= ((payl.att27_oorderke == probepayl.att27_oorderke));
                bucketFound &= ((payl.att31_oorderda == probepayl.att31_oorderda));
                bucketFound &= ((payl.att30_ototalpr == probepayl.att30_ototalpr));
            }
        }
        if(active) {
            atomicAdd(&(agg2[bucket]), ((float)att40_lquantit));
        }
        loopVar += step;
    }

}

__global__ void krnl_aggregation10(
    agg_ht<apayl10>* aht10, float* agg2, int* nout_result, str_offs* oatt4_cname_offset, char* iatt4_cname_char, int* oatt3_ccustkey, int* oatt27_oorderke, unsigned* oatt31_oorderda, float* oatt30_ototalpr, float* oatt2_sumqty) {
    str_t att4_cname;
    int att3_ccustkey;
    int att27_oorderke;
    unsigned att31_oorderda;
    float att30_ototalpr;
    float att2_sumqty;
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));

    int tid_aggregation10 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation10 = loopVar;
        active = (loopVar < 5000000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 10) --------
        if(active) {
            active &= ((aht10[tid_aggregation10].lock.lock == OnceLock::LOCK_DONE));
        }
        if(active) {
            apayl10 payl = aht10[tid_aggregation10].payload;
            att4_cname = payl.att4_cname;
            att3_ccustkey = payl.att3_ccustkey;
            att27_oorderke = payl.att27_oorderke;
            att31_oorderda = payl.att31_oorderda;
            att30_ototalpr = payl.att30_ototalpr;
        }
        if(active) {
            att2_sumqty = agg2[tid_aggregation10];
        }
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
            oatt4_cname_offset[wp] = toStringOffset ( iatt4_cname_char, att4_cname);
            oatt3_ccustkey[wp] = att3_ccustkey;
            oatt27_oorderke[wp] = att27_oorderke;
            oatt31_oorderda[wp] = att31_oorderda;
            oatt30_ototalpr[wp] = att30_ototalpr;
            oatt2_sumqty[wp] = att2_sumqty;
        }
        loopVar += step;
    }

}

int main() {
    int* iatt3_ccustkey;
    iatt3_ccustkey = ( int*) map_memory_file ( "mmdb/customer_c_custkey" );
    size_t* iatt4_cname_offset;
    iatt4_cname_offset = ( size_t*) map_memory_file ( "mmdb/customer_c_name_offset" );
    char* iatt4_cname_char;
    iatt4_cname_char = ( char*) map_memory_file ( "mmdb/customer_c_name_char" );
    int* iatt11_lorderke;
    iatt11_lorderke = ( int*) map_memory_file ( "mmdb/lineitem_l_orderkey" );
    int* iatt15_lquantit;
    iatt15_lquantit = ( int*) map_memory_file ( "mmdb/lineitem_l_quantity" );
    int* iatt27_oorderke;
    iatt27_oorderke = ( int*) map_memory_file ( "mmdb/orders_o_orderkey" );
    int* iatt28_ocustkey;
    iatt28_ocustkey = ( int*) map_memory_file ( "mmdb/orders_o_custkey" );
    float* iatt30_ototalpr;
    iatt30_ototalpr = ( float*) map_memory_file ( "mmdb/orders_o_totalprice" );
    unsigned* iatt31_oorderda;
    iatt31_oorderda = ( unsigned*) map_memory_file ( "mmdb/orders_o_orderdate" );
    int* iatt36_lorderke;
    iatt36_lorderke = ( int*) map_memory_file ( "mmdb/lineitem_l_orderkey" );
    int* iatt40_lquantit;
    iatt40_lquantit = ( int*) map_memory_file ( "mmdb/lineitem_l_quantity" );

    int nout_result;
    std::vector < str_offs > oatt4_cname_offset(2500000);
    std::vector < int > oatt3_ccustkey(2500000);
    std::vector < int > oatt27_oorderke(2500000);
    std::vector < unsigned > oatt31_oorderda(2500000);
    std::vector < float > oatt30_ototalpr(2500000);
    std::vector < float > oatt2_sumqty(2500000);

    // wake up gpu
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in wake up gpu! " << cudaGetErrorString( err ) << std::endl;
            ERROR("wake up gpu")
        }
    }

    int* d_iatt3_ccustkey;
    cudaMalloc((void**) &d_iatt3_ccustkey, 150000* sizeof(int) );
    size_t* d_iatt4_cname_offset;
    cudaMalloc((void**) &d_iatt4_cname_offset, (150000 + 1)* sizeof(size_t) );
    char* d_iatt4_cname_char;
    cudaMalloc((void**) &d_iatt4_cname_char, 2700009* sizeof(char) );
    int* d_iatt11_lorderke;
    cudaMalloc((void**) &d_iatt11_lorderke, 6001215* sizeof(int) );
    int* d_iatt15_lquantit;
    cudaMalloc((void**) &d_iatt15_lquantit, 6001215* sizeof(int) );
    int* d_iatt27_oorderke;
    cudaMalloc((void**) &d_iatt27_oorderke, 1500000* sizeof(int) );
    int* d_iatt28_ocustkey;
    cudaMalloc((void**) &d_iatt28_ocustkey, 1500000* sizeof(int) );
    float* d_iatt30_ototalpr;
    cudaMalloc((void**) &d_iatt30_ototalpr, 1500000* sizeof(float) );
    unsigned* d_iatt31_oorderda;
    cudaMalloc((void**) &d_iatt31_oorderda, 1500000* sizeof(unsigned) );
    int* d_iatt36_lorderke;
    d_iatt36_lorderke = d_iatt11_lorderke;
    int* d_iatt40_lquantit;
    d_iatt40_lquantit = d_iatt15_lquantit;
    int* d_nout_result;
    cudaMalloc((void**) &d_nout_result, 1* sizeof(int) );
    str_offs* d_oatt4_cname_offset;
    cudaMalloc((void**) &d_oatt4_cname_offset, 2500000* sizeof(str_offs) );
    int* d_oatt3_ccustkey;
    cudaMalloc((void**) &d_oatt3_ccustkey, 2500000* sizeof(int) );
    int* d_oatt27_oorderke;
    cudaMalloc((void**) &d_oatt27_oorderke, 2500000* sizeof(int) );
    unsigned* d_oatt31_oorderda;
    cudaMalloc((void**) &d_oatt31_oorderda, 2500000* sizeof(unsigned) );
    float* d_oatt30_ototalpr;
    cudaMalloc((void**) &d_oatt30_ototalpr, 2500000* sizeof(float) );
    float* d_oatt2_sumqty;
    cudaMalloc((void**) &d_oatt2_sumqty, 2500000* sizeof(float) );
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

    unique_ht<jpayl7>* d_jht7;
    cudaMalloc((void**) &d_jht7, 300000* sizeof(unique_ht<jpayl7>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht7, 300000);
    }
    agg_ht<apayl3>* d_aht3;
    cudaMalloc((void**) &d_aht3, 50000000* sizeof(agg_ht<apayl3>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_aht3, 50000000);
    }
    float* d_agg1;
    cudaMalloc((void**) &d_agg1, 50000000* sizeof(float) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg1, 0.0f, 50000000);
    }
    agg_ht<jpayl6>* d_jht6;
    cudaMalloc((void**) &d_jht6, 25000000* sizeof(agg_ht<jpayl6>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_jht6, 25000000);
    }
    unique_ht<jpayl9>* d_jht9;
    cudaMalloc((void**) &d_jht9, 6250000* sizeof(unique_ht<jpayl9>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht9, 6250000);
    }
    agg_ht<apayl10>* d_aht10;
    cudaMalloc((void**) &d_aht10, 5000000* sizeof(agg_ht<apayl10>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_aht10, 5000000);
    }
    float* d_agg2;
    cudaMalloc((void**) &d_agg2, 5000000* sizeof(float) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg2, 0.0f, 5000000);
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

    cudaMemcpy( d_iatt3_ccustkey, iatt3_ccustkey, 150000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt4_cname_offset, iatt4_cname_offset, (150000 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt4_cname_char, iatt4_cname_char, 2700009 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt11_lorderke, iatt11_lorderke, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt15_lquantit, iatt15_lquantit, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt27_oorderke, iatt27_oorderke, 1500000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt28_ocustkey, iatt28_ocustkey, 1500000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt30_ototalpr, iatt30_ototalpr, 1500000 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt31_oorderda, iatt31_oorderda, 1500000 * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy in! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy in")
        }
    }

    std::clock_t start_totalKernelTime153 = std::clock();
    std::clock_t start_krnl_customer1154 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_customer1<<<gridsize, blocksize>>>(d_iatt3_ccustkey, d_iatt4_cname_offset, d_iatt4_cname_char, d_jht7);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_customer1154 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_customer1! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_customer1")
        }
    }

    std::clock_t start_krnl_lineitem2155 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem2<<<gridsize, blocksize>>>(d_iatt11_lorderke, d_iatt15_lquantit, d_aht3, d_agg1);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem2155 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem2! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem2")
        }
    }

    std::clock_t start_krnl_aggregation3156 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation3<<<gridsize, blocksize>>>(d_aht3, d_agg1, d_jht6);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation3156 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation3! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation3")
        }
    }

    std::clock_t start_krnl_orders5157 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_orders5<<<gridsize, blocksize>>>(d_iatt27_oorderke, d_iatt28_ocustkey, d_iatt30_ototalpr, d_iatt31_oorderda, d_jht6, d_jht7, d_jht9);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_orders5157 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_orders5! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_orders5")
        }
    }

    std::clock_t start_krnl_lineitem28158 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem28<<<gridsize, blocksize>>>(d_iatt36_lorderke, d_iatt40_lquantit, d_jht9, d_aht10, d_agg2);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem28158 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem28! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem28")
        }
    }

    std::clock_t start_krnl_aggregation10159 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation10<<<gridsize, blocksize>>>(d_aht10, d_agg2, d_nout_result, d_oatt4_cname_offset, d_iatt4_cname_char, d_oatt3_ccustkey, d_oatt27_oorderke, d_oatt31_oorderda, d_oatt30_ototalpr, d_oatt2_sumqty);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation10159 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation10! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation10")
        }
    }

    std::clock_t stop_totalKernelTime153 = std::clock();
    cudaMemcpy( &nout_result, d_nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt4_cname_offset.data(), d_oatt4_cname_offset, 2500000 * sizeof(str_offs), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt3_ccustkey.data(), d_oatt3_ccustkey, 2500000 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt27_oorderke.data(), d_oatt27_oorderke, 2500000 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt31_oorderda.data(), d_oatt31_oorderda, 2500000 * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt30_ototalpr.data(), d_oatt30_ototalpr, 2500000 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt2_sumqty.data(), d_oatt2_sumqty, 2500000 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy out! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy out")
        }
    }

    cudaFree( d_iatt3_ccustkey);
    cudaFree( d_iatt4_cname_offset);
    cudaFree( d_iatt4_cname_char);
    cudaFree( d_jht7);
    cudaFree( d_iatt11_lorderke);
    cudaFree( d_iatt15_lquantit);
    cudaFree( d_aht3);
    cudaFree( d_agg1);
    cudaFree( d_jht6);
    cudaFree( d_iatt27_oorderke);
    cudaFree( d_iatt28_ocustkey);
    cudaFree( d_iatt30_ototalpr);
    cudaFree( d_iatt31_oorderda);
    cudaFree( d_jht9);
    cudaFree( d_aht10);
    cudaFree( d_agg2);
    cudaFree( d_nout_result);
    cudaFree( d_oatt4_cname_offset);
    cudaFree( d_oatt3_ccustkey);
    cudaFree( d_oatt27_oorderke);
    cudaFree( d_oatt31_oorderda);
    cudaFree( d_oatt30_ototalpr);
    cudaFree( d_oatt2_sumqty);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda free! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda free")
        }
    }

    std::clock_t start_finish160 = std::clock();
    printf("\nResult: %i tuples\n", nout_result);
    if((nout_result > 2500000)) {
        ERROR("Index out of range. Output size larger than allocated with expected result number.")
    }
    for ( int pv = 0; ((pv < 10) && (pv < nout_result)); pv += 1) {
        printf("c_name: ");
        stringPrint ( iatt4_cname_char, oatt4_cname_offset[pv]);
        printf("  ");
        printf("c_custkey: ");
        printf("%8i", oatt3_ccustkey[pv]);
        printf("  ");
        printf("o_orderkey: ");
        printf("%8i", oatt27_oorderke[pv]);
        printf("  ");
        printf("o_orderdate: ");
        printf("%10i", oatt31_oorderda[pv]);
        printf("  ");
        printf("o_totalprice: ");
        printf("%15.2f", oatt30_ototalpr[pv]);
        printf("  ");
        printf("sum_qty: ");
        printf("%15.2f", oatt2_sumqty[pv]);
        printf("  ");
        printf("\n");
    }
    if((nout_result > 10)) {
        printf("[...]\n");
    }
    printf("\n");
    std::clock_t stop_finish160 = std::clock();

    printf("<timing>\n");
    printf ( "%32s: %6.1f ms\n", "krnl_customer1", (stop_krnl_customer1154 - start_krnl_customer1154) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem2", (stop_krnl_lineitem2155 - start_krnl_lineitem2155) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation3", (stop_krnl_aggregation3156 - start_krnl_aggregation3156) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_orders5", (stop_krnl_orders5157 - start_krnl_orders5157) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem28", (stop_krnl_lineitem28158 - start_krnl_lineitem28158) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation10", (stop_krnl_aggregation10159 - start_krnl_aggregation10159) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "finish", (stop_finish160 - start_finish160) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "totalKernelTime", (stop_totalKernelTime153 - start_totalKernelTime153) / (double) (CLOCKS_PER_SEC / 1000) );
    printf("</timing>\n");
}
