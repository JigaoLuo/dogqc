#include <cassert>
#include <list>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <ctime>
#include <limits.h>
#include <float.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../dogqc/include/csv.h"
#include "../dogqc/include/util.h"
#include "../dogqc/include/mappedmalloc.h"
#include "../dogqc/include/util.cuh"
#include "../dogqc/include/hashing.cuh"

struct apayl5 {
    char att17_lreturnf;
    char att18_llinesta;
};
__device__ bool operator==(const apayl5& lhs, const apayl5& rhs) {
    return lhs.att17_lreturnf == rhs.att17_lreturnf && lhs.att18_llinesta == rhs.att18_llinesta;
}

constexpr int SHARED_MEMORY_HT_SIZE = 100;   /// In shared memory
constexpr int LINEITEM_SIZE = 6001215;       /// SF1
//constexpr int LINEITEM_SIZE = 59986052;      /// SF10, change the folder name to sf100

__device__ void sm_to_gm(
    agg_ht_sm<apayl5>* aht5, double* agg1, double* agg2, double* agg3, double* agg4, double* agg5, double* agg6, double* agg7, int* agg8, agg_ht<apayl5>* g_aht5, double* g_agg1, double* g_agg2, double* g_agg3, double* g_agg4, double* g_agg5, double* g_agg6, double* g_agg7, int* g_agg8) {
    char att17_lreturnf;
    char att18_llinesta;
    double att1_sumqty;
    double att2_sumbasep;
    double att3_sumdiscp;
    double att4_sumcharg;
    double att5_avgqty;
    double att6_avgprice;
    double att7_avgdisc;
    int att8_countord;

    int tid_aggregation5 = 0;
    unsigned loopVar = threadIdx.x;
    unsigned step = blockDim.x;
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation5 = loopVar;
        active = (loopVar < SHARED_MEMORY_HT_SIZE);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 5) --------
        if(active) {
            active &= ((aht5[tid_aggregation5].lock.lock == OnceLock::LOCK_DONE));
        }
        if(active) {
            apayl5 payl = aht5[tid_aggregation5].payload;
            att17_lreturnf = payl.att17_lreturnf;
            att18_llinesta = payl.att18_llinesta;
        }
        if(active) {
            att1_sumqty = agg1[tid_aggregation5];
            att2_sumbasep = agg2[tid_aggregation5];
            att3_sumdiscp = agg3[tid_aggregation5];
            att4_sumcharg = agg4[tid_aggregation5];
            att5_avgqty = agg5[tid_aggregation5];
            att6_avgprice = agg6[tid_aggregation5];
            att7_avgdisc = agg7[tid_aggregation5];
            att8_countord = agg8[tid_aggregation5];



        }

        int bucket = 0;
        if(active) {
            uint64_t hash5 = 0;
            hash5 = 0;
            if(active) {
                hash5 = hash ( (hash5 + ((uint64_t)att17_lreturnf)));
            }
            if(active) {
                hash5 = hash ( (hash5 + ((uint64_t)att18_llinesta)));
            }
            apayl5 payl;
            payl.att17_lreturnf = att17_lreturnf;
            payl.att18_llinesta = att18_llinesta;
            int bucketFound = 0;
            int numLookups = 0;
            while(!(bucketFound)) {
                bucket = hashAggregateGetBucket ( g_aht5, 200, hash5, numLookups, &(payl));
                if((bucket != -1)) {
                    apayl5 probepayl = g_aht5[bucket].payload;
                    bucketFound = 1;
                    bucketFound &= ((payl.att17_lreturnf == probepayl.att17_lreturnf));
                    bucketFound &= ((payl.att18_llinesta == probepayl.att18_llinesta));
                }
            }
        }
        if((active && (bucket != -1))) {
            atomicAdd(&(g_agg1[bucket]), ((double)att1_sumqty));
            atomicAdd(&(g_agg2[bucket]), ((double)att2_sumbasep));
            atomicAdd(&(g_agg3[bucket]), ((double)att3_sumdiscp));
            atomicAdd(&(g_agg4[bucket]), ((double)att4_sumcharg));
            atomicAdd(&(g_agg5[bucket]), ((double)att5_avgqty));
            atomicAdd(&(g_agg6[bucket]), ((double)att6_avgprice));
            atomicAdd(&(g_agg7[bucket]), ((double)att7_avgdisc));
            atomicAdd(&(g_agg8[bucket]), ((int)att8_countord));
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem1(
    int* iatt13_lquantit, float* iatt14_lextende, float* iatt15_ldiscoun, float* iatt16_ltax, char* iatt17_lreturnf, char* iatt18_llinesta, unsigned* iatt19_lshipdat, agg_ht<apayl5>* g_aht5, double* g_agg1, double* g_agg2, double* g_agg3, double* g_agg4, double* g_agg5, double* g_agg6, double* g_agg7, int* g_agg8) {
    volatile __shared__ int HT_FULL_FLAG;
    HT_FULL_FLAG = 0;
    extern __shared__ char shared_memory[];
    agg_ht_sm<apayl5>* aht5;
    aht5 = ((agg_ht_sm<apayl5>*)shared_memory);
    initSMAggHT(aht5, SHARED_MEMORY_HT_SIZE);
    double* agg1;
    agg1 = ((double*)(shared_memory + (sizeof (agg_ht_sm<apayl5>) * SHARED_MEMORY_HT_SIZE)));
    initSMAggArray(agg1, SHARED_MEMORY_HT_SIZE);
    double* agg2;
    agg2 = ((double*)((shared_memory + (sizeof (agg_ht_sm<apayl5>) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)));
    initSMAggArray(agg2, SHARED_MEMORY_HT_SIZE);
    double* agg3;
    agg3 = ((double*)(((shared_memory + (sizeof (agg_ht_sm<apayl5>) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)));
    initSMAggArray(agg3, SHARED_MEMORY_HT_SIZE);
    double* agg4;
    agg4 = ((double*)((((shared_memory + (sizeof (agg_ht_sm<apayl5>) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)));
    initSMAggArray(agg4, SHARED_MEMORY_HT_SIZE);
    double* agg5;
    agg5 = ((double*)(((((shared_memory + (sizeof (agg_ht_sm<apayl5>) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)));
    initSMAggArray(agg5, SHARED_MEMORY_HT_SIZE);
    double* agg6;
    agg6 = ((double*)((((((shared_memory + (sizeof (agg_ht_sm<apayl5>) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)));
    initSMAggArray(agg6, SHARED_MEMORY_HT_SIZE);
    double* agg7;
    agg7 = ((double*)(((((((shared_memory + (sizeof (agg_ht_sm<apayl5>) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)));
    initSMAggArray(agg7, SHARED_MEMORY_HT_SIZE);
    int* agg8;
    agg8 = ((int*)((((((((shared_memory + (sizeof (agg_ht_sm<apayl5>) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)) + (sizeof (double) * SHARED_MEMORY_HT_SIZE)));
    initSMAggArray(agg8, SHARED_MEMORY_HT_SIZE);
    __syncthreads();

    int att13_lquantit;
    double att14_lextende;
    double att15_ldiscoun;
    double att16_ltax;
    char att17_lreturnf;
    char att18_llinesta;
    unsigned att19_lshipdat;
    double att25_charge;
    double att26_discpric;

    int tid_lineitem1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_lineitem1 = loopVar;
        active = (loopVar < LINEITEM_SIZE);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att13_lquantit = iatt13_lquantit[tid_lineitem1];
            att14_lextende = iatt14_lextende[tid_lineitem1];
            att15_ldiscoun = iatt15_ldiscoun[tid_lineitem1];
            att16_ltax = iatt16_ltax[tid_lineitem1];
            att17_lreturnf = iatt17_lreturnf[tid_lineitem1];
            att18_llinesta = iatt18_llinesta[tid_lineitem1];
            att19_lshipdat = iatt19_lshipdat[tid_lineitem1];
        }
        // -------- selection (opId: 2) --------
        if(active) {
            active = (att19_lshipdat <= 19980902);
        }
        // -------- map (opId: 3) --------
        if(active) {
            att25_charge = ((att14_lextende * ((float)1.0f - att15_ldiscoun)) * ((float)1.0f + att16_ltax));
        }
        // -------- map (opId: 4) --------
        if(active) {
            att26_discpric = (att14_lextende * ((float)1.0f - att15_ldiscoun));
        }
        // -------- aggregation (opId: 5) --------
        int bucket = 0;
        if(active) {
            uint64_t hash5 = 0;
            hash5 = 0;
            if(active) {
                hash5 = hash ( (hash5 + ((uint64_t)att17_lreturnf)));
            }
            if(active) {
                hash5 = hash ( (hash5 + ((uint64_t)att18_llinesta)));
            }
            apayl5 payl;
            payl.att17_lreturnf = att17_lreturnf;
            payl.att18_llinesta = att18_llinesta;
            int bucketFound = 0;
            int numLookups = 0;
            while(!(bucketFound)) {
                bucket = hashAggregateGetBucket ( aht5, SHARED_MEMORY_HT_SIZE, hash5, numLookups, &(payl));
                if((bucket != -1)) {
                    apayl5 probepayl = aht5[bucket].payload;
                    bucketFound = 1;
                    bucketFound &= ((payl.att17_lreturnf == probepayl.att17_lreturnf));
                    bucketFound &= ((payl.att18_llinesta == probepayl.att18_llinesta));
                }
                else {
                    assert((bucketFound == 0));
                    loopVar -= step;
                    atomicAdd(((int*)&(HT_FULL_FLAG)), 1);
                    break;
                }
            }
        }


        unsigned int active_mask = __match_any_sync(__activemask(), (active && (bucket != -1)));
        cg::thread_block cta = cg::this_thread_block();
        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
        unsigned int group_mask_cg = tile32.match_any(bucket);
        unsigned int group_mask = __match_any_sync(active_mask, bucket);
        if((active && (bucket != -1))) {
            // Get a mask for a hash group with the same bucket.
            // Find the lowest-numbered active lane: the leader mask.
            int elected_lane = __ffs(group_mask) - 1;
            // A warp-wise aggregation into the elected_lane.
//            double att13_lquantit_warp_agg = att13_lquantit;
//            double att13_lquantit_warp_agg_ = 0;
//            for (unsigned int mask = group_mask, current_lane = 0; mask != 0; mask >>= 1, ++current_lane) {
//                if (mask & 1) {
//                    att13_lquantit_warp_agg_  += __shfl_sync(group_mask, att13_lquantit, current_lane);
//                }
//            }

////            unsigned laneid; asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneid));
//            cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
//            assert (group_mask == group_mask_cg);
//            cooperative_groups::coalesced_group subtile = cooperative_groups::labeled_partition(g, group_mask);
//            int s = subtile.size();
//            int r = subtile.thread_rank();
//            for (int offset = 32 / 2; offset > 0; offset /= 2) {
//                auto att13_lquantit_shfl = subtile.shfl_down(att13_lquantit, offset);
//                auto att14_lextende_shfl = subtile.shfl_down(att14_lextende, offset);
//                auto att26_discpric_shfl = subtile.shfl_down(att26_discpric, offset);
//                auto att25_charge_shfl   = subtile.shfl_down(att25_charge, offset);
//                auto att15_ldiscoun_shfl = subtile.shfl_down(att15_ldiscoun, offset);
//                if (r + offset < s) {
//                    att13_lquantit += att13_lquantit_shfl;
//                    att14_lextende += att14_lextende_shfl;initSMAggArray
//                    att26_discpric += att26_discpric_shfl;
//                    att25_charge   += att25_charge_shfl;
//                    att15_ldiscoun += att15_ldiscoun_shfl;
//                }
//            }
//            if (subtile.thread_rank() == 0 /*leader lane*/) {
////            if (laneid == elected_lane) {
//                atomicAdd(&(agg1[bucket]), ((double)att13_lquantit));
////                atomicAdd(&(agg2[bucket]), ((double)att14_lextende));
////                atomicAdd(&(agg3[bucket]), ((double)att26_discpric))initSMAggArray;
////                atomicAdd(&(agg4[bucket]), ((double)att25_charge));
////                atomicAdd(&(agg5[bucket]), ((double)att13_lquantit));
////                atomicAdd(&(agg6[bucket]), ((double)att14_lextende));
////                atomicAdd(&(agg7[bucket]), ((double)att15_ldiscoun));
////                atomicAdd(&(agg8[bucket]), ((int) subtile.size() ));
//            }
        }

        if(active && bucket != -1) {
            atomicAdd(&(agg8[bucket]), ((int)1));
            atomicAdd(&(agg1[bucket]), ((double)att13_lquantit));
            atomicAdd(&(agg2[bucket]), ((double)att14_lextende));
            atomicAdd(&(agg3[bucket]), ((double)att26_discpric));
            atomicAdd(&(agg4[bucket]), ((double)att25_charge));
            atomicAdd(&(agg5[bucket]), ((double)att13_lquantit));
            atomicAdd(&(agg6[bucket]), ((double)att14_lextende));
            atomicAdd(&(agg7[bucket]), ((double)att15_ldiscoun));
        }

        __syncthreads();
        if((HT_FULL_FLAG != 0)) {
            sm_to_gm ( aht5, agg1, agg2, agg3, agg4, agg5, agg6, agg7, agg8, g_aht5, g_agg1, g_agg2, g_agg3, g_agg4, g_agg5, g_agg6, g_agg7, g_agg8);
            __threadfence_block();
            initSMAggHT(aht5, SHARED_MEMORY_HT_SIZE);
            initSMAggArray(agg1, SHARED_MEMORY_HT_SIZE);
            initSMAggArray(agg2, SHARED_MEMORY_HT_SIZE);
            initSMAggArray(agg3, SHARED_MEMORY_HT_SIZE);
            initSMAggArray(agg4, SHARED_MEMORY_HT_SIZE);
            initSMAggArray(agg5, SHARED_MEMORY_HT_SIZE);
            initSMAggArray(agg6, SHARED_MEMORY_HT_SIZE);
            initSMAggArray(agg7, SHARED_MEMORY_HT_SIZE);
            initSMAggArray(agg8, SHARED_MEMORY_HT_SIZE);
            if((threadIdx.x == 0)) {
                HT_FULL_FLAG = 0;
            }
            __syncthreads();
        }
        loopVar += step;
    }

    __syncthreads();
    sm_to_gm ( aht5, agg1, agg2, agg3, agg4, agg5, agg6, agg7, agg8, g_aht5, g_agg1, g_agg2, g_agg3, g_agg4, g_agg5, g_agg6, g_agg7, g_agg8);
}

__global__ void krnl_aggregation5(
    agg_ht<apayl5>* aht5, double* agg1, double* agg2, double* agg3, double* agg4, double* agg5, double* agg6, double* agg7, int* agg8, int* nout_result, char* oatt17_lreturnf, char* oatt18_llinesta, double* oatt1_sumqty, double* oatt2_sumbasep, double* oatt3_sumdiscp, double* oatt5_avgqty, double* oatt6_avgprice, double* oatt7_avgdisc, int* oatt8_countord) {
    char att17_lreturnf;
    char att18_llinesta;
    double att1_sumqty;
    double att2_sumbasep;
    double att3_sumdiscp;
    double att4_sumcharg;
    double att5_avgqty;
    double att6_avgprice;
    double att7_avgdisc;
    int att8_countord;
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));

    int tid_aggregation5 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation5 = loopVar;
        active = (loopVar < 200);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 5) --------
        if(active) {
            active &= ((aht5[tid_aggregation5].lock.lock == OnceLock::LOCK_DONE));
        }
        if(active) {
            apayl5 payl = aht5[tid_aggregation5].payload;
            att17_lreturnf = payl.att17_lreturnf;
            att18_llinesta = payl.att18_llinesta;
        }
        if(active) {
            att1_sumqty = agg1[tid_aggregation5];
            att2_sumbasep = agg2[tid_aggregation5];
            att3_sumdiscp = agg3[tid_aggregation5];
            att4_sumcharg = agg4[tid_aggregation5];
            att5_avgqty = agg5[tid_aggregation5];
            att6_avgprice = agg6[tid_aggregation5];
            att7_avgdisc = agg7[tid_aggregation5];
            att8_countord = agg8[tid_aggregation5];
            att5_avgqty = (att5_avgqty / ((double)att8_countord));
            att6_avgprice = (att6_avgprice / ((double)att8_countord));
            att7_avgdisc = (att7_avgdisc / ((double)att8_countord));
        }
        // -------- projection (no code) (opId: 6) --------
        // -------- materialize (opId: 7) --------
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
            oatt17_lreturnf[wp] = att17_lreturnf;
            oatt18_llinesta[wp] = att18_llinesta;
            oatt1_sumqty[wp] = att1_sumqty;
            oatt2_sumbasep[wp] = att2_sumbasep;
            oatt3_sumdiscp[wp] = att3_sumdiscp;
            oatt5_avgqty[wp] = att5_avgqty;
            oatt6_avgprice[wp] = att6_avgprice;
            oatt7_avgdisc[wp] = att7_avgdisc;
            oatt8_countord[wp] = att8_countord;
        }
        loopVar += step;
    }

}

int main() {
    int* iatt13_lquantit;
    iatt13_lquantit = ( int*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_quantity" );
    float* iatt14_lextende;
    iatt14_lextende = ( float*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_extendedprice" );
    float* iatt15_ldiscoun;
    iatt15_ldiscoun = ( float*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_discount" );
    float* iatt16_ltax;
    iatt16_ltax = ( float*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_tax" );
    char* iatt17_lreturnf;
    iatt17_lreturnf = ( char*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_returnflag" );
    char* iatt18_llinesta;
    iatt18_llinesta = ( char*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_linestatus" );
    unsigned* iatt19_lshipdat;
    iatt19_lshipdat = ( unsigned*) map_memory_file ( "mmdb/tpch-dbgen-sf1/lineitem_l_shipdate" );

    int nout_result;
    std::vector < char > oatt17_lreturnf(100);
    std::vector < char > oatt18_llinesta(100);
    std::vector < double > oatt1_sumqty(100);
    std::vector < double > oatt2_sumbasep(100);
    std::vector < double > oatt3_sumdiscp(100);
    std::vector < double > oatt5_avgqty(100);
    std::vector < double > oatt6_avgprice(100);
    std::vector < double > oatt7_avgdisc(100);
    std::vector < int > oatt8_countord(100);

    // wake up gpu
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in wake up gpu! " << cudaGetErrorString( err ) << std::endl;
            ERROR("wake up gpu")
        }
    }

    int* d_iatt13_lquantit;
    cudaMalloc((void**) &d_iatt13_lquantit, LINEITEM_SIZE* sizeof(int) );
    float* d_iatt14_lextende;
    cudaMalloc((void**) &d_iatt14_lextende, LINEITEM_SIZE* sizeof(float) );
    float* d_iatt15_ldiscoun;
    cudaMalloc((void**) &d_iatt15_ldiscoun, LINEITEM_SIZE* sizeof(float) );
    float* d_iatt16_ltax;
    cudaMalloc((void**) &d_iatt16_ltax, LINEITEM_SIZE* sizeof(float) );
    char* d_iatt17_lreturnf;
    cudaMalloc((void**) &d_iatt17_lreturnf, LINEITEM_SIZE* sizeof(char) );
    char* d_iatt18_llinesta;
    cudaMalloc((void**) &d_iatt18_llinesta, LINEITEM_SIZE* sizeof(char) );
    unsigned* d_iatt19_lshipdat;
    cudaMalloc((void**) &d_iatt19_lshipdat, LINEITEM_SIZE* sizeof(unsigned) );
    int* d_nout_result;
    cudaMalloc((void**) &d_nout_result, 1* sizeof(int) );
    char* d_oatt17_lreturnf;
    cudaMalloc((void**) &d_oatt17_lreturnf, 100* sizeof(char) );
    char* d_oatt18_llinesta;
    cudaMalloc((void**) &d_oatt18_llinesta, 100* sizeof(char) );
    double* d_oatt1_sumqty;
    cudaMalloc((void**) &d_oatt1_sumqty, 100* sizeof(double) );
    double* d_oatt2_sumbasep;
    cudaMalloc((void**) &d_oatt2_sumbasep, 100* sizeof(double) );
    double* d_oatt3_sumdiscp;
    cudaMalloc((void**) &d_oatt3_sumdiscp, 100* sizeof(double) );
    double* d_oatt5_avgqty;
    cudaMalloc((void**) &d_oatt5_avgqty, 100* sizeof(double) );
    double* d_oatt6_avgprice;
    cudaMalloc((void**) &d_oatt6_avgprice, 100* sizeof(double) );
    double* d_oatt7_avgdisc;
    cudaMalloc((void**) &d_oatt7_avgdisc, 100* sizeof(double) );
    int* d_oatt8_countord;
    cudaMalloc((void**) &d_oatt8_countord, 100* sizeof(int) );
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

    agg_ht<apayl5>* d_aht5;
    cudaMalloc((void**) &d_aht5, 200* sizeof(agg_ht<apayl5>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_aht5, 200);
    }
    double* d_agg1;
    cudaMalloc((void**) &d_agg1, 200* sizeof(double) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg1, 0.0, 200);
    }
    double* d_agg2;
    cudaMalloc((void**) &d_agg2, 200* sizeof(double) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg2, 0.0, 200);
    }
    double* d_agg3;
    cudaMalloc((void**) &d_agg3, 200* sizeof(double) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg3, 0.0, 200);
    }
    double* d_agg4;
    cudaMalloc((void**) &d_agg4, 200* sizeof(double) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg4, 0.0, 200);
    }
    double* d_agg5;
    cudaMalloc((void**) &d_agg5, 200* sizeof(double) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg5, 0.0, 200);
    }
    double* d_agg6;
    cudaMalloc((void**) &d_agg6, 200* sizeof(double) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg6, 0.0, 200);
    }
    double* d_agg7;
    cudaMalloc((void**) &d_agg7, 200* sizeof(double) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg7, 0.0, 200);
    }
    int* d_agg8;
    cudaMalloc((void**) &d_agg8, 200* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg8, 0, 200);
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

    cudaMemcpy( d_iatt13_lquantit, iatt13_lquantit, LINEITEM_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt14_lextende, iatt14_lextende, LINEITEM_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt15_ldiscoun, iatt15_ldiscoun, LINEITEM_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt16_ltax, iatt16_ltax, LINEITEM_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt17_lreturnf, iatt17_lreturnf, LINEITEM_SIZE * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt18_llinesta, iatt18_llinesta, LINEITEM_SIZE * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt19_lshipdat, iatt19_lshipdat, LINEITEM_SIZE * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy in! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy in")
        }
    }

    int shared_memory_usage = (((((((((sizeof (agg_ht_sm<apayl5>) + sizeof (double)) + sizeof (double)) + sizeof (double)) + sizeof (double)) + sizeof (double)) + sizeof (double)) + sizeof (double)) + sizeof (int)) * SHARED_MEMORY_HT_SIZE);
    std::cout << "Shared memory usage: " << shared_memory_usage << " bytes" << std::endl;
    cudaFuncSetAttribute ( krnl_lineitem1, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_usage);
    std::clock_t start_totalKernelTime0 = std::clock();
    std::clock_t start_krnl_lineitem11 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem1<<<gridsize, blocksize,shared_memory_usage>>>(d_iatt13_lquantit, d_iatt14_lextende, d_iatt15_ldiscoun, d_iatt16_ltax, d_iatt17_lreturnf, d_iatt18_llinesta, d_iatt19_lshipdat, d_aht5, d_agg1, d_agg2, d_agg3, d_agg4, d_agg5, d_agg6, d_agg7, d_agg8);
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

    std::clock_t start_krnl_aggregation52 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation5<<<gridsize, blocksize>>>(d_aht5, d_agg1, d_agg2, d_agg3, d_agg4, d_agg5, d_agg6, d_agg7, d_agg8, d_nout_result, d_oatt17_lreturnf, d_oatt18_llinesta, d_oatt1_sumqty, d_oatt2_sumbasep, d_oatt3_sumdiscp, d_oatt5_avgqty, d_oatt6_avgprice, d_oatt7_avgdisc, d_oatt8_countord);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation52 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation5! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation5")
        }
    }

    std::clock_t stop_totalKernelTime0 = std::clock();
    cudaMemcpy( &nout_result, d_nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt17_lreturnf.data(), d_oatt17_lreturnf, 100 * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt18_llinesta.data(), d_oatt18_llinesta, 100 * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt1_sumqty.data(), d_oatt1_sumqty, 100 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt2_sumbasep.data(), d_oatt2_sumbasep, 100 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt3_sumdiscp.data(), d_oatt3_sumdiscp, 100 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt5_avgqty.data(), d_oatt5_avgqty, 100 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt6_avgprice.data(), d_oatt6_avgprice, 100 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt7_avgdisc.data(), d_oatt7_avgdisc, 100 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt8_countord.data(), d_oatt8_countord, 100 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy out! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy out")
        }
    }

    cudaFree( d_iatt13_lquantit);
    cudaFree( d_iatt14_lextende);
    cudaFree( d_iatt15_ldiscoun);
    cudaFree( d_iatt16_ltax);
    cudaFree( d_iatt17_lreturnf);
    cudaFree( d_iatt18_llinesta);
    cudaFree( d_iatt19_lshipdat);
    cudaFree( d_aht5);
    cudaFree( d_agg1);
    cudaFree( d_agg2);
    cudaFree( d_agg3);
    cudaFree( d_agg4);
    cudaFree( d_agg5);
    cudaFree( d_agg6);
    cudaFree( d_agg7);
    cudaFree( d_agg8);
    cudaFree( d_nout_result);
    cudaFree( d_oatt17_lreturnf);
    cudaFree( d_oatt18_llinesta);
    cudaFree( d_oatt1_sumqty);
    cudaFree( d_oatt2_sumbasep);
    cudaFree( d_oatt3_sumdiscp);
    cudaFree( d_oatt5_avgqty);
    cudaFree( d_oatt6_avgprice);
    cudaFree( d_oatt7_avgdisc);
    cudaFree( d_oatt8_countord);
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
    if((nout_result > 100)) {
        ERROR("Index out of range. Output size larger than allocated with expected result number.")
    }
    for ( int pv = 0; ((pv < 10) && (pv < nout_result)); pv += 1) {
        printf("l_returnflag: ");
        printf("%c", oatt17_lreturnf[pv]);
        printf("  ");
        printf("l_linestatus: ");
        printf("%c", oatt18_llinesta[pv]);
        printf("  ");
        printf("sum_qty: ");
        printf("%15.2f", oatt1_sumqty[pv]);
        printf("  ");
        printf("sum_base_price: ");
        printf("%15.2f", oatt2_sumbasep[pv]);
        printf("  ");
        printf("sum_disc_price: ");
        printf("%15.2f", oatt3_sumdiscp[pv]);
        printf("  ");
        printf("avg_qty: ");
        printf("%15.2f", oatt5_avgqty[pv]);
        printf("  ");
        printf("avg_price: ");
        printf("%15.2f", oatt6_avgprice[pv]);
        printf("  ");
        printf("avg_disc: ");
        printf("%15.2f", oatt7_avgdisc[pv]);
        printf("  ");
        printf("count_order: ");
        printf("%8i", oatt8_countord[pv]);
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
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation5", (stop_krnl_aggregation52 - start_krnl_aggregation52) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "finish", (stop_finish3 - start_finish3) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "totalKernelTime", (stop_totalKernelTime0 - start_totalKernelTime0) / (double) (CLOCKS_PER_SEC / 1000) );
    printf("</timing>\n");
}
