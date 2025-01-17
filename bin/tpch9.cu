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
struct jpayl3 {
    int att2_nnationk;
    str_t att3_nname;
};
struct jpayl8 {
    str_t att3_nname;
    int att6_ssuppkey;
};
struct jpayl7 {
    int att13_ppartkey;
};
struct jpayl10 {
    str_t att3_nname;
    int att6_ssuppkey;
    int att13_ppartkey;
    float att25_pssupply;
};
struct jpayl13 {
    str_t att3_nname;
    int att27_lorderke;
    float att43_amount;
};
struct apayl15 {
    str_t att3_nname;
    unsigned att53_oyear;
};

__global__ void krnl_nation1(
    int* iatt2_nnationk, size_t* iatt3_nname_offset, char* iatt3_nname_char, unique_ht<jpayl3>* jht3) {
    int att2_nnationk;
    str_t att3_nname;

    int tid_nation1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_nation1 = loopVar;
        active = (loopVar < 25);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att2_nnationk = iatt2_nnationk[tid_nation1];
            att3_nname = stringScan ( iatt3_nname_offset, iatt3_nname_char, tid_nation1);
        }
        // -------- hash join build (opId: 3) --------
        if(active) {
            jpayl3 payl3;
            payl3.att2_nnationk = att2_nnationk;
            payl3.att3_nname = att3_nname;
            uint64_t hash3;
            hash3 = 0;
            if(active) {
                hash3 = hash ( (hash3 + ((uint64_t)att2_nnationk)));
            }
            hashBuildUnique ( jht3, 50, hash3, &(payl3));
        }
        loopVar += step;
    }

}

__global__ void krnl_supplier2(
    int* iatt6_ssuppkey, int* iatt9_snationk, unique_ht<jpayl3>* jht3, unique_ht<jpayl8>* jht8) {
    int att6_ssuppkey;
    int att9_snationk;
    int att2_nnationk;
    str_t att3_nname;

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
            att6_ssuppkey = iatt6_ssuppkey[tid_supplier1];
            att9_snationk = iatt9_snationk[tid_supplier1];
        }
        // -------- hash join probe (opId: 3) --------
        uint64_t hash3 = 0;
        if(active) {
            hash3 = 0;
            if(active) {
                hash3 = hash ( (hash3 + ((uint64_t)att9_snationk)));
            }
        }
        jpayl3* probepayl3;
        int numLookups3 = 0;
        if(active) {
            active = hashProbeUnique ( jht3, 50, hash3, numLookups3, &(probepayl3));
        }
        int bucketFound3 = 0;
        int probeActive3 = active;
        while((probeActive3 && !(bucketFound3))) {
            jpayl3 jprobepayl3 = *(probepayl3);
            att2_nnationk = jprobepayl3.att2_nnationk;
            att3_nname = jprobepayl3.att3_nname;
            bucketFound3 = 1;
            bucketFound3 &= ((att2_nnationk == att9_snationk));
            if(!(bucketFound3)) {
                probeActive3 = hashProbeUnique ( jht3, 50, hash3, numLookups3, &(probepayl3));
            }
        }
        active = bucketFound3;
        // -------- hash join build (opId: 8) --------
        if(active) {
            jpayl8 payl8;
            payl8.att3_nname = att3_nname;
            payl8.att6_ssuppkey = att6_ssuppkey;
            uint64_t hash8;
            hash8 = 0;
            if(active) {
                hash8 = hash ( (hash8 + ((uint64_t)att6_ssuppkey)));
            }
            hashBuildUnique ( jht8, 20000, hash8, &(payl8));
        }
        loopVar += step;
    }

}

__global__ void krnl_part4(
    int* iatt13_ppartkey, size_t* iatt14_pname_offset, char* iatt14_pname_char, unique_ht<jpayl7>* jht7) {
    int att13_ppartkey;
    str_t att14_pname;
    str_t c1 = stringConstant ( "%green%", 7);

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
            att13_ppartkey = iatt13_ppartkey[tid_part1];
            att14_pname = stringScan ( iatt14_pname_offset, iatt14_pname_char, tid_part1);
        }
        // -------- selection (opId: 5) --------
        if(active) {
            active = stringLikeCheck ( att14_pname, c1);
        }
        // -------- hash join build (opId: 7) --------
        if(active) {
            jpayl7 payl7;
            payl7.att13_ppartkey = att13_ppartkey;
            uint64_t hash7;
            hash7 = 0;
            if(active) {
                hash7 = hash ( (hash7 + ((uint64_t)att13_ppartkey)));
            }
            hashBuildUnique ( jht7, 20000, hash7, &(payl7));
        }
        loopVar += step;
    }

}

__global__ void krnl_partsupp6(
    int* iatt22_pspartke, int* iatt23_pssuppke, float* iatt25_pssupply, unique_ht<jpayl7>* jht7, unique_ht<jpayl8>* jht8, unique_ht<jpayl10>* jht10) {
    int att22_pspartke;
    int att23_pssuppke;
    float att25_pssupply;
    int att13_ppartkey;
    str_t att3_nname;
    int att6_ssuppkey;

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
            att22_pspartke = iatt22_pspartke[tid_partsupp1];
            att23_pssuppke = iatt23_pssuppke[tid_partsupp1];
            att25_pssupply = iatt25_pssupply[tid_partsupp1];
        }
        // -------- hash join probe (opId: 7) --------
        uint64_t hash7 = 0;
        if(active) {
            hash7 = 0;
            if(active) {
                hash7 = hash ( (hash7 + ((uint64_t)att22_pspartke)));
            }
        }
        jpayl7* probepayl7;
        int numLookups7 = 0;
        if(active) {
            active = hashProbeUnique ( jht7, 20000, hash7, numLookups7, &(probepayl7));
        }
        int bucketFound7 = 0;
        int probeActive7 = active;
        while((probeActive7 && !(bucketFound7))) {
            jpayl7 jprobepayl7 = *(probepayl7);
            att13_ppartkey = jprobepayl7.att13_ppartkey;
            bucketFound7 = 1;
            bucketFound7 &= ((att13_ppartkey == att22_pspartke));
            if(!(bucketFound7)) {
                probeActive7 = hashProbeUnique ( jht7, 20000, hash7, numLookups7, &(probepayl7));
            }
        }
        active = bucketFound7;
        // -------- hash join probe (opId: 8) --------
        uint64_t hash8 = 0;
        if(active) {
            hash8 = 0;
            if(active) {
                hash8 = hash ( (hash8 + ((uint64_t)att23_pssuppke)));
            }
        }
        jpayl8* probepayl8;
        int numLookups8 = 0;
        if(active) {
            active = hashProbeUnique ( jht8, 20000, hash8, numLookups8, &(probepayl8));
        }
        int bucketFound8 = 0;
        int probeActive8 = active;
        while((probeActive8 && !(bucketFound8))) {
            jpayl8 jprobepayl8 = *(probepayl8);
            att3_nname = jprobepayl8.att3_nname;
            att6_ssuppkey = jprobepayl8.att6_ssuppkey;
            bucketFound8 = 1;
            bucketFound8 &= ((att6_ssuppkey == att23_pssuppke));
            if(!(bucketFound8)) {
                probeActive8 = hashProbeUnique ( jht8, 20000, hash8, numLookups8, &(probepayl8));
            }
        }
        active = bucketFound8;
        // -------- hash join build (opId: 10) --------
        if(active) {
            jpayl10 payl10;
            payl10.att3_nname = att3_nname;
            payl10.att6_ssuppkey = att6_ssuppkey;
            payl10.att13_ppartkey = att13_ppartkey;
            payl10.att25_pssupply = att25_pssupply;
            uint64_t hash10;
            hash10 = 0;
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att13_ppartkey)));
            }
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att6_ssuppkey)));
            }
            hashBuildUnique ( jht10, 1600000, hash10, &(payl10));
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem9(
    int* iatt27_lorderke, int* iatt28_lpartkey, int* iatt29_lsuppkey, int* iatt31_lquantit, float* iatt32_lextende, float* iatt33_ldiscoun, unique_ht<jpayl10>* jht10, multi_ht* jht13, jpayl13* jht13_payload) {
    int att27_lorderke;
    int att28_lpartkey;
    int att29_lsuppkey;
    int att31_lquantit;
    float att32_lextende;
    float att33_ldiscoun;
    str_t att3_nname;
    int att6_ssuppkey;
    int att13_ppartkey;
    float att25_pssupply;
    float att43_amount;

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
            att27_lorderke = iatt27_lorderke[tid_lineitem1];
            att28_lpartkey = iatt28_lpartkey[tid_lineitem1];
            att29_lsuppkey = iatt29_lsuppkey[tid_lineitem1];
            att31_lquantit = iatt31_lquantit[tid_lineitem1];
            att32_lextende = iatt32_lextende[tid_lineitem1];
            att33_ldiscoun = iatt33_ldiscoun[tid_lineitem1];
        }
        // -------- hash join probe (opId: 10) --------
        uint64_t hash10 = 0;
        if(active) {
            hash10 = 0;
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att28_lpartkey)));
            }
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att29_lsuppkey)));
            }
        }
        jpayl10* probepayl10;
        int numLookups10 = 0;
        if(active) {
            active = hashProbeUnique ( jht10, 1600000, hash10, numLookups10, &(probepayl10));
        }
        int bucketFound10 = 0;
        int probeActive10 = active;
        while((probeActive10 && !(bucketFound10))) {
            jpayl10 jprobepayl10 = *(probepayl10);
            att3_nname = jprobepayl10.att3_nname;
            att6_ssuppkey = jprobepayl10.att6_ssuppkey;
            att13_ppartkey = jprobepayl10.att13_ppartkey;
            att25_pssupply = jprobepayl10.att25_pssupply;
            bucketFound10 = 1;
            bucketFound10 &= ((att13_ppartkey == att28_lpartkey));
            bucketFound10 &= ((att6_ssuppkey == att29_lsuppkey));
            if(!(bucketFound10)) {
                probeActive10 = hashProbeUnique ( jht10, 1600000, hash10, numLookups10, &(probepayl10));
            }
        }
        active = bucketFound10;
        // -------- map (opId: 11) --------
        if(active) {
            att43_amount = ((att32_lextende * ((float)1.0f - att33_ldiscoun)) - (att25_pssupply * att31_lquantit));
        }
        // -------- hash join build (opId: 13) --------
        if(active) {
            uint64_t hash13 = 0;
            if(active) {
                hash13 = 0;
                if(active) {
                    hash13 = hash ( (hash13 + ((uint64_t)att27_lorderke)));
                }
            }
            hashCountMulti ( jht13, 600120, hash13);
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem9_ins(
    int* iatt27_lorderke, int* iatt28_lpartkey, int* iatt29_lsuppkey, int* iatt31_lquantit, float* iatt32_lextende, float* iatt33_ldiscoun, unique_ht<jpayl10>* jht10, multi_ht* jht13, jpayl13* jht13_payload, int* offs13) {
    int att27_lorderke;
    int att28_lpartkey;
    int att29_lsuppkey;
    int att31_lquantit;
    float att32_lextende;
    float att33_ldiscoun;
    str_t att3_nname;
    int att6_ssuppkey;
    int att13_ppartkey;
    float att25_pssupply;
    float att43_amount;

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
            att27_lorderke = iatt27_lorderke[tid_lineitem1];
            att28_lpartkey = iatt28_lpartkey[tid_lineitem1];
            att29_lsuppkey = iatt29_lsuppkey[tid_lineitem1];
            att31_lquantit = iatt31_lquantit[tid_lineitem1];
            att32_lextende = iatt32_lextende[tid_lineitem1];
            att33_ldiscoun = iatt33_ldiscoun[tid_lineitem1];
        }
        // -------- hash join probe (opId: 10) --------
        uint64_t hash10 = 0;
        if(active) {
            hash10 = 0;
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att28_lpartkey)));
            }
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att29_lsuppkey)));
            }
        }
        jpayl10* probepayl10;
        int numLookups10 = 0;
        if(active) {
            active = hashProbeUnique ( jht10, 1600000, hash10, numLookups10, &(probepayl10));
        }
        int bucketFound10 = 0;
        int probeActive10 = active;
        while((probeActive10 && !(bucketFound10))) {
            jpayl10 jprobepayl10 = *(probepayl10);
            att3_nname = jprobepayl10.att3_nname;
            att6_ssuppkey = jprobepayl10.att6_ssuppkey;
            att13_ppartkey = jprobepayl10.att13_ppartkey;
            att25_pssupply = jprobepayl10.att25_pssupply;
            bucketFound10 = 1;
            bucketFound10 &= ((att13_ppartkey == att28_lpartkey));
            bucketFound10 &= ((att6_ssuppkey == att29_lsuppkey));
            if(!(bucketFound10)) {
                probeActive10 = hashProbeUnique ( jht10, 1600000, hash10, numLookups10, &(probepayl10));
            }
        }
        active = bucketFound10;
        // -------- map (opId: 11) --------
        if(active) {
            att43_amount = ((att32_lextende * ((float)1.0f - att33_ldiscoun)) - (att25_pssupply * att31_lquantit));
        }
        // -------- hash join build (opId: 13) --------
        if(active) {
            uint64_t hash13 = 0;
            if(active) {
                hash13 = 0;
                if(active) {
                    hash13 = hash ( (hash13 + ((uint64_t)att27_lorderke)));
                }
            }
            jpayl13 payl;
            payl.att3_nname = att3_nname;
            payl.att27_lorderke = att27_lorderke;
            payl.att43_amount = att43_amount;
            hashInsertMulti ( jht13, jht13_payload, offs13, 600120, hash13, &(payl));
        }
        loopVar += step;
    }

}

__global__ void krnl_orders12(
    int* iatt44_oorderke, unsigned* iatt48_oorderda, multi_ht* jht13, jpayl13* jht13_payload, agg_ht<apayl15>* aht15, float* agg1) {
    int att44_oorderke;
    unsigned att48_oorderda;
    unsigned warplane = (threadIdx.x % 32);
    str_t att3_nname;
    int att27_lorderke;
    float att43_amount;
    unsigned att53_oyear;

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
            att44_oorderke = iatt44_oorderke[tid_orders1];
            att48_oorderda = iatt48_oorderda[tid_orders1];
        }
        // -------- hash join probe (opId: 13) --------
        // -------- multiprobe multi broadcast (opId: 13) --------
        int matchEnd13 = 0;
        int matchEndBuf13 = 0;
        int matchOffset13 = 0;
        int matchOffsetBuf13 = 0;
        int probeActive13 = active;
        int att44_oorderke_bcbuf13;
        unsigned att48_oorderda_bcbuf13;
        uint64_t hash13 = 0;
        if(probeActive13) {
            hash13 = 0;
            if(active) {
                hash13 = hash ( (hash13 + ((uint64_t)att44_oorderke)));
            }
            probeActive13 = hashProbeMulti ( jht13, 600120, hash13, matchOffsetBuf13, matchEndBuf13);
        }
        unsigned activeProbes13 = __ballot_sync(ALL_LANES,probeActive13);
        int num13 = 0;
        num13 = (matchEndBuf13 - matchOffsetBuf13);
        unsigned wideProbes13 = __ballot_sync(ALL_LANES,(num13 >= 32));
        att44_oorderke_bcbuf13 = att44_oorderke;
        att48_oorderda_bcbuf13 = att48_oorderda;
        while((activeProbes13 > 0)) {
            unsigned tupleLane;
            unsigned broadcastLane;
            int numFilled = 0;
            int num = 0;
            while(((numFilled < 32) && activeProbes13)) {
                if((wideProbes13 > 0)) {
                    tupleLane = (__ffs(wideProbes13) - 1);
                    wideProbes13 -= (1 << tupleLane);
                }
                else {
                    tupleLane = (__ffs(activeProbes13) - 1);
                }
                num = __shfl_sync(ALL_LANES,num13,tupleLane);
                if((numFilled && ((numFilled + num) > 32))) {
                    break;
                }
                if((warplane >= numFilled)) {
                    broadcastLane = tupleLane;
                    matchOffset13 = (warplane - numFilled);
                }
                numFilled += num;
                activeProbes13 -= (1 << tupleLane);
            }
            matchOffset13 += __shfl_sync(ALL_LANES,matchOffsetBuf13,broadcastLane);
            matchEnd13 = __shfl_sync(ALL_LANES,matchEndBuf13,broadcastLane);
            att44_oorderke = __shfl_sync(ALL_LANES,att44_oorderke_bcbuf13,broadcastLane);
            att48_oorderda = __shfl_sync(ALL_LANES,att48_oorderda_bcbuf13,broadcastLane);
            probeActive13 = (matchOffset13 < matchEnd13);
            while(__any_sync(ALL_LANES,probeActive13)) {
                active = probeActive13;
                active = 0;
                jpayl13 payl;
                if(probeActive13) {
                    payl = jht13_payload[matchOffset13];
                    att3_nname = payl.att3_nname;
                    att27_lorderke = payl.att27_lorderke;
                    att43_amount = payl.att43_amount;
                    active = 1;
                    active &= ((att27_lorderke == att44_oorderke));
                    matchOffset13 += 32;
                    probeActive13 &= ((matchOffset13 < matchEnd13));
                }
                // -------- map (opId: 14) --------
                if(active) {
                    att53_oyear = (att48_oorderda / 10000);
                }
                // -------- aggregation (opId: 15) --------
                int bucket = 0;
                if(active) {
                    uint64_t hash15 = 0;
                    hash15 = 0;
                    hash15 = hash ( (hash15 + stringHash ( att3_nname)));
                    if(active) {
                        hash15 = hash ( (hash15 + ((uint64_t)att53_oyear)));
                    }
                    apayl15 payl;
                    payl.att3_nname = att3_nname;
                    payl.att53_oyear = att53_oyear;
                    int bucketFound = 0;
                    int numLookups = 0;
                    while(!(bucketFound)) {
                        bucket = hashAggregateGetBucket ( aht15, 2000, hash15, numLookups, &(payl));
                        apayl15 probepayl = aht15[bucket].payload;
                        bucketFound = 1;
                        bucketFound &= (stringEquals ( payl.att3_nname, probepayl.att3_nname));
                        bucketFound &= ((payl.att53_oyear == probepayl.att53_oyear));
                    }
                }
                if(active) {
                    atomicAdd(&(agg1[bucket]), ((float)att43_amount));
                }
            }
        }
        loopVar += step;
    }

}

__global__ void krnl_aggregation15(
    agg_ht<apayl15>* aht15, float* agg1, int* nout_result, str_offs* oatt3_nname_offset, char* iatt3_nname_char, unsigned* oatt53_oyear, float* oatt1_sumprofi) {
    str_t att3_nname;
    unsigned att53_oyear;
    float att1_sumprofi;
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));

    int tid_aggregation15 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation15 = loopVar;
        active = (loopVar < 2000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 15) --------
        if(active) {
            active &= ((aht15[tid_aggregation15].lock.lock == OnceLock::LOCK_DONE));
        }
        if(active) {
            apayl15 payl = aht15[tid_aggregation15].payload;
            att3_nname = payl.att3_nname;
            att53_oyear = payl.att53_oyear;
        }
        if(active) {
            att1_sumprofi = agg1[tid_aggregation15];
        }
        // -------- projection (no code) (opId: 16) --------
        // -------- materialize (opId: 17) --------
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
            oatt3_nname_offset[wp] = toStringOffset ( iatt3_nname_char, att3_nname);
            oatt53_oyear[wp] = att53_oyear;
            oatt1_sumprofi[wp] = att1_sumprofi;
        }
        loopVar += step;
    }

}

int main() {
    int* iatt2_nnationk;
    iatt2_nnationk = ( int*) map_memory_file ( "mmdb/nation_n_nationkey" );
    size_t* iatt3_nname_offset;
    iatt3_nname_offset = ( size_t*) map_memory_file ( "mmdb/nation_n_name_offset" );
    char* iatt3_nname_char;
    iatt3_nname_char = ( char*) map_memory_file ( "mmdb/nation_n_name_char" );
    int* iatt6_ssuppkey;
    iatt6_ssuppkey = ( int*) map_memory_file ( "mmdb/supplier_s_suppkey" );
    int* iatt9_snationk;
    iatt9_snationk = ( int*) map_memory_file ( "mmdb/supplier_s_nationkey" );
    int* iatt13_ppartkey;
    iatt13_ppartkey = ( int*) map_memory_file ( "mmdb/part_p_partkey" );
    size_t* iatt14_pname_offset;
    iatt14_pname_offset = ( size_t*) map_memory_file ( "mmdb/part_p_name_offset" );
    char* iatt14_pname_char;
    iatt14_pname_char = ( char*) map_memory_file ( "mmdb/part_p_name_char" );
    int* iatt22_pspartke;
    iatt22_pspartke = ( int*) map_memory_file ( "mmdb/partsupp_ps_partkey" );
    int* iatt23_pssuppke;
    iatt23_pssuppke = ( int*) map_memory_file ( "mmdb/partsupp_ps_suppkey" );
    float* iatt25_pssupply;
    iatt25_pssupply = ( float*) map_memory_file ( "mmdb/partsupp_ps_supplycost" );
    int* iatt27_lorderke;
    iatt27_lorderke = ( int*) map_memory_file ( "mmdb/lineitem_l_orderkey" );
    int* iatt28_lpartkey;
    iatt28_lpartkey = ( int*) map_memory_file ( "mmdb/lineitem_l_partkey" );
    int* iatt29_lsuppkey;
    iatt29_lsuppkey = ( int*) map_memory_file ( "mmdb/lineitem_l_suppkey" );
    int* iatt31_lquantit;
    iatt31_lquantit = ( int*) map_memory_file ( "mmdb/lineitem_l_quantity" );
    float* iatt32_lextende;
    iatt32_lextende = ( float*) map_memory_file ( "mmdb/lineitem_l_extendedprice" );
    float* iatt33_ldiscoun;
    iatt33_ldiscoun = ( float*) map_memory_file ( "mmdb/lineitem_l_discount" );
    int* iatt44_oorderke;
    iatt44_oorderke = ( int*) map_memory_file ( "mmdb/orders_o_orderkey" );
    unsigned* iatt48_oorderda;
    iatt48_oorderda = ( unsigned*) map_memory_file ( "mmdb/orders_o_orderdate" );

    int nout_result;
    std::vector < str_offs > oatt3_nname_offset(1000);
    std::vector < unsigned > oatt53_oyear(1000);
    std::vector < float > oatt1_sumprofi(1000);

    // wake up gpu
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in wake up gpu! " << cudaGetErrorString( err ) << std::endl;
            ERROR("wake up gpu")
        }
    }

    int* d_iatt2_nnationk;
    cudaMalloc((void**) &d_iatt2_nnationk, 25* sizeof(int) );
    size_t* d_iatt3_nname_offset;
    cudaMalloc((void**) &d_iatt3_nname_offset, (25 + 1)* sizeof(size_t) );
    char* d_iatt3_nname_char;
    cudaMalloc((void**) &d_iatt3_nname_char, 186* sizeof(char) );
    int* d_iatt6_ssuppkey;
    cudaMalloc((void**) &d_iatt6_ssuppkey, 10000* sizeof(int) );
    int* d_iatt9_snationk;
    cudaMalloc((void**) &d_iatt9_snationk, 10000* sizeof(int) );
    int* d_iatt13_ppartkey;
    cudaMalloc((void**) &d_iatt13_ppartkey, 200000* sizeof(int) );
    size_t* d_iatt14_pname_offset;
    cudaMalloc((void**) &d_iatt14_pname_offset, (200000 + 1)* sizeof(size_t) );
    char* d_iatt14_pname_char;
    cudaMalloc((void**) &d_iatt14_pname_char, 6550230* sizeof(char) );
    int* d_iatt22_pspartke;
    cudaMalloc((void**) &d_iatt22_pspartke, 800000* sizeof(int) );
    int* d_iatt23_pssuppke;
    cudaMalloc((void**) &d_iatt23_pssuppke, 800000* sizeof(int) );
    float* d_iatt25_pssupply;
    cudaMalloc((void**) &d_iatt25_pssupply, 800000* sizeof(float) );
    int* d_iatt27_lorderke;
    cudaMalloc((void**) &d_iatt27_lorderke, 6001215* sizeof(int) );
    int* d_iatt28_lpartkey;
    cudaMalloc((void**) &d_iatt28_lpartkey, 6001215* sizeof(int) );
    int* d_iatt29_lsuppkey;
    cudaMalloc((void**) &d_iatt29_lsuppkey, 6001215* sizeof(int) );
    int* d_iatt31_lquantit;
    cudaMalloc((void**) &d_iatt31_lquantit, 6001215* sizeof(int) );
    float* d_iatt32_lextende;
    cudaMalloc((void**) &d_iatt32_lextende, 6001215* sizeof(float) );
    float* d_iatt33_ldiscoun;
    cudaMalloc((void**) &d_iatt33_ldiscoun, 6001215* sizeof(float) );
    int* d_iatt44_oorderke;
    cudaMalloc((void**) &d_iatt44_oorderke, 1500000* sizeof(int) );
    unsigned* d_iatt48_oorderda;
    cudaMalloc((void**) &d_iatt48_oorderda, 1500000* sizeof(unsigned) );
    int* d_nout_result;
    cudaMalloc((void**) &d_nout_result, 1* sizeof(int) );
    str_offs* d_oatt3_nname_offset;
    cudaMalloc((void**) &d_oatt3_nname_offset, 1000* sizeof(str_offs) );
    unsigned* d_oatt53_oyear;
    cudaMalloc((void**) &d_oatt53_oyear, 1000* sizeof(unsigned) );
    float* d_oatt1_sumprofi;
    cudaMalloc((void**) &d_oatt1_sumprofi, 1000* sizeof(float) );
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

    unique_ht<jpayl3>* d_jht3;
    cudaMalloc((void**) &d_jht3, 50* sizeof(unique_ht<jpayl3>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht3, 50);
    }
    unique_ht<jpayl8>* d_jht8;
    cudaMalloc((void**) &d_jht8, 20000* sizeof(unique_ht<jpayl8>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht8, 20000);
    }
    unique_ht<jpayl7>* d_jht7;
    cudaMalloc((void**) &d_jht7, 20000* sizeof(unique_ht<jpayl7>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht7, 20000);
    }
    unique_ht<jpayl10>* d_jht10;
    cudaMalloc((void**) &d_jht10, 1600000* sizeof(unique_ht<jpayl10>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht10, 1600000);
    }
    multi_ht* d_jht13;
    cudaMalloc((void**) &d_jht13, 600120* sizeof(multi_ht) );
    jpayl13* d_jht13_payload;
    cudaMalloc((void**) &d_jht13_payload, 600120* sizeof(jpayl13) );
    {
        int gridsize=920;
        int blocksize=128;
        initMultiHT<<<gridsize, blocksize>>>(d_jht13, 600120);
    }
    int* d_offs13;
    cudaMalloc((void**) &d_offs13, 1* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_offs13, 0, 1);
    }
    agg_ht<apayl15>* d_aht15;
    cudaMalloc((void**) &d_aht15, 2000* sizeof(agg_ht<apayl15>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_aht15, 2000);
    }
    float* d_agg1;
    cudaMalloc((void**) &d_agg1, 2000* sizeof(float) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg1, 0.0f, 2000);
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

    cudaMemcpy( d_iatt2_nnationk, iatt2_nnationk, 25 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt3_nname_offset, iatt3_nname_offset, (25 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt3_nname_char, iatt3_nname_char, 186 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt6_ssuppkey, iatt6_ssuppkey, 10000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt9_snationk, iatt9_snationk, 10000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt13_ppartkey, iatt13_ppartkey, 200000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt14_pname_offset, iatt14_pname_offset, (200000 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt14_pname_char, iatt14_pname_char, 6550230 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt22_pspartke, iatt22_pspartke, 800000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt23_pssuppke, iatt23_pssuppke, 800000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt25_pssupply, iatt25_pssupply, 800000 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt27_lorderke, iatt27_lorderke, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt28_lpartkey, iatt28_lpartkey, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt29_lsuppkey, iatt29_lsuppkey, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt31_lquantit, iatt31_lquantit, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt32_lextende, iatt32_lextende, 6001215 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt33_ldiscoun, iatt33_ldiscoun, 6001215 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt44_oorderke, iatt44_oorderke, 1500000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt48_oorderda, iatt48_oorderda, 1500000 * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy in! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy in")
        }
    }

    std::clock_t start_totalKernelTime75 = std::clock();
    std::clock_t start_krnl_nation176 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_nation1<<<gridsize, blocksize>>>(d_iatt2_nnationk, d_iatt3_nname_offset, d_iatt3_nname_char, d_jht3);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_nation176 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_nation1! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_nation1")
        }
    }

    std::clock_t start_krnl_supplier277 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_supplier2<<<gridsize, blocksize>>>(d_iatt6_ssuppkey, d_iatt9_snationk, d_jht3, d_jht8);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_supplier277 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_supplier2! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_supplier2")
        }
    }

    std::clock_t start_krnl_part478 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_part4<<<gridsize, blocksize>>>(d_iatt13_ppartkey, d_iatt14_pname_offset, d_iatt14_pname_char, d_jht7);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_part478 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_part4! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_part4")
        }
    }

    std::clock_t start_krnl_partsupp679 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_partsupp6<<<gridsize, blocksize>>>(d_iatt22_pspartke, d_iatt23_pssuppke, d_iatt25_pssupply, d_jht7, d_jht8, d_jht10);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_partsupp679 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_partsupp6! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_partsupp6")
        }
    }

    std::clock_t start_krnl_lineitem980 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem9<<<gridsize, blocksize>>>(d_iatt27_lorderke, d_iatt28_lpartkey, d_iatt29_lsuppkey, d_iatt31_lquantit, d_iatt32_lextende, d_iatt33_ldiscoun, d_jht10, d_jht13, d_jht13_payload);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem980 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem9! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem9")
        }
    }

    std::clock_t start_scanMultiHT81 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        scanMultiHT<<<gridsize, blocksize>>>(d_jht13, 600120, d_offs13);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_scanMultiHT81 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in scanMultiHT! " << cudaGetErrorString( err ) << std::endl;
            ERROR("scanMultiHT")
        }
    }

    std::clock_t start_krnl_lineitem9_ins82 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem9_ins<<<gridsize, blocksize>>>(d_iatt27_lorderke, d_iatt28_lpartkey, d_iatt29_lsuppkey, d_iatt31_lquantit, d_iatt32_lextende, d_iatt33_ldiscoun, d_jht10, d_jht13, d_jht13_payload, d_offs13);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem9_ins82 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem9_ins! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem9_ins")
        }
    }

    std::clock_t start_krnl_orders1283 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_orders12<<<gridsize, blocksize>>>(d_iatt44_oorderke, d_iatt48_oorderda, d_jht13, d_jht13_payload, d_aht15, d_agg1);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_orders1283 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_orders12! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_orders12")
        }
    }

    std::clock_t start_krnl_aggregation1584 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation15<<<gridsize, blocksize>>>(d_aht15, d_agg1, d_nout_result, d_oatt3_nname_offset, d_iatt3_nname_char, d_oatt53_oyear, d_oatt1_sumprofi);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation1584 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation15! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation15")
        }
    }

    std::clock_t stop_totalKernelTime75 = std::clock();
    cudaMemcpy( &nout_result, d_nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt3_nname_offset.data(), d_oatt3_nname_offset, 1000 * sizeof(str_offs), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt53_oyear.data(), d_oatt53_oyear, 1000 * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt1_sumprofi.data(), d_oatt1_sumprofi, 1000 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy out! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy out")
        }
    }

    cudaFree( d_iatt2_nnationk);
    cudaFree( d_iatt3_nname_offset);
    cudaFree( d_iatt3_nname_char);
    cudaFree( d_jht3);
    cudaFree( d_iatt6_ssuppkey);
    cudaFree( d_iatt9_snationk);
    cudaFree( d_jht8);
    cudaFree( d_iatt13_ppartkey);
    cudaFree( d_iatt14_pname_offset);
    cudaFree( d_iatt14_pname_char);
    cudaFree( d_jht7);
    cudaFree( d_iatt22_pspartke);
    cudaFree( d_iatt23_pssuppke);
    cudaFree( d_iatt25_pssupply);
    cudaFree( d_jht10);
    cudaFree( d_iatt27_lorderke);
    cudaFree( d_iatt28_lpartkey);
    cudaFree( d_iatt29_lsuppkey);
    cudaFree( d_iatt31_lquantit);
    cudaFree( d_iatt32_lextende);
    cudaFree( d_iatt33_ldiscoun);
    cudaFree( d_jht13);
    cudaFree( d_jht13_payload);
    cudaFree( d_offs13);
    cudaFree( d_iatt44_oorderke);
    cudaFree( d_iatt48_oorderda);
    cudaFree( d_aht15);
    cudaFree( d_agg1);
    cudaFree( d_nout_result);
    cudaFree( d_oatt3_nname_offset);
    cudaFree( d_oatt53_oyear);
    cudaFree( d_oatt1_sumprofi);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda free! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda free")
        }
    }

    std::clock_t start_finish85 = std::clock();
    printf("\nResult: %i tuples\n", nout_result);
    if((nout_result > 1000)) {
        ERROR("Index out of range. Output size larger than allocated with expected result number.")
    }
    for ( int pv = 0; ((pv < 10) && (pv < nout_result)); pv += 1) {
        printf("n_name: ");
        stringPrint ( iatt3_nname_char, oatt3_nname_offset[pv]);
        printf("  ");
        printf("o_year: ");
        printf("%10i", oatt53_oyear[pv]);
        printf("  ");
        printf("sum_profit: ");
        printf("%15.2f", oatt1_sumprofi[pv]);
        printf("  ");
        printf("\n");
    }
    if((nout_result > 10)) {
        printf("[...]\n");
    }
    printf("\n");
    std::clock_t stop_finish85 = std::clock();

    printf("<timing>\n");
    printf ( "%32s: %6.1f ms\n", "krnl_nation1", (stop_krnl_nation176 - start_krnl_nation176) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_supplier2", (stop_krnl_supplier277 - start_krnl_supplier277) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_part4", (stop_krnl_part478 - start_krnl_part478) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_partsupp6", (stop_krnl_partsupp679 - start_krnl_partsupp679) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem9", (stop_krnl_lineitem980 - start_krnl_lineitem980) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "scanMultiHT", (stop_scanMultiHT81 - start_scanMultiHT81) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem9_ins", (stop_krnl_lineitem9_ins82 - start_krnl_lineitem9_ins82) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_orders12", (stop_krnl_orders1283 - start_krnl_orders1283) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation15", (stop_krnl_aggregation1584 - start_krnl_aggregation1584) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "finish", (stop_finish85 - start_finish85) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "totalKernelTime", (stop_totalKernelTime75 - start_totalKernelTime75) / (double) (CLOCKS_PER_SEC / 1000) );
    printf("</timing>\n");
}
