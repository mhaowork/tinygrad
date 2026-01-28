// opts: Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.THREAD, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4)
// source: .cache/beam_latest_2048_candidates_debug.log
// kernel: r_64_4_1024_4_4
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_64_4_1024_4_4(float* restrict data0_256, float* restrict data1_4194304, int core_id) {
  float acc0[16];
  int gidx0 = core_id; /* 4 */
  for (int Lidx2 = 0; Lidx2 < 64; Lidx2++) {
    *(acc0+0) = 0.0f;
    *(acc0+1) = 0.0f;
    *(acc0+2) = 0.0f;
    *(acc0+3) = 0.0f;
    *(acc0+4) = 0.0f;
    *(acc0+5) = 0.0f;
    *(acc0+6) = 0.0f;
    *(acc0+7) = 0.0f;
    *(acc0+8) = 0.0f;
    *(acc0+9) = 0.0f;
    *(acc0+10) = 0.0f;
    *(acc0+11) = 0.0f;
    *(acc0+12) = 0.0f;
    *(acc0+13) = 0.0f;
    *(acc0+14) = 0.0f;
    *(acc0+15) = 0.0f;
    for (int Ridx0 = 0; Ridx0 < 1024; Ridx0++) {
      int alu16 = ((gidx0<<20)+(Lidx2<<14)+(Ridx0<<4));
      float4 val0 = (*((float4*)((data1_4194304+(alu16+4)))));
      float4 val1 = (*((float4*)((data1_4194304+(alu16+8)))));
      float4 val2 = (*((float4*)((data1_4194304+(alu16+12)))));
      float4 val3 = (*((float4*)((data1_4194304+alu16))));
      *(acc0+0) = ((*(acc0+0))+val3[0]);
      *(acc0+1) = ((*(acc0+1))+val3[1]);
      *(acc0+2) = ((*(acc0+2))+val3[2]);
      *(acc0+3) = ((*(acc0+3))+val3[3]);
      *(acc0+4) = ((*(acc0+4))+val0[0]);
      *(acc0+5) = ((*(acc0+5))+val0[1]);
      *(acc0+6) = ((*(acc0+6))+val0[2]);
      *(acc0+7) = ((*(acc0+7))+val0[3]);
      *(acc0+8) = ((*(acc0+8))+val1[0]);
      *(acc0+9) = ((*(acc0+9))+val1[1]);
      *(acc0+10) = ((*(acc0+10))+val1[2]);
      *(acc0+11) = ((*(acc0+11))+val1[3]);
      *(acc0+12) = ((*(acc0+12))+val2[0]);
      *(acc0+13) = ((*(acc0+13))+val2[1]);
      *(acc0+14) = ((*(acc0+14))+val2[2]);
      *(acc0+15) = ((*(acc0+15))+val2[3]);
    }
    *(data0_256+((gidx0<<6)+Lidx2)) = ((*(acc0+0))+(*(acc0+1))+(*(acc0+2))+(*(acc0+3))+(*(acc0+4))+(*(acc0+5))+(*(acc0+6))+(*(acc0+7))+(*(acc0+8))+(*(acc0+9))+(*(acc0+10))+(*(acc0+11))+(*(acc0+12))+(*(acc0+13))+(*(acc0+14))+(*(acc0+15)));
  }
}
