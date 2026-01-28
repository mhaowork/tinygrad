; ---- r[90m_[0m[97m32[0m[90m_[0m[94m8[0m[90m_[0m[31m4096[0m[90m_[0m[35m4[0m[90m[0m gs=[8, 1, 1] ----
define void @r_32_8_4096_4(float* noalias align 32 %data0, float* noalias align 32 %data1, i32 %core_id) #0
{
  %reg_0 = alloca [1 x <4 x float>]
  %v0 = add i32 %core_id, 0
  %v1_z = insertelement <1 x float> poison, float 0.0, i32 0
  %v1 = shufflevector <1 x float> %v1_z, <1 x float> poison, <4 x i32> zeroinitializer
  %v2 = getelementptr inbounds <4 x float>, <4 x float>* %reg_0, i32 0
  %v3 = mul i32 %v0, 32
  %v4 = mul i32 %v0, 524288
  br label %loop_entry_2
loop_entry_2:
  br label %loop_latch_2
loop_latch_2:
  %v5 = phi i32 [ 0, %loop_entry_2 ], [ %v5phi, %loop_footer_2 ]
  %v5phi = add i32 %v5, 1
  %v5cmp = icmp ult i32 %v5, 32
  br i1 %v5cmp, label %loop_body_2, label %loop_exit_2
loop_body_2:
  %v6 = getelementptr inbounds <4 x float>, <4 x float>* %reg_0, i32 0
  %v7 = mul i32 %v5, 16384
  %v8 = add nsw i32 %v4, %v7
  store <4 x float> %v1, <4 x float>* %v6
  br label %loop_entry_0
loop_entry_0:
  br label %loop_latch_0
loop_latch_0:
  %v10 = phi i32 [ 0, %loop_entry_0 ], [ %v10phi, %loop_footer_0 ]
  %v10phi = add i32 %v10, 1
  %v10cmp = icmp ult i32 %v10, 4096
  br i1 %v10cmp, label %loop_body_0, label %loop_exit_0
loop_body_0:
  %v11 = getelementptr inbounds <4 x float>, <4 x float>* %reg_0, i32 0
  %v12 = load <4 x float>, <4 x float>* %v11
  %v13 = mul i32 %v10, 4
  %v14 = add nsw i32 %v8, %v13
  %v15 = getelementptr inbounds float, float* %data1, i32 %v14
  %v16 = load <4 x float>, <4 x float>* %v15
  %v17 = fadd nsz arcp contract afn <4 x float> %v12, %v16
  store <4 x float> %v17, <4 x float>* %v2
  br label %loop_footer_0
loop_footer_0:
  br label %loop_latch_0
loop_exit_0:
  %v20 = getelementptr inbounds <4 x float>, <4 x float>* %reg_0, i32 0
  %v21 = load <4 x float>, <4 x float>* %v20
  %v22 = extractelement <4 x float> %v21, i32 0
  %v23 = extractelement <4 x float> %v21, i32 1
  %v24 = extractelement <4 x float> %v21, i32 2
  %v25 = extractelement <4 x float> %v21, i32 3
  %v26 = add nsw i32 %v3, %v5
  %v27 = getelementptr inbounds float, float* %data0, i32 %v26
  %v28 = fadd nsz arcp contract afn float %v22, %v23
  %v29 = fadd nsz arcp contract afn float %v28, %v24
  %v30 = fadd nsz arcp contract afn float %v29, %v25
  store float %v30, float* %v27
  br label %loop_footer_2
loop_footer_2:
  br label %loop_latch_2
loop_exit_2:
  ret void
}
attributes #0 = { alwaysinline nounwind "no-builtins" "no-trapping-math"="true" }

; ---- r[90m_[0m[31m64[0m[90m_[0m[35m4[0m[90m[0m gs=[1, 1, 1] ----
define void @r_64_4(float* noalias align 32 %data0, float* noalias align 32 %data1, i32 %core_id) #0
{
  %reg_0 = alloca [1 x <4 x float>]
  %v0_z = insertelement <1 x float> poison, float 0.0, i32 0
  %v0 = shufflevector <1 x float> %v0_z, <1 x float> poison, <4 x i32> zeroinitializer
  %v1 = getelementptr inbounds <4 x float>, <4 x float>* %reg_0, i32 0
  store <4 x float> %v0, <4 x float>* %v1
  br label %loop_entry_0
loop_entry_0:
  br label %loop_latch_0
loop_latch_0:
  %v3 = phi i32 [ 0, %loop_entry_0 ], [ %v3phi, %loop_footer_0 ]
  %v3phi = add i32 %v3, 1
  %v3cmp = icmp ult i32 %v3, 64
  br i1 %v3cmp, label %loop_body_0, label %loop_exit_0
loop_body_0:
  %v4 = getelementptr inbounds <4 x float>, <4 x float>* %reg_0, i32 0
  %v5 = load <4 x float>, <4 x float>* %v4
  %v6 = mul i32 %v3, 4
  %v7 = getelementptr inbounds float, float* %data1, i32 %v6
  %v8 = load <4 x float>, <4 x float>* %v7
  %v9 = fadd nsz arcp contract afn <4 x float> %v5, %v8
  store <4 x float> %v9, <4 x float>* %v1
  br label %loop_footer_0
loop_footer_0:
  br label %loop_latch_0
loop_exit_0:
  %v12 = getelementptr inbounds <4 x float>, <4 x float>* %reg_0, i32 0
  %v13 = load <4 x float>, <4 x float>* %v12
  %v14 = extractelement <4 x float> %v13, i32 0
  %v15 = extractelement <4 x float> %v13, i32 1
  %v16 = extractelement <4 x float> %v13, i32 2
  %v17 = extractelement <4 x float> %v13, i32 3
  %v18 = getelementptr inbounds float, float* %data0, i32 0
  %v19 = fadd nsz arcp contract afn float %v14, %v15
  %v20 = fadd nsz arcp contract afn float %v19, %v16
  %v21 = fadd nsz arcp contract afn float %v20, %v17
  store float %v21, float* %v18
  ret void
}
attributes #0 = { alwaysinline nounwind "no-builtins" "no-trapping-math"="true" }