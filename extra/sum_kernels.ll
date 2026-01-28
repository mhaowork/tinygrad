; ---- r[90m_[0m[97m32[0m[90m_[0m[94m8[0m[90m_[0m[31m4096[0m[90m_[0m[35m4[0m[90m[0m ----
define void @r_32_8_4096_4(float* noalias align 32 %data0, float* noalias align 32 %data1, i32 %core_id) #0
{
  %reg_0 = alloca [4 x float]
  %v0 = add i32 %core_id, 0
  %v1 = getelementptr inbounds float, float* %reg_0, i32 0
  %v2 = getelementptr inbounds float, float* %reg_0, i32 1
  %v3 = getelementptr inbounds float, float* %reg_0, i32 2
  %v4 = getelementptr inbounds float, float* %reg_0, i32 3
  %v5 = mul i32 %v0, 32
  %v6 = mul i32 %v0, 524288
  br label %loop_entry_2
loop_entry_2:
  br label %loop_latch_2
loop_latch_2:
  %v7 = phi i32 [ 0, %loop_entry_2 ], [ %v7phi, %loop_footer_2 ]
  %v7phi = add i32 %v7, 1
  %v7cmp = icmp ult i32 %v7, 32
  br i1 %v7cmp, label %loop_body_2, label %loop_exit_2
loop_body_2:
  %v8 = getelementptr inbounds float, float* %reg_0, i32 0
  %v9 = getelementptr inbounds float, float* %reg_0, i32 1
  %v10 = getelementptr inbounds float, float* %reg_0, i32 2
  %v11 = getelementptr inbounds float, float* %reg_0, i32 3
  %v12 = mul i32 %v7, 16384
  %v13 = add nsw i32 %v6, %v12
  store float 0.0, float* %v8
  store float 0.0, float* %v9
  store float 0.0, float* %v11
  store float 0.0, float* %v11
  br label %loop_entry_0
loop_entry_0:
  br label %loop_latch_0
loop_latch_0:
  %v18 = phi i32 [ 0, %loop_entry_0 ], [ %v18phi, %loop_footer_0 ]
  %v18phi = add i32 %v18, 1
  %v18cmp = icmp ult i32 %v18, 4096
  br i1 %v18cmp, label %loop_body_0, label %loop_exit_0
loop_body_0:
  %v19 = getelementptr inbounds float, float* %reg_0, i32 0
  %v20 = load float, float* %v19
  %v21 = getelementptr inbounds float, float* %reg_0, i32 1
  %v22 = load float, float* %v21
  %v23 = getelementptr inbounds float, float* %reg_0, i32 2
  %v24 = load float, float* %v23
  %v25 = getelementptr inbounds float, float* %reg_0, i32 3
  %v26 = load float, float* %v25
  %v27 = mul i32 %v18, 4
  %v28 = add nsw i32 %v13, %v27
  %v29 = getelementptr inbounds float, float* %data1, i32 %v28
  %v30 = load <4 x float>, <4 x float>* %v29
  %v31 = extractelement <4 x float> %v30, i32 0
  %v32 = extractelement <4 x float> %v30, i32 1
  %v33 = extractelement <4 x float> %v30, i32 2
  %v34 = extractelement <4 x float> %v30, i32 3
  %v35 = fadd nsz arcp contract afn float %v20, %v31
  %v36 = fadd nsz arcp contract afn float %v22, %v32
  %v37 = fadd nsz arcp contract afn float %v24, %v33
  %v38 = fadd nsz arcp contract afn float %v26, %v34
  store float %v35, float* %v1
  store float %v36, float* %v2
  store float %v37, float* %v3
  store float %v38, float* %v4
  br label %loop_footer_0
loop_footer_0:
  br label %loop_latch_0
loop_exit_0:
  %v44 = getelementptr inbounds float, float* %reg_0, i32 0
  %v45 = load float, float* %v44
  %v46 = getelementptr inbounds float, float* %reg_0, i32 1
  %v47 = load float, float* %v46
  %v48 = getelementptr inbounds float, float* %reg_0, i32 2
  %v49 = load float, float* %v48
  %v50 = getelementptr inbounds float, float* %reg_0, i32 3
  %v51 = load float, float* %v50
  %v52 = add nsw i32 %v5, %v7
  %v53 = getelementptr inbounds float, float* %data0, i32 %v52
  %v54 = fadd nsz arcp contract afn float %v45, %v47
  %v55 = fadd nsz arcp contract afn float %v54, %v49
  %v56 = fadd nsz arcp contract afn float %v55, %v51
  store float %v56, float* %v53
  br label %loop_footer_2
loop_footer_2:
  br label %loop_latch_2
loop_exit_2:
  ret void
}
attributes #0 = { alwaysinline nounwind "no-builtins" "no-trapping-math"="true" }

; ---- r[90m_[0m[31m64[0m[90m_[0m[35m4[0m[90m[0m ----
define void @r_64_4(float* noalias align 32 %data0, float* noalias align 32 %data1, i32 %core_id) #0
{
  %reg_0 = alloca [4 x float]
  %v0 = getelementptr inbounds float, float* %reg_0, i32 0
  %v1 = getelementptr inbounds float, float* %reg_0, i32 1
  %v2 = getelementptr inbounds float, float* %reg_0, i32 2
  %v3 = getelementptr inbounds float, float* %reg_0, i32 3
  store float 0.0, float* %v0
  store float 0.0, float* %v1
  store float 0.0, float* %v2
  store float 0.0, float* %v3
  br label %loop_entry_0
loop_entry_0:
  br label %loop_latch_0
loop_latch_0:
  %v8 = phi i32 [ 0, %loop_entry_0 ], [ %v8phi, %loop_footer_0 ]
  %v8phi = add i32 %v8, 1
  %v8cmp = icmp ult i32 %v8, 64
  br i1 %v8cmp, label %loop_body_0, label %loop_exit_0
loop_body_0:
  %v9 = getelementptr inbounds float, float* %reg_0, i32 0
  %v10 = load float, float* %v9
  %v11 = getelementptr inbounds float, float* %reg_0, i32 1
  %v12 = load float, float* %v11
  %v13 = getelementptr inbounds float, float* %reg_0, i32 2
  %v14 = load float, float* %v13
  %v15 = getelementptr inbounds float, float* %reg_0, i32 3
  %v16 = load float, float* %v15
  %v17 = mul i32 %v8, 4
  %v18 = getelementptr inbounds float, float* %data1, i32 %v17
  %v19 = load <4 x float>, <4 x float>* %v18
  %v20 = extractelement <4 x float> %v19, i32 0
  %v21 = extractelement <4 x float> %v19, i32 1
  %v22 = extractelement <4 x float> %v19, i32 2
  %v23 = extractelement <4 x float> %v19, i32 3
  %v24 = fadd nsz arcp contract afn float %v10, %v20
  %v25 = fadd nsz arcp contract afn float %v12, %v21
  %v26 = fadd nsz arcp contract afn float %v14, %v22
  %v27 = fadd nsz arcp contract afn float %v16, %v23
  store float %v24, float* %v0
  store float %v25, float* %v1
  store float %v26, float* %v2
  store float %v27, float* %v3
  br label %loop_footer_0
loop_footer_0:
  br label %loop_latch_0
loop_exit_0:
  %v33 = getelementptr inbounds float, float* %reg_0, i32 0
  %v34 = load float, float* %v33
  %v35 = getelementptr inbounds float, float* %reg_0, i32 1
  %v36 = load float, float* %v35
  %v37 = getelementptr inbounds float, float* %reg_0, i32 2
  %v38 = load float, float* %v37
  %v39 = getelementptr inbounds float, float* %reg_0, i32 3
  %v40 = load float, float* %v39
  %v41 = getelementptr inbounds float, float* %data0, i32 0
  %v42 = fadd nsz arcp contract afn float %v34, %v36
  %v43 = fadd nsz arcp contract afn float %v42, %v38
  %v44 = fadd nsz arcp contract afn float %v43, %v40
  store float %v44, float* %v41
  ret void
}
attributes #0 = { alwaysinline nounwind "no-builtins" "no-trapping-math"="true" }