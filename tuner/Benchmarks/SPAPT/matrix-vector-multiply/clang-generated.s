	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 15
	.globl	__Z12mat_vec_multPA256_fPfS1_   ## -- Begin function _Z12mat_vec_multPA256_fPfS1_
	.p2align	4, 0x90
__Z12mat_vec_multPA256_fPfS1_:          ## @_Z12mat_vec_multPA256_fPfS1_
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	$0, -32(%rbp)
LBB0_1:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_3 Depth 2
	cmpq	$256, -32(%rbp)                 ## imm = 0x100
	jae	LBB0_8
## %bb.2:                               ##   in Loop: Header=BB0_1 Depth=1
	movq	$0, -40(%rbp)
LBB0_3:                                 ##   Parent Loop BB0_1 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	cmpq	$256, -40(%rbp)                 ## imm = 0x100
	jae	LBB0_6
## %bb.4:                               ##   in Loop: Header=BB0_3 Depth=2
	movq	-8(%rbp), %rax
	movq	-32(%rbp), %rcx
	shlq	$10, %rcx
	addq	%rcx, %rax
	movq	-40(%rbp), %rcx
	movss	(%rax,%rcx,4), %xmm0            ## xmm0 = mem[0],zero,zero,zero
	movq	-16(%rbp), %rax
	movq	-40(%rbp), %rcx
	mulss	(%rax,%rcx,4), %xmm0
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rcx
	addss	(%rax,%rcx,4), %xmm0
	movss	%xmm0, (%rax,%rcx,4)
## %bb.5:                               ##   in Loop: Header=BB0_3 Depth=2
	movq	-40(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -40(%rbp)
	jmp	LBB0_3
LBB0_6:                                 ##   in Loop: Header=BB0_1 Depth=1
	jmp	LBB0_7
LBB0_7:                                 ##   in Loop: Header=BB0_1 Depth=1
	movq	-32(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -32(%rbp)
	jmp	LBB0_1
LBB0_8:
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	_main                           ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$264224, %rsp                   ## imm = 0x40820
	movq	___stack_chk_guard@GOTPCREL(%rip), %rax
	movq	(%rax), %rax
	movq	%rax, -8(%rbp)
	movl	$0, -264212(%rbp)
	leaq	-262160(%rbp), %rdi
	callq	__Z19initializeRandom_2DIfLm256ELm256EEvPAT1__T_
	leaq	-263184(%rbp), %rdi
	callq	__Z19initializeRandom_1DIfLm256EEvPT_
	leaq	-264208(%rbp), %rdi
	xorps	%xmm0, %xmm0
	callq	__Z13initialize_1DIfLm256EEvPT_S0_
	leaq	-262160(%rbp), %rdi
	leaq	-263184(%rbp), %rsi
	leaq	-264208(%rbp), %rdx
	callq	__Z12mat_vec_multPA256_fPfS1_
	leaq	-262160(%rbp), %rdi
	leaq	-263184(%rbp), %rsi
	leaq	-264208(%rbp), %rdx
	callq	__Z6verifyIfLm256ELm256EEbPAT1__T_PS0_S3_
	testb	$1, %al
	jne	LBB1_2
## %bb.1:
	movl	$1, -264212(%rbp)
	jmp	LBB1_3
LBB1_2:
	movl	$0, -264212(%rbp)
LBB1_3:
	movl	-264212(%rbp), %eax
	movl	%eax, -264216(%rbp)             ## 4-byte Spill
	movq	___stack_chk_guard@GOTPCREL(%rip), %rax
	movq	(%rax), %rax
	movq	-8(%rbp), %rcx
	cmpq	%rcx, %rax
	jne	LBB1_5
## %bb.4:
	movl	-264216(%rbp), %eax             ## 4-byte Reload
	addq	$264224, %rsp                   ## imm = 0x40820
	popq	%rbp
	retq
LBB1_5:
	callq	___stack_chk_fail
	ud2
	.cfi_endproc
                                        ## -- End function
	.globl	__Z19initializeRandom_2DIfLm256ELm256EEvPAT1__T_ ## -- Begin function _Z19initializeRandom_2DIfLm256ELm256EEvPAT1__T_
	.weak_definition	__Z19initializeRandom_2DIfLm256ELm256EEvPAT1__T_
	.p2align	4, 0x90
__Z19initializeRandom_2DIfLm256ELm256EEvPAT1__T_: ## @_Z19initializeRandom_2DIfLm256ELm256EEvPAT1__T_
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	xorl	%eax, %eax
	movl	%eax, %edi
	callq	_time
	movl	%eax, %edi
	callq	_srand
	movq	$0, -16(%rbp)
LBB2_1:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB2_3 Depth 2
	cmpq	$256, -16(%rbp)                 ## imm = 0x100
	jae	LBB2_8
## %bb.2:                               ##   in Loop: Header=BB2_1 Depth=1
	movq	$0, -24(%rbp)
LBB2_3:                                 ##   Parent Loop BB2_1 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	cmpq	$256, -24(%rbp)                 ## imm = 0x100
	jae	LBB2_6
## %bb.4:                               ##   in Loop: Header=BB2_3 Depth=2
	callq	_rand
	cvtsi2ss	%eax, %xmm0
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rcx
	shlq	$10, %rcx
	addq	%rcx, %rax
	movq	-24(%rbp), %rcx
	movss	%xmm0, (%rax,%rcx,4)
## %bb.5:                               ##   in Loop: Header=BB2_3 Depth=2
	movq	-24(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -24(%rbp)
	jmp	LBB2_3
LBB2_6:                                 ##   in Loop: Header=BB2_1 Depth=1
	jmp	LBB2_7
LBB2_7:                                 ##   in Loop: Header=BB2_1 Depth=1
	movq	-16(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -16(%rbp)
	jmp	LBB2_1
LBB2_8:
	addq	$32, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	__Z19initializeRandom_1DIfLm256EEvPT_ ## -- Begin function _Z19initializeRandom_1DIfLm256EEvPT_
	.weak_definition	__Z19initializeRandom_1DIfLm256EEvPT_
	.p2align	4, 0x90
__Z19initializeRandom_1DIfLm256EEvPT_:  ## @_Z19initializeRandom_1DIfLm256EEvPT_
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	xorl	%eax, %eax
	movl	%eax, %edi
	callq	_time
	movl	%eax, %edi
	callq	_srand
	movq	$0, -16(%rbp)
LBB3_1:                                 ## =>This Inner Loop Header: Depth=1
	cmpq	$256, -16(%rbp)                 ## imm = 0x100
	jae	LBB3_4
## %bb.2:                               ##   in Loop: Header=BB3_1 Depth=1
	callq	_rand
	cvtsi2ss	%eax, %xmm0
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rcx
	movss	%xmm0, (%rax,%rcx,4)
## %bb.3:                               ##   in Loop: Header=BB3_1 Depth=1
	movq	-16(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -16(%rbp)
	jmp	LBB3_1
LBB3_4:
	addq	$16, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	__Z13initialize_1DIfLm256EEvPT_S0_ ## -- Begin function _Z13initialize_1DIfLm256EEvPT_S0_
	.weak_definition	__Z13initialize_1DIfLm256EEvPT_S0_
	.p2align	4, 0x90
__Z13initialize_1DIfLm256EEvPT_S0_:     ## @_Z13initialize_1DIfLm256EEvPT_S0_
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movss	%xmm0, -12(%rbp)
	movq	$0, -24(%rbp)
LBB4_1:                                 ## =>This Inner Loop Header: Depth=1
	cmpq	$256, -24(%rbp)                 ## imm = 0x100
	jae	LBB4_4
## %bb.2:                               ##   in Loop: Header=BB4_1 Depth=1
	movss	-12(%rbp), %xmm0                ## xmm0 = mem[0],zero,zero,zero
	movq	-8(%rbp), %rax
	movq	-24(%rbp), %rcx
	movss	%xmm0, (%rax,%rcx,4)
## %bb.3:                               ##   in Loop: Header=BB4_1 Depth=1
	movq	-24(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -24(%rbp)
	jmp	LBB4_1
LBB4_4:
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	__Z6verifyIfLm256ELm256EEbPAT1__T_PS0_S3_ ## -- Begin function _Z6verifyIfLm256ELm256EEbPAT1__T_PS0_S3_
	.weak_definition	__Z6verifyIfLm256ELm256EEbPAT1__T_PS0_S3_
	.p2align	4, 0x90
__Z6verifyIfLm256ELm256EEbPAT1__T_PS0_S3_: ## @_Z6verifyIfLm256ELm256EEbPAT1__T_PS0_S3_
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$1120, %rsp                     ## imm = 0x460
	movq	___stack_chk_guard@GOTPCREL(%rip), %rax
	movq	(%rax), %rax
	movq	%rax, -8(%rbp)
	movq	%rdi, -1056(%rbp)
	movq	%rsi, -1064(%rbp)
	movq	%rdx, -1072(%rbp)
	movq	$0, -1080(%rbp)
LBB5_1:                                 ## =>This Inner Loop Header: Depth=1
	cmpq	$256, -1080(%rbp)               ## imm = 0x100
	jae	LBB5_4
## %bb.2:                               ##   in Loop: Header=BB5_1 Depth=1
	movq	-1080(%rbp), %rax
	xorps	%xmm0, %xmm0
	movss	%xmm0, -1040(%rbp,%rax,4)
## %bb.3:                               ##   in Loop: Header=BB5_1 Depth=1
	movq	-1080(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -1080(%rbp)
	jmp	LBB5_1
LBB5_4:
	movq	$0, -1088(%rbp)
LBB5_5:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB5_7 Depth 2
	cmpq	$256, -1088(%rbp)               ## imm = 0x100
	jae	LBB5_12
## %bb.6:                               ##   in Loop: Header=BB5_5 Depth=1
	movq	$0, -1096(%rbp)
LBB5_7:                                 ##   Parent Loop BB5_5 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	cmpq	$256, -1096(%rbp)               ## imm = 0x100
	jae	LBB5_10
## %bb.8:                               ##   in Loop: Header=BB5_7 Depth=2
	movq	-1056(%rbp), %rax
	movq	-1088(%rbp), %rcx
	shlq	$10, %rcx
	addq	%rcx, %rax
	movq	-1096(%rbp), %rcx
	movss	(%rax,%rcx,4), %xmm0            ## xmm0 = mem[0],zero,zero,zero
	movq	-1064(%rbp), %rax
	movq	-1096(%rbp), %rcx
	mulss	(%rax,%rcx,4), %xmm0
	movq	-1088(%rbp), %rax
	addss	-1040(%rbp,%rax,4), %xmm0
	movss	%xmm0, -1040(%rbp,%rax,4)
## %bb.9:                               ##   in Loop: Header=BB5_7 Depth=2
	movq	-1096(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -1096(%rbp)
	jmp	LBB5_7
LBB5_10:                                ##   in Loop: Header=BB5_5 Depth=1
	jmp	LBB5_11
LBB5_11:                                ##   in Loop: Header=BB5_5 Depth=1
	movq	-1088(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -1088(%rbp)
	jmp	LBB5_5
LBB5_12:
	movq	$0, -1104(%rbp)
LBB5_13:                                ## =>This Inner Loop Header: Depth=1
	cmpq	$256, -1104(%rbp)               ## imm = 0x100
	jae	LBB5_18
## %bb.14:                              ##   in Loop: Header=BB5_13 Depth=1
	movq	-1104(%rbp), %rax
	movss	-1040(%rbp,%rax,4), %xmm0       ## xmm0 = mem[0],zero,zero,zero
	movq	-1072(%rbp), %rax
	movq	-1104(%rbp), %rcx
	ucomiss	(%rax,%rcx,4), %xmm0
	jne	LBB5_15
	jp	LBB5_15
	jmp	LBB5_16
LBB5_15:
	movb	$0, -1041(%rbp)
	jmp	LBB5_19
LBB5_16:                                ##   in Loop: Header=BB5_13 Depth=1
	jmp	LBB5_17
LBB5_17:                                ##   in Loop: Header=BB5_13 Depth=1
	movq	-1104(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -1104(%rbp)
	jmp	LBB5_13
LBB5_18:
	movb	$1, -1041(%rbp)
LBB5_19:
	movb	-1041(%rbp), %al
	movb	%al, -1105(%rbp)                ## 1-byte Spill
	movq	___stack_chk_guard@GOTPCREL(%rip), %rax
	movq	(%rax), %rax
	movq	-8(%rbp), %rcx
	cmpq	%rcx, %rax
	jne	LBB5_21
## %bb.20:
	movb	-1105(%rbp), %al                ## 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
	addq	$1120, %rsp                     ## imm = 0x460
	popq	%rbp
	retq
LBB5_21:
	callq	___stack_chk_fail
	ud2
	.cfi_endproc
                                        ## -- End function
.subsections_via_symbols
