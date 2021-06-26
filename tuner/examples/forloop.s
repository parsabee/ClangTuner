	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 15
	.globl	___forloop_input_cc_13_3        ## -- Begin function __forloop_input_cc_13_3
	.p2align	4, 0x90
___forloop_input_cc_13_3:               ## @__forloop_input_cc_13_3
	.cfi_startproc
## %bb.0:
	movq	8(%rsp), %rax
	movq	48(%rsp), %rcx
	xorl	%edx, %edx
	cmpq	$255, %rdx
	jg	LBB0_3
	.p2align	4, 0x90
LBB0_2:                                 ## =>This Inner Loop Header: Depth=1
	movl	(%rsi,%rdx,4), %edi
	addl	(%rax,%rdx,4), %edi
	movl	%edi, (%rcx,%rdx,4)
	incq	%rdx
	cmpq	$255, %rdx
	jle	LBB0_2
LBB0_3:
	retq
	.cfi_endproc
                                        ## -- End function
.subsections_via_symbols
