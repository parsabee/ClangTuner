	.text
	.file	"LLVMDialectModule"
	.globl	forloop                         # -- Begin function forloop
	.p2align	4, 0x90
	.type	forloop,@function
forloop:                                # @forloop
	.cfi_startproc
# %bb.0:
	movq	8(%rsp), %rax
	movq	48(%rsp), %rcx
	xorl	%edx, %edx
	cmpq	$255, %rdx
	jg	.LBB0_3
	.p2align	4, 0x90
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	movl	(%rsi,%rdx,4), %edi
	addl	(%rax,%rdx,4), %edi
	movl	%edi, (%rcx,%rdx,4)
	incq	%rdx
	cmpq	$255, %rdx
	jle	.LBB0_2
.LBB0_3:
	retq
.Lfunc_end0:
	.size	forloop, .Lfunc_end0-forloop
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
