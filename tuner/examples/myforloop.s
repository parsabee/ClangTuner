	.text
	.file	"myforloop.cpp"
	.globl	_Z7forloopPiS_S_                # -- Begin function _Z7forloopPiS_S_
	.p2align	4, 0x90
	.type	_Z7forloopPiS_S_,@function
_Z7forloopPiS_S_:                       # @_Z7forloopPiS_S_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -32(%rbp)
	movq	%rsi, -24(%rbp)
	movq	%rdx, -16(%rbp)
	movl	$0, -4(%rbp)
.LBB0_1:                                # %for.cond
                                        # =>This Inner Loop Header: Depth=1
	cmpl	$256, -4(%rbp)                  # imm = 0x100
	jge	.LBB0_4
# %bb.2:                                # %for.body
                                        #   in Loop: Header=BB0_1 Depth=1
	movq	-32(%rbp), %rax
	movslq	-4(%rbp), %rcx
	movl	(%rax,%rcx,4), %eax
	movq	-24(%rbp), %rcx
	movslq	-4(%rbp), %rdx
	addl	(%rcx,%rdx,4), %eax
	movq	-16(%rbp), %rcx
	movslq	-4(%rbp), %rdx
	movl	%eax, (%rcx,%rdx,4)
# %bb.3:                                # %for.inc
                                        #   in Loop: Header=BB0_1 Depth=1
	movl	-4(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -4(%rbp)
	jmp	.LBB0_1
.LBB0_4:                                # %for.end
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	_Z7forloopPiS_S_, .Lfunc_end0-_Z7forloopPiS_S_
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 13.0.0 (https://github.com/parsabee/llvm-project.git 87afefcd22c53f7bdc68b5a13492e7f2bfc9837a)"
	.section	".note.GNU-stack","",@progbits
