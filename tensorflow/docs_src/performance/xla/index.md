# XLA 概述

> 注意： XLA 暂时是实验性的，仍处于 alpha 版本。而且大部分用例并不会带来
> 性能（提高速度或减少内存使用量）上的提高。我们这么早就发布了 XLA，这样的话，
> 开源社区就可以为它的开发贡献力量了，而且有助于走出一条与硬件加速器整合之路。

XLA （加速线性代数）是一种线性代数的专用编译器，可用于优化 TensorFlow 的计算。
其目标旨在提高速度、内存使用量以及对服务器和移动平台的可移植性。一开始，大
部分用户将不会从 XLA 中得到太大的好处，但是我们欢迎大家通过 @{$jit$just-in-time (JIT) 编译} 或 @{$tfcompile$ahead-of-time (AOT) 编译} 
来使用 XLA 做实验。特别是那些专注于新硬件加速器的开发者，尤其应该试一试 XLA。

XLA 框架是实验性的，且处于活跃的开发状态。因而，虽然已有操作的语义不太可能发生改变，
但 XLA 不同，可以想见 XLA 中会不断加入更多操作，以覆盖更多重要的用例。XLA 的开发团队
欢迎来自于社区的任何反馈，包括缺失的功能，以及通过 GitHub 提交的社区贡献。

## 我们为什么推出 XLA？

让 TensorFlow 用上 XLA，我们追求多个目标：

*   *改进执行速度*： 对子图进行编译，以减少短时操作的执行时间，进而消除 TensorFlow 运行时相关的开销；
    融合管道化的操作以减少内存开销；针对已知张量形状优化，以支持更积极的常数传播。

*   *改进内存使用*： 分析内存使用并调度，原则上可消除很多临时的缓存。

*   *减少对定制操作的依赖*： 通过提高底层操作自动融合的性能，让其和定制操作中的手工融合一样高效，从而消除很多定制操作的必要性。

*   *减少移动足迹*： 提前编译子图，并生成一对文件（对象/头文件），它们可以直接编译到另一个应用程序中，从而消除 TensorFlow 运行时。
    这样做的结果是移动推理的足迹会减少数个数量级。

*   *改善可移植性*： 为新硬件编写新的后端会先对容易一些，因为大部分 TensorFlow 程序不需要怎么修改就可以在新硬件上跑了。
    这和专门为新硬件定制一体化操作形成对比，因为那样做的话 TensorFlow 程序需要重写才能用这些新的操作。

## How does XLA work?

The input language to XLA is called "HLO IR", or just HLO (High Level
Optimizer). The semantics of HLO are described on the
@{$operation_semantics$Operation Semantics} page. It
is most convenient to think of HLO as a [compiler
IR](https://en.wikipedia.org/wiki/Intermediate_representation).

XLA takes graphs ("computations") defined in HLO and compiles them into machine
instructions for various architectures. XLA is modular in the sense that it is
easy to slot in an alternative backend to @{$developing_new_backend$target some novel HW architecture}. The CPU backend for x64 and ARM64 as
well as the NVIDIA GPU backend are in the TensorFlow source tree.

The following diagram shows the compilation process in XLA:

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img src="https://www.tensorflow.org/images/how-does-xla-work.png">
</div>

XLA comes with several optimizations and analyzes that are target-independent,
such as [CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination),
target-independent operation fusion, and buffer analysis for allocating runtime
memory for the computation.

After the target-independent step, XLA sends the HLO computation to a backend.
The backend can perform further HLO-level analyzes and optimizations, this time
with target specific information and needs in mind. For example, the XLA GPU
backend may perform operation fusion beneficial specifically for the GPU
programming model and determine how to partition the computation into streams.
At this stage, backends may also pattern-match certain operations or
combinations thereof to optimized library calls.

The next step is target-specific code generation. The CPU and GPU backends
included with XLA use [LLVM](http://llvm.org) for low-level IR, optimization,
and code-generation. These backends emit the LLVM IR necessary to represent the
XLA HLO computation in an efficient manner, and then invoke LLVM to emit native
code from this LLVM IR.

The GPU backend currently supports NVIDIA GPUs via the LLVM NVPTX backend; the
CPU backend supports multiple CPU ISAs.

## Supported Platforms

XLA currently supports @{$jit$JIT compilation} on x86-64 and NVIDIA GPUs; and
@{$tfcompile$AOT compilation} for x86-64 and ARM.
