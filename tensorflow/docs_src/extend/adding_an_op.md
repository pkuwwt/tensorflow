# 添加一个新操作（Op）

注意：默认情况下，[TensorFlow 官网](http://tensorflow.org) 显示的是最新稳定版的文档。
本文档中的用法说明针对的是从源码构建的 TensorFlow。你很可能希望从 TensorFlow 的 `master` 版本开始构建。
那么，你就应该采纳[本文档的 `master` 版本](https://www.tensorflow.org/versions/master/extend/adding_an_op)
中的用法，两种版本之间是有可能存在变动的。

如果你想要创建一个在已有 TensorFlow 库中不存在的操作，我们建议你先从 Python 入手，即写一个已有 Python 操作或函数的复合操作。
如果这样不可行，你就需要创建一个定制的 C++ 操作了。下面是你可能需要这样做的一些理由：

*   将你的操作表示成复合操作不太容易或不可能。
*   已有基本操作的复合操作效率不高。
*   你想手工实现一些基本操作的复合，因为未来的编译器做这种融合可能会比较困难。

比如，假设你希望实现类似于“最大值池化（MaxPool）”的“中值池化”操作，只不过不再是计算最大值，而是在滑动窗口上计算中值。
这种操作是可能用已有操作复合得到的，比如使用 ExtractImagePatches 和 TopK，但是这可能在性能上、或内存开销上不如原生操作，
因为你可以在单一的融合操作中采用一些高明的策略。大体上，首先尝试用复合操作来实现你的想法总是值得一试的，只有当复合操作很困难或
低效时才考虑添加一个新的操作。

为了加入一个定制操作，你需要做如下工作：

1.  在一个 C++ 文件中注册这个新操作。操作的注册为此操作的功能定义了一个接口（规范）。
    比如，操作的注册定义了此操作的名称和它的输入输出。它还定义了 shape 函数，用于获取张量的形状。
2.  用 C++ 实现这个操作。一个操作的实现又被称为一个内核，它是你在上一步注册的规范的一个具体实现。
    对于不同的输入输出类型或架构（比如不同的 CPU 或 GPU），可能要实现不同的内核。
3.  创建一个 Python 包装器（可选）。这个包装器为此操作的公共 API，用于在 Python 中创建此操作。
    从操作的注册中可以产生一个默认的包装器，它可以直接使用，或加入。
4.  编写一个函数来计算此操作的梯度（可选）。
5.  测试此操作。为方便起见，我们通常在 Python 中测试，但也你可以在 C++ 中测试。如果你定义了梯度，
    你可以用 Python 的 @{tf.test.compute_gradient_error$梯度检查器} 来验证。参见脚本 
     [`relu_op_test.py`](https://www.tensorflow.org/code/tensorflow/python/kernel_tests/relu_op_test.py)，
     它提供了一个例子，展示如何测试类似于 Relu 的算子的前向函数及梯度。

编写新操作代码前，你需要：

*   熟悉 C++
*   必须安装有 @{$install$TensorFlow 二进制代码}，或必须下载有 @{$install_sources$TensorFlow 源码}，并能够构建

[TOC]

## 定义操作接口

操作接口的定义是通过在 TensorFlow 系统中注册来实现的。在此注册过程中，需要指定操作名称、输入（类型和名称）、
输出（类型和名称），以及文档字符串和此操作要求的任何[属性](#属性)。

下面展示注册的具体过程。假设你想创建一个操作，其输入是一个 `int32` 类型的张量，而输出是此张量的一个副本，
副本除第一个元素设为零之外其它都不变。为实现这样一个操作，我们先创建一个名称 `zero_out.cc` 的文件。然后调用 
`REGISTER_OP` 这个宏，用它来定义你的操作：

```c++
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

于是，我们注册了一个名为 `ZeroOut` 的操作，它的输入（命名为 `to_zero`）和输出（命名为 `zeroed`）都是 32 比特整数类型的张量。
此操作利用一个形状函数来确保输出张量的形状与输入张量保持一致。比如，如果输入张量的形状为 [10, 20]，则此形状函数将输出张量的形状也
指定为 [10, 20]。

>   关于命名的备注：操作名称必须首字母大写，而且不能和库中已经注册的其它操作重名。

## 实现操作的内核

定义了接口之后，接下来就需要为此操作提供一个或多个内核实现了。
为了实现这些内核，创建一个继承自 `OpKernel` 的类，并重载 `Compute` 方法。 
`Compute` 方法有一个类型为 `OpKernelContext*` 的参数 `context`，根据这个参数可以访问到一些有用信息，
比如输入输出张量。

将你的内核加到上面创建的文件中。这个内核的代码形如：

```c++
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 得到输入张量
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // 创建输出张量
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // 除第一个元素外，输出张量的其它所有元素都设置为 0 
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // 如果可能的话，保留第一个输入值
    if (N > 0) output_flat(0) = input(0);
  }
};
```

实现完内核之后，将其注册到 TensorFlow 系统中。在注册中，你还要指定此内核运行时不同约束条件。
比如，你可能有一个内核是针对 CPU 的，而还有一个是针对 GPU 的。

为了给 `ZeroOut` 操作加上约束条件，将下面的代码加到 `zero_out.cc` 文件中：

```c++
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

>   重要提示：你的 OpKernel 子类的实例有可能会被并发访问，所以 `Compute` 方法必须是线程安全的。
>   可以用线程互斥锁来保护类成员的每一次访问。更好的办法是，不要通过类成员来共享状态！
>   可以考虑使用一个 [`ResourceMgr`](https://www.tensorflow.org/code/tensorflow/core/framework/resource_mgr.h)
>   来跟踪操作的状态。

### 多线程 CPU 内核

为了编写一个多线程 CPU 内核，可使用́
[`work_sharder.h`](https://www.tensorflow.org/code/tensorflow/core/util/work_sharder.h)
中的 Shared 函数。在 intro-op 线程模式下，此函数让不同的线程共享同一个计算函数（参见 
[`config.proto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto) 中定义的 intra_op_parallelism_threads  模式）。

### GPU 内核

一个 GPU 内核的实现包括两个部分：OpKernel 子类、CUDA 内核及其启动代码。

有时候 OpKernel 实现可由 CPU 和 GPU 内核共享，这一部分代码可以完成诸如检查输入和创建输出之类的任务。
如果采用这种方案，则我们建议用如下实现方式：

1. 为适应多种设备，要定义模板化的 OpKernel，并定义张量的基本类型
2. 为了实际计算输出， Compute 函数要调用一个模板化的函子结构
3. 此函子针对 CPU 设备（CPUDevice）的模板特化可在同一个文件中定义，但针对 GPU 设备（GPUDevice）的模板特化要单独定义在一个 .cu.cc 文件中，因为它需要用 CUDA 编译器来编译。

下面是一个实现的示例：

```c++
// kernel_example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct ExampleFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif KERNEL_EXAMPLE_H_
```

```c++
// kernel_example.cc
#include "example.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// 实际计算的 CPU 模板特化
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};

// OpKernel 子类的定义
// 模板参数 <T> 为张量的数据类型
template <typename Device, typename T>
class ExampleOp : public OpKernel {
 public:
  explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 获得输入张量
    const Tensor& input_tensor = context->input(0);

    // 创建输出张量
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // 计算
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    ExampleFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// 注册 CPU 上的内核́́
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExampleOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// 注册 GPU 上的内核
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* 在 kernel_example.cu.cc 中显式声明模板实例化 */ \
  extern template ExampleFunctor<GPUDevice, float>;              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
```

```c++
// kernel_example.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "example.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// 定义 CUDA 内核́
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * ldg(in + i);
  }
}

// 定义启动 CUDA 内核的 GPU 实现
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  // 启动 CUDA 内核
  //
  // 参见 core/util/cuda_kernel_helper.h 中的计算线程块数目和每块线程数（thread_per_block）的示例
  int block_count = 1024;
  int thread_per_block = 20;
  ExampleCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// 显式示例化函子，这些函子用于处理注册的那些 OpKernel 支持的类型
template struct ExampleFunctor<GPUDevice, float>;
template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
```

## 构建操作的库文件
### 用系统编译器来编译操作（TensorFlow 二进制安装）

你可以用 `C++` 编译器来编译 `zero_out.cc`，比如你的系统上的 `g++` 或 `clang` 都是可以的。
用 PIP 包管理器来安装二进制 TensorFlow 时，已经包含了编译操作所需的头文件和库文件，具体的安装目录则取决于你的操作系统。
不过，TensorFlow 的 python 库提供了 `get_include` 函数来获得头文件目录位置，也提供了 `get_lib` 函数来获得链接所需
库文件的目录位置。

下面是 Ubuntu 机器上这两个函数的输出结果：

```bash
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python2.7/site-packages/tensorflow/include'
>>> tf.sysconfig.get_lib()
'/usr/local/lib/python2.7/site-packages/tensorflow'
```

假如你的系统上安装了 `g++`，下面的命令可于将你的操作编译成一个动态库。

```bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
```

在 Mac OS X 上，构建 `.so` 文件时还需要额外的编译标志 "-undefined dynamic_lookup" 。

>   注意，如果 `gcc` 版本 `>=5`，则 gcc 使用的新的 C++ [ABC](https://gcc.gnu.org/gcc-5/changes.html#libstdcxx)。
>   TensorFlow 官网上提供的二进制 pip 包用的是 `gcc4`，即它用的旧的 ABI。如果你用 `gcc>=5` 来编译你的操作库文件，
>   在命令行中加入 `-D_GLIBCXX_USE_CXX11_ABI=0` 来让生成的库文件与旧的 ABI 兼容。
>   此外，如果你在用从源码构建的 TensorFlow ，记得在用 bazel 命令编译 Python 包时中加上编译选项 `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"`。

### 使用 bazel 编译操作（TensorFlow 源码安装）

如果你有 TensorFlow 源码，则你可以利用 TensorFLow 的构建系统来编译你的操作。把一个 BUILD 文件放在 
[`tensorflow/core/user_ops`][user_ops] 目录中，文件中为 Bazel 的构建规则，内容如下：

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```

运行下列命令来构建 `zero_out.so`.

```bash
$ bazel build --config opt //tensorflow/core/user_ops:zero_out.so
```

>   注意：虽然你可以用标准 `cc_library` 规则来生成一个共享库文件（`.so` 文件），
>   我们还是强烈推荐使用 `tf_custom_op_library` 宏。这个宏加了一些必要的依赖项，
>   而且还包含一些检查，以确保输出的共享库文件与 TensorFlow 的插件加载机制兼容。

## 在 Python 中使用新的操作

TensorFlow Python API 提供了 @{tf.load_op_library} 函数来加载动态链接库，并将其注册到 TensorFlow 框架中。
`load_op_library` 返回一个 Python 模块，其中就包含了你的新操作的 Python 包装器，以及它的内核。
因而，一旦你构建完操作，你就可以按下面的方式中在 Python 中让它运行起来了：

```python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
with tf.Session(''):
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

# 打印
array([[1, 0], [0, 0]], dtype=int32)
```

需要注意，生成的函数采用蛇形命令规则（snake\_case），这是为了遵守 [PEP8](https://www.python.org/dev/peps/pep-0008/) 规范。
所以，如果你的操作在 C++ 代码中命名为 `ZeroOut`，则它的 Python 函数名会变成 `zero_out`。

为了让该操作可以像常规函数一样从某个模块中导入（`import`），则可以在 Python 源码中调用 `load_op_library` 函数：

```python
import tensorflow as tf

zero_out_module = tf.load_op_library('./zero_out.so')
zero_out = zero_out_module.zero_out
```

## 确认操作是可行的
确认你编写的操作是否可成功运行的一个好办法是写一个测试。创建文件 `zero_out_op_test.py`，内容如下：

```python
import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('./zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.test.main()
```

然后，运行该测试（假设你已经安装了 TensorFlow）：

```sh
$ python zero_out_op_test.py
```

## 在操作中加入高级功能

现在你已经知道如何实现和构建一个基本的操作（更恰当地说，是一个受限的操作），那么接下来，我们将介绍你在编写新操作时通常会用到的一些更复杂的功能，包括：

*   [条件检查和验证](#conditional_checks_and_validation)
*   [操作注册](#op_registration)
    *   [属性](#attrs)
    *   [属性类型](#attr_types)
    *   [多态](#polymorphism)
    *   [输入输出](#inputs_and_outputs)
    *   [后向兼容](#backwards_compatibility)
*   [GPU 支持](#gpu_support)
    *   [为 GPU 设备编译内核](#compiling_the_kernel_for_the_gpu_device)
*   [在 Python 中实现梯度计算](#implement_the_gradient_in_python)
*   [C++ 中的形状函数](#shape_functions_in_c)

### 条件检查和验证

上述示例假定操作的输入是任意形状的张量。但如果我们只处理矢量呢？那么我们就需要在 OpKernel 的实现中加入一个检查：

```c++
  void Compute(OpKernelContext* context) override {
    // 获得输入张量
    const Tensor& input_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("ZeroOut expects a 1-D vector."));
    // ...
  }
```

这里我们加了一个断言，它要求输入是一个矢量，否则将设置 `InvalidArgument` 状态。
[`OP_REQUIRES` 宏][validation-macros] 有三个参数：

*   上下文 `context`：既可以是一个 `OpKernelContext`，也可以是一个 `OpKernelConstruction` 指针（参见
    [`tensorflow/core/framework/op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_kernel.h) 文件），宏
    会调用它们的的 `SetStatus()` 方法。
*   条件：关于检查形状的更多的函数，参见文件
    [`tensorflow/core/framework/tensor_shape.h`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.h)
*   错误本身：它由一个 `Status` 对象表示，参见文件
    [`tensorflow/core/lib/core/status.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/status.h)。
    一个 `Status` 对象包含一个类型（常为 `InvalidArgument`，参见更多类型的列表）和一个消息。构建一个错误的函数参见文件
    [`tensorflow/core/lib/core/errors.h`][validation-macros]。

另外，如果你想测试从某个函数返回的一个 `Status` 对象是否为错误，是的话就将错误返回，
这时你可使用宏 [`OP_REQUIRES_OK`][validation-macros]。
这两个宏都会在错误报错时返回错误对象。

### 操作的注册

#### 属性

操作可以有属性，当一个操作被加到计算图中时，它的属性就会被赋值。这些属性用于配置此操作，它们的值既可以在内核实现中访问到，
也可以在操作注册时的输入输出类型中访问到。相较于输入，参数的使用要尽量避免，因为输入更为灵活一些。这是因为属性是常数，
必须在计算图构造时定义。 而输入作为张量，它的值是动态的；即输入的值在每一步都可以修改，比如使用 feed_dict。
属性主要用于无法使用输入的场合：任何会影响到操作的特征（输入输出的数目和类型）的时候，或无法在每一步修改的时候。

你需要在注册操作时定义属性，定义时要指定名称和使用 `Attr` 方法的类型，此方法的参数规范如下：

```
<name>: <attr-type-expr>
```

其中 `<name>` 必须由字母开头，后面可以是字母、数字或下划线，而 `<attr-type-expr>` 一个类型表达式（参见[下方](#attr_types)）。

比如，如果你想让 `ZeroOut` 操作保留一个用户指定的下标，而不仅仅是第 0 个元素，你可以按下面的方式来注册操作：

<pre class="prettyprint"><code class="lang-cpp">
REGISTER\_OP("ZeroOut")
    <b>.Attr("preserve\_index: int")</b>
    .Input("to\_zero: int32")
    .Output("zeroed: int32");
</code></pre>

（注意，[属性类型](#attr_types)与输入输出的@{tf.DType$张量类型}是不一样的。）

你实现的内核可以在构造函数中通过 `context` 参数来访问属性：
<pre class="prettyprint"><code class="lang-cpp">
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction\* context) : OpKernel(context) {<b>
    // 获取待保存的下标值
    OP\_REQUIRES\_OK(context,
                   context-&gt;GetAttr("preserve\_index", &preserve\_index\_));
    // 检查 preserve\_index 是否为正值
    OP\_REQUIRES(context, preserve\_index_ &gt;= 0,
                errors::InvalidArgument("Need preserve\_index &gt;= 0, got ",
                                        preserve\_index_));
  </b>}
  void Compute(OpKernelContext\* context) override {
    // ...
  }
 <b>private:
  int preserve\_index\_;</b>
};
</code></pre>

还可以在 `Compute` 方法中访问到这个参数：
<pre class="prettyprint"><code class="lang-cpp">
  void Compute(OpKernelContext\* context) override {
    // ...
<br/>
    <b>// 我们用保存的属性来检查动态输入的合法性
    // 所以，我们检查 preserve\_index 是否在允许的值域范围内
    OP\_REQUIRES(context, preserve\_index_ &lt; input.dimension(0),
                errors::InvalidArgument("preserve\_index out of range"));<br/>
    </b>// 将输出张量中所有元素设置为 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output\_flat(i) = 0;
    }<br/>
    <b>// 保存指定位置的输入值
    output\_flat(preserve\_index\_) = input(preserve\_index\_);</b>
  }
</code></pre>

#### 属性类型

属性支持下列数据类型：

* `string`：任意字节序列（不要求是 UTF8 编码）
* `int`：有符号整数
* `float`: 浮点数
* `bool`: True 或 false
* `type`： [`DataType`][DataTypeString] 的其中一个（非引用）值
* `shape`：一个 [`TensorShapeProto`][TensorShapeProto]
* `tensor`：一个 [`TensorProto`][TensorProto]
* `list(<type>)`： `<type>` 的列表，其中 `<type>` 为其中一种上述类型
  注意： `list(list(<type>))` 是非法的。

欲了解限定性列表，参见 [`op_def_builder.cc:FinalizeAttr`][FinalizeAttr]。

##### 默认值和约束

属性可以有默认值，有一些属性则还可以有约束。为了定义一个有约束的属性，可以使用下列属性类型表达式（`<attr-type-expr>`）：

* `{'<string1>', '<string2>'}`：表示在 `<string1>` 或 `<string2>` 这两种取值中二选一。
当你使用这种语法时，系统自动推断出属性类型为 `string`。这相当于模仿构造了一个枚举：

  ```c++
  REGISTER_OP("EnumExample")
      .Attr("e: {'apple', 'orange'}");
  ```

* `{<type1>, <type2>}`: 属性类型为 `type`，表示取值是 `<type1>` 类型或 `<type2>` 类型二者之一，
  其中 `<type1>` 和 `<type2>` 为两种@{tf.DType$张量类型}。同样，你也不需要指定属性类型为 `type`，
  因为这个信息是可以从 `{...}` 这个张量类型列表推断出来的。比如，下面的例子中属性 `t` 必须是 `int32`、
  `float` 或 `bool` 中的一种类型：

  ```c++
  REGISTER_OP("RestrictedTypeExample")
      .Attr("t: {int32, float, bool}");
  ```

* 常用的类型约束可以有如下别名：
    * `numbertype`：`type` 类型被限制为数值类型（不是字符串，也不是布尔类型）
    * `realnumbertype`：类似于 `numbertype` 类型，但不包括复数类型
    * `quantizedtype`：类型与 `numbertype` 类型，但只包括量化数值类型
    
    属性所支持的类型列表可通过 [`tensorflow/core/framework/types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/types.h) 中的一些函数来定义（比如 `NumberTypes()`）。
    
    在本例中，属性 `t` 必须是下面一种数值类型：

    ```c++
    REGISTER_OP("NumberType")
        .Attr("t: numbertype");
    ```

    对于这个操作：

    ```python
    tf.number_type(t=tf.int32)  # 合法
    tf.number_type(t=tf.bool)   # 不合法
    ```

    列表可以和其他列表及单个类型组合起来。下面的操作允许属性 `t` 为任意数值类型或布尔类型：

    ```c++
    REGISTER_OP("NumberOrBooleanType")
        .Attr("t: {numbertype, bool}");
    ```

    对于这个操作：

    ```python
    tf.number_or_boolean_type(t=tf.int32)  # 合法
    tf.number_or_boolean_type(t=tf.bool)   # 合法
    tf.number_or_boolean_type(t=tf.string) # 不合法
    ```

* `int >= <n>`：取值必须是整型，且要求大于等于 `<n>`，其中 `<n>` 是一个自然数。

  比如，下列操作注册中，指定了属性 `a` 必须为一个至少为 `2` 的值：

  ```c++
  REGISTER_OP("MinIntExample")
      .Attr("a: int >= 2");
  ```

* `list(<type>) >= <n>`: 取值为`<type>` 类型的一个列表，其长度大于等于 `<n>`。

  比如，下列操作注册指定属性 `a` 是一个类型列表（要么是 `int32`，要么是 `float`），且要求长度大于等于 `3`：

  ```c++
  REGISTER_OP("TypeListExample")
      .Attr("a: list({int32, float}) >= 3");
  ```

为设置一个属性的默认值（让它在生成代码中成为可选项），可以在最后加上 `= <default>`，如下面代码所示：

```c++
REGISTER_OP("AttrDefaultExample")
    .Attr("i: int = 0");
```

这种默认值语法正是计算图的 GraphDef 定义的协议缓存表达中所用的语法。

下面的示例展示如何为所有类型指定默认值：

```c++
REGISTER_OP("AttrDefaultExampleForAllTypes")
   .Attr("s: string = 'foo'")
   .Attr("i: int = 0")
   .Attr("f: float = 1.0")
   .Attr("b: bool = true")
   .Attr("ty: type = DT_INT32")
   .Attr("sh: shape = { dim { size: 1 } dim { size: 2 } }")
   .Attr("te: tensor = { dtype: DT_INT32 int_val: 5 }")
   .Attr("l_empty: list(int) = []")
   .Attr("l_int: list(int) = [2, 3, 5, 7]");
```

注意：若值类型为 `type`，则使用 @{tf.DType$类型的 `DT_*` 名称}。

#### 多态

##### 类型多态

有些操作支持不同类型的输入或产生不同类型的输出，这时你可以在此操作的注册中为[一个输入或输出类型](#输入和输出)指定[一个属性](#属性)。
典型情况下，你还要为支持的每种类型注册一个 `OpKernel`。

比如，如果你想让 `ZeroOut` 操作既支持 `int32` 数值类型的张量，还要支持 `float` 类型，那么此操作的注册过程将类似于：
<pre class="prettyprint"><code class="lang-cpp">
REGISTER\_OP("ZeroOut")
    <b>.Attr("T: {float, int32}")</b>
    .Input("to\_zero: <b>T</b>")
    .Output("zeroed: <b>T</b>");
</code></pre>

现在，此操作在注册中指定了输入类型必须是 `float` 或 `int32`，而它的输出类型将保持一致，因为都是 `T` 类型。

> <a id="naming"></a> 关于命名的备注：输入、输出和属性一般都应该使用蛇形命名。
> 不过有一个例外情况，那就是属性被用作输入类型、或用于输入类型时。这样的属性会在操作被加入到计算图中自动推断出来，
> 即它们不会在操作的函数中出现。比如，ZeroOut 最终的定义将产生一个如下的 Python 函数：
>
> ```python
> def zero_out(to_zero, name=None):
>   """...
>   参数：
>     to_zero: 表示一个 `Tensor`。必须两种类型之一： `float32`、 `int32`。
>     name: 操作的名称（可选）
>
>   返回值：
>     一个 `Tensor`，与 `to_zero` 类型相同
>   """
> ```
>
> 如果 `to_zero` 中传入一个 `int32` 张量，则 `T` 自动被设置为 `int32` （实际上是 `DT_INT32`）。
> 这时推断出来的属性的命名方式为首字母大小或单词首字母大写。
>
> 与这种情况不同的是，有时候我们需要为用一个类型属性来为操作指定输出类型：
>
> ```c++
> REGISTER_OP("StringToNumber")
>     .Input("string_tensor: string")
>     .Output("output: out_type")
>     .Attr("out_type: {float, int32} = DT_FLOAT");
>     .Doc(R"doc(
> 将输入张量中的每个字符串转换为指定的数值类型。
> )doc");
> ```
>
> 这时，用户需要指定输出类型，这也反映到了生成的 Python 代码中，如下所示：
>
> ```python
> def string_to_number(string_tensor, out_type=None, name=None):
>   """将输入张量中的每个字符串转换为指定的数值类型。
>
>   参数：
>     string_tensor: `string` 类型的一个 `Tensor`
>     out_type: 可选的 `tf.DType`，即 `tf.float32` 和 `tf.int32` 二者之一，默认为 `tf.float32`。
>     name: 操作名称（可选）
>
>   返回值：
>     类型为 `out_type` 的一个 `Tensor`
>   """
> ```

<pre class="prettyprint"><code class="lang-cpp">
\#include "tensorflow/core/framework/op_kernel.h"<br/>
class ZeroOut<b>Int32</b>Op : public OpKernel {
  // 和前面一样
};<br/>
class ZeroOut<b>Float</b>Op : public OpKernel {
 public:
  explicit ZeroOut<b>Float</b>Op(OpKernelConstruction\* context)
      : OpKernel(context) {}<br/>
  void Compute(OpKernelContext\* context) override {
    // 获得输入张量
    const Tensor& input\_tensor = context-&gt;input(0);
    auto input = input\_tensor.flat&lt;<b>float</b>&gt;();<br/>
    // 产生输出张量
    Tensor* output = NULL;
    OP\_REQUIRES\_OK(context,
                   context-&gt;allocate\_output(0, input_tensor.shape(), &output));
    auto output\_flat = output-&gt;template flat&lt;<b>float</b>&gt;();<br/>
    // 将输出张量中的所有元素设置为 0
    const int N = input.size();
    for (int i = 0; i &lt; N; i++) {
      output\_flat(i) = 0;
    }<br/>
    // 保留第一个输入值́
    if (N &gt; 0) output\_flat(0) = input(0);
  }
};<br/><b>
// 注意：TypeConstraint&lt;int32&gt;("T") 表示属性 `T` （定义在操作注册代码中）必须是 `int32` 类型的，
// 即将模板实例化了。</b>
REGISTER\_KERNEL\_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE\_CPU)
    <b>.TypeConstraint&lt;int32&gt;("T"),</b>
    ZeroOutOp<b>Int32</b>);
<b>REGISTER\_KERNEL\_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE\_CPU)
    .TypeConstraint&lt;float&gt;("T"),
    ZeroOutFloatOp);
</b></code></pre>

> 为了[后向兼容](#后向兼容)，在将属性加到已有操作中时，你需要指定一个[默认值](#默认值约束)：
>
> <pre class="prettyprint"><code class="lang-cpp">
> REGISTER\_OP("ZeroOut")
>   <b>.Attr("T: {float, int32} = DT_INT32")</b>
>   .Input("to\_zero: T")
>   .Output("zeroed: T")
> </code></pre>

如果你还想添加更多类型，比如说 `double` 类型，你要稍微修改一下注册代码：
<pre class="prettyprint"><code class="lang-cpp">
REGISTER\_OP("ZeroOut")
    <b>.Attr("T: {float, <b>double,</b> int32}")</b>
    .Input("to\_zero: <b>T</b>")
    .Output("zeroed: <b>T</b>");
</code></pre>

为了避免像上面的代码一样为多个 `OpKernel` 编写冗余代码，你可以使用 C++ 模板。
不过，你仍然需要为每一次加载注册一个内核（调用 `REGISTER_KERNEL_BUILDER`）。
<pre class="prettyprint"><code class="lang-cpp">
<b>template &lt;typename T&gt;</b>
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction\* context) : OpKernel(context) {}<br/>
  void Compute(OpKernelContext\* context) override {
    // 获得输入张量
    const Tensor& input\_tensor = context-&gt;input(0);
    auto input = input\_tensor.flat<b>&lt;T&gt;</b>();<br/>
    // 产生输出张量
    Tensor* output = NULL;
    OP\_REQUIRES\_OK(context,
                   context-&gt;allocate\_output(0, input_tensor.shape(), &output));
    auto output\_flat = output-&gt;template flat<b>&lt;T&gt;</b>();<br/>
    // 将输出张量中的所有元素设置为 0
    const int N = input.size();
    for (int i = 0; i &lt; N; i++) {
      output\_flat(i) = 0;
    }<br/>
    // 保留第一个输入值́
    if (N &gt; 0) output\_flat(0) = input(0);
  }
};<br/>
// 注意：TypeConstraint&lt;int32&gt;("T") 表示属性 `T` （定义在操作注册代码中）必须是 `int32` 类型的，
// 即将模板实例化了。</b>
REGISTER\_KERNEL\_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE\_CPU)
    .TypeConstraint&lt;int32&gt;("T"),
    <b>ZeroOutOp&lt;int32&gt;</b>);
REGISTER\_KERNEL\_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE\_CPU)
    .TypeConstraint&lt;float&gt;("T"),
    <b>ZeroOutOp&lt;float&gt;</b>);
<b>REGISTER\_KERNEL\_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE\_CPU)
    .TypeConstraint&lt;double&gt;("T"),
    ZeroOutOp&lt;double&gt;);
</b></code></pre>

如果加载次数还不少，那你可以借助于一个宏来简化代码。

```c++
#include "tensorflow/core/framework/op_kernel.h"

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
```

根据你为内核注册的类型列表的不同，你还可以使用 [`tensorflow/core/framework/register_types.h`][register_types] 中提供的宏：

```c++
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

REGISTER_OP("ZeroOut")
    .Attr("T: realnumbertype")
    .Input("to_zero: T")
    .Output("zeroed: T");

template <typename T>
class ZeroOutOp : public OpKernel { ... };

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
```

##### 列表作为输入输出

除了能够接受或产生不同类型，操作还消耗或产生数目不一的张量。

在下一个例子中，属性 `T` 指的是类型 `type` 的一个*列表*，它被用作输入 `in` 和输出 `out` 的类型。即输入和输出都某个类型的张量列表（而且输入输出列表的大小和类型都是完全一样的，因为它们由是类型 `T`）。

```c++
REGISTER_OP("PolymorphicListExample")
    .Attr("T: list(type)")
    .Input("in: T")
    .Output("out: T");
```

你也可以对列表中元素的类型施加限制。在下一个例子中，输入是 `float` 或 `double` 类型张量的列表。比如，若输入类型是 `(float, double, float)`，而输出类型也必须是 `(float, double, float)`。

```c++
REGISTER_OP("ListTypeRestrictionExample")
    .Attr("T: list({float, double})")
    .Input("in: T")
    .Output("out: T");
```

如果你要求列表中所有张量的类型都相同，则你可以这样：

```c++
REGISTER_OP("IntListInputExample")
    .Attr("N: int")
    .Input("in: N * int32")
    .Output("out: int32");
```

此例中，输入是 `int32` 类型的张量的列表，其中 `int` 属性 `N` 用来指定此列表的长度。

我们也可以实现 [类型多态](#类型多态)。在下一个示例中，输入是长度为 `N` 的张量列表，这些张量的类型为 `T`（但还没指定），而输出则为指定类型的单个张量：

```c++
REGISTER_OP("SameListInputExample")
    .Attr("N: int")
    .Attr("T: type")
    .Input("in: N * T")
    .Output("out: T");
```

默认情况下，张量列表的长度至少为 1。你可以用 [相应属性上的 `">="` 约束](#默认值约束) 来修改默认值。在下一个示例中，输入是长度至少为 2 的 `int32` 张量列表：

```c++
REGISTER_OP("MinLengthIntListExample")
    .Attr("N: int >= 2")
    .Input("in: N * int32")
    .Output("out: int32");
```

同样的语法也可以用到 `"list(type)"` 类型的属性上：

```c++
REGISTER_OP("MinimumLengthPolymorphicListExample")
    .Attr("T: list(type) >= 3")
    .Input("in: T")
    .Output("out: T");
```

#### 输入和输出

下面对前面的示例做个总结，一个操作注册可以指定多个输入输出：

```c++
REGISTER_OP("MultipleInsAndOuts")
    .Input("y: int32")
    .Input("z: float")
    .Output("a: string")
    .Output("b: int32");
```

每个输入或输出的规范的格式如下：

```
<name>: <io-type-expr>
```

其中 `<name>` 的首字符必须是字母，后面的字符可以是字母、数字或下划线。`<io-type-expr>` 是下列表达式之一：

* `<type>`：支持的输入类型，比如 `float`、`int32`、`string`。这个表达式指定了 `type` 类型的单个张量。

  参见 @{tf.DType$支持的张量类型列表}。

  ```c++
  REGISTER_OP("BuiltInTypesExample")
      .Input("integers: int32")
      .Input("complex_numbers: complex64");
  ```

* `<attr-type>`：一个[属性](#属性)的名称，此属性的类型可以是 `type` 或 `list(type)`（可以有类型限制）。这个语法可以实现[多态操作](#多态)。 

  ```c++
  REGISTER_OP("PolymorphicSingleInput")
      .Attr("T: type")
      .Input("in: T");

  REGISTER_OP("RestrictedPolymorphicSingleInput")
      .Attr("T: {int32, int64}")
      .Input("in: T");
  ```

  引用类型为 `list(type)` 的属性可以让你接受一个张量序列。

  ```c++
  REGISTER_OP("ArbitraryTensorSequenceExample")
      .Attr("T: list(type)")
      .Input("in: T")
      .Output("out: T");

  REGISTER_OP("RestrictedTensorSequenceExample")
      .Attr("T: list({int32, int64})")
      .Input("in: T")
      .Output("out: T");
  ```

  注意，输出 `out` 中的张量的类型和数目与输入 `in` 是一样的，因为它们都是 `T` 类型。

* 相同类型的张量序列：`<number> * <type>`, 其中 `<number>` 为类型为 `int` 的一个[属性](#属性)。`<type>` 可以是 @{tf.DType$诸如 `int32` 或 `float` 这样的特定类型} 或 类型为 `type` 的一个属性的名称。第一种情况中，操作可接受 `int32` 张量的列表，示例如下：

  ```c++
  REGISTER_OP("Int32SequenceExample")
      .Attr("NumTensors: int")
      .Input("in: NumTensors * int32")
  ```

  此操作接受任意类型的张量列表，只要它们的类型都一样：

  ```c++
  REGISTER_OP("SameTypeSequenceExample")
      .Attr("NumTensors: int")
      .Attr("T: type")
      .Input("in: NumTensors * T")
  ```

* 对单个张量的引用：`Ref(<type>)`，其中 `<type>` 是上述类型中的一种。

> 关于命名的备注：输入的类型中用到的任何属性都会被推断出来。按惯例，这些被推断的属性名要首字线大写（比如 `T` 或 `N`）。其它情况下，输入、输出和属性的名称和函数参数命名方式一致，比如 `num_outputs`。更多细节，参考 [前面关于命名的备注](#命名)。

更多细节，参考 [`tensorflow/core/framework/op_def_builder.h`][op_def_builder]。

#### 后向兼容性

Let's assume you have written a nice, custom op and shared it with others, so
you have happy customers using your operation.  However, you'd like to make
changes to the op in some way.

In general, changes to existing, checked-in specifications must be
backwards-compatible: changing the specification of an op must not break prior
serialized `GraphDef` protocol buffers constructed from older specifications.
The details of `GraphDef` compatibility are
@{$version_compat#compatibility_of_graphs_and_checkpoints$described here}.

There are several ways to preserve backwards-compatibility.

1. Any new attrs added to an operation must have default values defined, and
   with that default value the op must have the original behavior. To change an
   operation from not polymorphic to polymorphic, you *must* give a default
   value to the new type attr to preserve the original signature by default. For
   example, if your operation was:

       REGISTER_OP("MyGeneralUnaryOp")
           .Input("in: float")
           .Output("out: float");

   you can make it polymorphic in a backwards-compatible way using:

       REGISTER_OP("MyGeneralUnaryOp")
           .Input("in: T")
           .Output("out: T")
           .Attr("T: numerictype = DT_FLOAT");

2. You can safely make a constraint on an attr less restrictive.  For example,
   you can change from `{int32, int64}` to `{int32, int64, float}` or `type`.
   Or you may change from `{"apple", "orange"}` to `{"apple", "banana",
   "orange"}` or `string`.

3. You can change single inputs / outputs into list inputs / outputs, as long as
   the default for the list type matches the old signature.

4. You can add a new list input / output, if it defaults to empty.

5. Namespace any new ops you create, by prefixing the op names with something
   unique to your project. This avoids having your op colliding with any ops
   that might be included in future versions of TensorFlow.

6. Plan ahead! Try to anticipate future uses for the op. Some signature changes
   can't be done in a compatible way (for example, making a list of the same
   type into a list of varying types).

The full list of safe and unsafe changes can be found in
[`tensorflow/core/framework/op_compatibility_test.cc`](https://www.tensorflow.org/code/tensorflow/core/framework/op_compatibility_test.cc).
If you cannot make your change to an operation backwards compatible, then create
a new operation with a new name with the new semantics.

Also note that while these changes can maintain `GraphDef` compatibility, the
generated Python code may change in a way that isn't compatible with old
callers.  The Python API may be kept compatible by careful changes in a
hand-written Python wrapper, by keeping the old signature except possibly adding
new optional arguments to the end.  Generally incompatible changes may only be
made when TensorFlow's changes major versions, and must conform to the
@{$version_compat#compatibility_of_graphs_and_checkpoints$`GraphDef` version semantics}.

### GPU 支持

You can implement different OpKernels and register one for CPU and another for
GPU, just like you can [register kernels for different types](#polymorphism).
There are several examples of kernels with GPU support in
[`tensorflow/core/kernels/`](https://www.tensorflow.org/code/tensorflow/core/kernels/).
Notice some kernels have a CPU version in a `.cc` file, a GPU version in a file
ending in `_gpu.cu.cc`, and some code shared in common in a `.h` file.

For example, the @{tf.pad} has
everything but the GPU kernel in [`tensorflow/core/kernels/pad_op.cc`][pad_op].
The GPU kernel is in
[`tensorflow/core/kernels/pad_op_gpu.cu.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op_gpu.cu.cc),
and the shared code is a templated class defined in
[`tensorflow/core/kernels/pad_op.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.h).
We organize the code this way for two reasons: it allows you to share common
code among the CPU and GPU implementations, and it puts the GPU implementation
into a separate file so that it can be compiled only by the GPU compiler.

One thing to note, even when the GPU kernel version of `pad` is used, it still
needs its `"paddings"` input in CPU memory.  To mark that inputs or outputs are
kept on the CPU, add a `HostMemory()` call to the kernel registration, e.g.:

```c++
#define REGISTER_GPU_KERNEL(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Pad")                  \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .HostMemory("paddings"), \
                          PadOp<GPUDevice, T>)
```

#### 为 GPU 设备编译内核

Look at
[cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc)
for an example that uses a CUDA kernel to implement an op. The
`tf_custom_op_library` accepts a `gpu_srcs` argument in which the list of source
files containing the CUDA kernels (`*.cu.cc` files) can be specified. For use
with a binary installation of TensorFlow, the CUDA kernels have to be compiled
with NVIDIA's `nvcc` compiler. Here is the sequence of commands you can use to
compile the
[cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc)
and
[cuda_op_kernel.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cc)
into a single dynamically loadable library:

```bash
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
cuda_op_kernel.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L$TF_LIB -ltensorflow_framework
```

`cuda_op_kernel.so` produced above can be loaded as usual in Python, using the
`tf.load_op_library` function.

Note that if your CUDA libraries are not installed in `/usr/local/lib64`,
you'll need to specify the path explicitly in the second (g++) command above.
For example, add `-L /usr/local/cuda-8.0/lib64/` if your CUDA is installed in
`/usr/local/cuda-8.0`.

>   Note in some linux settings, additional options to `nvcc` compiling step are needed. Add `-D_MWAITXINTRIN_H_INCLUDED` to the `nvcc` command line to avoid errors from `mwaitxintrin.h`.

### 在 Python 中实现梯度计算

Given a graph of ops, TensorFlow uses automatic differentiation
(backpropagation) to add new ops representing gradients with respect to the
existing ops (see
@{$python/train#gradient_computation$Gradient Computation}).
To make automatic differentiation work for new ops, you must register a gradient
function which computes gradients with respect to the ops' inputs given
gradients with respect to the ops' outputs.

Mathematically, if an op computes \\(y = f(x)\\) the registered gradient op
converts gradients \\(\partial L/ \partial y\\) of loss \\(L\\) with respect to
\\(y\\) into gradients \\(\partial L/ \partial x\\) with respect to \\(x\\) via
the chain rule:

$$\frac{\partial L}{\partial x}
    = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}
    = \frac{\partial L}{\partial y} \frac{\partial f}{\partial x}.$$

In the case of `ZeroOut`, only one entry in the input affects the output, so the
gradient with respect to the input is a sparse "one hot" tensor.  This is
expressed as follows:

```python
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
  """The gradients for `zero_out`.

  Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  to_zero = op.inputs[0]
  shape = array_ops.shape(to_zero)
  index = array_ops.zeros_like(shape)
  first_grad = array_ops.reshape(grad, [-1])[0]
  to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
  return [to_zero_grad]  # List of one Tensor, since we have one input
```

Details about registering gradient functions with
@{tf.RegisterGradient}:

* For an op with one output, the gradient function will take an
  @{tf.Operation} `op` and a
  @{tf.Tensor} `grad` and build new ops
  out of the tensors
  [`op.inputs[i]`](../../api_docs/python/framework.md#Operation.inputs),
  [`op.outputs[i]`](../../api_docs/python/framework.md#Operation.outputs), and `grad`.  Information
  about any attrs can be found via
  @{tf.Operation.get_attr}.

* If the op has multiple outputs, the gradient function will take `op` and
  `grads`, where `grads` is a list of gradients with respect to each output.
  The result of the gradient function must be a list of `Tensor` objects
  representing the gradients with respect to each input.

* If there is no well-defined gradient for some input, such as for integer
  inputs used as indices, the corresponding returned gradient should be
  `None`.  For example, for an op taking a floating point tensor `x` and an
  integer index `i`, the gradient function would `return [x_grad, None]`.

* If there is no meaningful gradient for the op at all, you often will not have
  to register any gradient, and as long as the op's gradient is never needed,
  you will be fine. In some cases, an op has no well-defined gradient but can
  be involved in the computation of the gradient. Here you can use
  `ops.NotDifferentiable` to automatically propagate zeros backwards.

Note that at the time the gradient function is called, only the data flow graph
of ops is available, not the tensor data itself.  Thus, all computation must be
performed using other tensorflow ops, to be run at graph execution time.

### C++ 中的形状函数

The TensorFlow API has a feature called "shape inference" that provides
information about the shapes of tensors without having to execute the
graph. Shape inference is supported by "shape functions" that are registered for
each op type in the C++ `REGISTER_OP` declaration, and perform two roles:
asserting that the shapes of the inputs are compatible during graph
construction, and specifying the shapes for the outputs.

Shape functions are defined as operations on the
`shape_inference::InferenceContext` class. For example, in the shape function
for ZeroOut:

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

`c->set_output(0, c->input(0));` declares that the first output's shape should
be set to the first input's shape. If the output is selected by its index as in the above example, the second parameter of `set_output` should be a `ShapeHandle` object. You can create an empty `ShapeHandle` object by its default constructor. The `ShapeHandle` object for an input with index `idx` can be obtained by `c->input(idx)`.

There are a number of common shape functions
that apply to many ops, such as `shape_inference::UnchangedShape` which can be
found in [common_shape_fns.h](https://www.tensorflow.org/code/tensorflow/core/framework/common_shape_fns.h) and used as follows:

```c++
REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
```

A shape function can also constrain the shape of an input. For the version of
[`ZeroOut` with a vector shape constraint](#validation), the shape function
would be as follows:

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      c->set_output(0, input);
      return Status::OK();
    });
```

The `WithRank` call validates that the input shape `c->input(0)` has
a shape with exactly one dimension (or if the input shape is unknown,
the output shape will be a vector with one unknown dimension).

If your op is [polymorphic with multiple inputs](#polymorphism), you can use
members of `InferenceContext` to determine the number of shapes to check, and
`Merge` to validate that the shapes are all compatible (alternatively, access
attributes that indicate the lengths, with `InferenceContext::GetAttr`, which
provides access to the attributes of the op).

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ::tensorflow::shape_inference::ShapeHandle output;
      for (size_t i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 2, &input));
        TF_RETURN_IF_ERROR(c->Merge(output, input, &output));
      }
      c->set_output(0, output);
      return Status::OK();
    });
```

Since shape inference is an optional feature, and the shapes of tensors may vary
dynamically, shape functions must be robust to incomplete shape information for
any of the inputs. The `Merge` method in [`InferenceContext`](https://www.tensorflow.org/code/tensorflow/core/framework/shape_inference.h)
allows the caller to assert that two shapes are the same, even if either
or both of them do not have complete information. Shape functions are defined
for all of the core TensorFlow ops and provide many different usage examples.

The `InferenceContext` class has a number of functions that can be used to
define shape function manipulations.  For example, you can validate that a
particular dimension has a very specific value using `InferenceContext::Dim` and
`InferenceContext::WithValue`; you can specify that an output dimension is the
sum / product of two input dimensions using `InferenceContext::Add` and
`InferenceContext::Multiply`. See the `InferenceContext` class for
all of the various shape manipulations you can specify. The following example sets
shape of the first output to (n, 3), where first input has shape (n, ...)

```c++
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 3));
    return Status::OK();
});
```

If you have a complicated shape function, you should consider adding a test for
validating that various input shape combinations produce the expected output
shape combinations.  You can see examples of how to write these tests in some
our
[core ops tests](https://www.tensorflow.org/code/tensorflow/core/ops/array_ops_test.cc).
(The syntax of `INFER_OK` and `INFER_ERROR` are a little cryptic, but try to be
compact in representing input and output shape specifications in tests.  For
now, see the surrounding comments in those tests to get a sense of the shape
string specification).


[core-array_ops]:https://www.tensorflow.org/code/tensorflow/core/ops/array_ops.cc
[python-user_ops]:https://www.tensorflow.org/code/tensorflow/python/user_ops/user_ops.py
[tf-kernels]:https://www.tensorflow.org/code/tensorflow/core/kernels/
[user_ops]:https://www.tensorflow.org/code/tensorflow/core/user_ops/
[pad_op]:https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.cc
[standard_ops-py]:https://www.tensorflow.org/code/tensorflow/python/ops/standard_ops.py
[standard_ops-cc]:https://www.tensorflow.org/code/tensorflow/cc/ops/standard_ops.h
[python-BUILD]:https://www.tensorflow.org/code/tensorflow/python/BUILD
[validation-macros]:https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h
[op_def_builder]:https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.h
[register_types]:https://www.tensorflow.org/code/tensorflow/core/framework/register_types.h
[FinalizeAttr]:https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.cc
[DataTypeString]:https://www.tensorflow.org/code/tensorflow/core/framework/types.cc
[python-BUILD]:https://www.tensorflow.org/code/tensorflow/python/BUILD
[types-proto]:https://www.tensorflow.org/code/tensorflow/core/framework/types.proto
[TensorShapeProto]:https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.proto
[TensorProto]:https://www.tensorflow.org/code/tensorflow/core/framework/tensor.proto
