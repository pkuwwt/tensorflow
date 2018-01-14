# 性能指南

本指南包含了一些优化 TensorFlow 代码的最佳实践，它包含以下几节内容：

*   [一般性最佳实践](#一般性最佳实践) 涵盖多种模型类型和硬件的通用主题。
*   [GPU 上的优化](#GPU-上的优化) 针对 GPU 的相关技巧的细节。
*   [CPU 上的优化](#CPU-上的优化) 针对 CPU 的细节。

## 一般性最佳实践

下面的几节内容为涵盖多种硬件和模型的最佳实践，它们是：

*   [输入管线的优化](#输入管线的优化)
*   [数据格式](#数据格式)
*   [通用的融合操作](#通用的融合操作)
*   [从源码构建和安装](#从源码构建和安装)

### 输入管线的优化

典型的模型会从磁盘加载数据，然后处理并发送到网络中。比如，模型按照下列数据流过程来处理 JPEG 图像：
从磁盘加载图像，将 JPEG 解码加载到一个张量中，裁剪和边缘垫值，以及可能的翻转的变形，然后按批次投入训练。
这个数据流被称为输入管线。随着 GPU 和其它加速硬件运行得越来越快，数据预处理就成了性能的瓶颈。

确定输入管线是否为瓶颈可能会比较复杂。一种最直接的方法是让输入管线后的那个模型只包含单个操作（得到一个平凡模型），然后测量其每秒处理的样例数。
如果整个模型和平凡模型之间的效率差异极小，则输入管线很有可能是一个瓶颈。下面是发现瓶颈问题的其它一些方法：

*   通过运行 `nvidia-smi -l 2` 来检查一个 GPU 是否已经在使用。如果 GPU 利用率没有接近 80-100%，则此输入管线可能是个瓶颈。
*   生成一个时间线，并检查它是否有大块的空白时间段（等待时间）。生成时间线的示例参见教程 @{$jit$XLA JIT}。
*   检查 CPU 使用情况。有可能出现的情况是：管线已经优化，却仍然没有足够的 CPU 时钟来处理这个管线。
*   估计所需的吞吐量，确认磁盘可以应付这样规模的吞吐量。一些云服务网络提供的磁盘速度甚至低到 50 MB/秒，这比机械磁盘（150 MB/秒）、SATA SSD （500 MB/秒）、以及 PCIe SSD （2000+ MB/秒）都要慢。


#### CPU 上的预处理

将输入管线的操作放在 CPU 上可以显著提高性能。让 CPU 处理输入管线，可以解放 GPU，让它专注于训练。为了确保预处理是在 CPU 上进行，可将预处理操作按如下方式包装一下：

```python
with tf.device('/cpu:0'):
  # 用于获得和处理图像或数据的函数
  distorted_inputs = load_and_distort_images()
```

如果使用 `tf.estimator.Estimator`，输入函数会自动用 CPU 执行。

#### 使用 Dataset API

在构建输入管线时，我们推荐使用 @{$datasets$Dataset API}，而不是原来的 `queue_runner`。
这个 API 出现在 TensorFlow 1.2 的 contrib 模块中，未来会被加到核心代码中。
[ResNet 示例](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator/cifar10_main.py)
（来自于论文 [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)）中训练 CIFAR-10 展示了如何结合 `tf.estimator.Estimator` 来使用 Dataset API。Dataset API利用了 C++ 多线程，且比基于 Python 的 `queue_runner` 具有更小的开销，后者受 Python 的多线程性能所累。

虽然用一个 `feed_dict` 字典来输入数据非常灵活，但在大部分例子中，使用 `feed_dict` 并不能很好地扩展。
不过，如果只用到了一个 GPU，这并没有什么影响。即便如此，我们也强烈推荐使用 Dataset API。下面的用法应尽量避免： 

```python
# 如果输入数据量大，feed_dict 通常导致次优的性能
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

#### 使用大文件
加载大量的小文件会极大地影响 I/O 性能。一种获得最大的 I/O 吞吐量的方法是将输入数据预处理为更大的 `TFRecord` 文件（约 100MB 大小）。
对于较小的数据集（200MB~1GB），最好的方法通常是将整个数据集加载到内存。资料[下载和转换为 TFRecord 格式](https://github.com/tensorflow/models/tree/master/research/slim#Data) 中介绍了创建 `TFRecords` 的相关信息和脚本，
而[脚本](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py) 
可用于将 CIFAR-10 数据集转化为 `TFRecords`。

### 数据格式

数据格式指的是传递给指定操作的张量结构。下面的讨论专门针对表示图像的四维张量。在 TensorFlow 中，
四维张量中的部分成员常用如下一些字母来表示：

*   N 表示一个训练批次中的图像数目
*   H 表示垂直维度（高度方向）中的像元数目
*   W 表示水平维度（宽度方向）中的像元数目
*   C 表示通道数。比如，1 表示黑白或灰度图像，而 3 表示 RGB 图像。

在 TensorFlow 中，有两种命名规范分别表示最常用的两种数据格式：

*   `NCHW` 或 `channels_first`
*   `NHWC` 或 `channels_last`


TensorFlow默认采用 `NHWC`，而在 NVIDIA GPU 上使用 [cuDNN](https://developer.nvidia.com/cudnn) 时，`NCHW` 格式是最优选择。

实践中最好的方式是让你的模型同时支持这两种数据格式。这可以让你在 GPU 上训练完了之后直接将模型用于 CPU 上的推理。
如果 TensorFlow 编译时用了 [Intel MKL](#tensorflow_with_intel_mkl-dnn) 优化，很多操作会被优化并支持 `NCHW`，特别是基于 CNN 的模型相关的操作。如果你没有使用 MKL，有些操作在使用 `NCHW` 时无法在 CPU 上运行。

这里我们简要介绍一下这两种格式的历史。TensorFlow 最开始使用 `NHWC` 是因为它在 CPU 上稍微快一点。
但长期以来，我们一直在编写工具，让计算图可以自动重写，从而让两种格式的切换变得透明化，来实现一些优化。
我们发现，尽管 `NCHW` 在一般情况下效率是最高的，但有些 GPU 操作在使用 `NHWC` 时确实更快一些。

### 通用融合操作

融合操作是将多个操作合并为单个内核，从而提高性能。TensorFlow 自带了大量的融合操作，而且 @{$xla$XLA} 
会尽可能地创建融合操作，来自动地提高性能。下面，我们将挑选出一些融合操作，这些操作可以极大地提高性能，但往往会被忽视。

#### 融合批量标准化

融合批量标准化（Fused batch norm）是将批量标准化所需的多个操作合并为一个内核。批量标准化是一个开销很大的过程，对于一些模型而言，它会占用很大比例的操作时间。通过使用融合批量标准化，可以实现 12%-30% 的加速。

常用的批量标准化有两种，都支持融合。TensorFlow 1.3 版本中开始支持对核心函数 @{tf.layers.batch_normalization} 添加融合支持。

```python
bn = tf.layers.batch_normalization(
    input_layer, fused=True, data_format='NCHW')
```

社区贡献（contrib）中的 @{tf.contrib.layers.batch_norm} 函数则从 TensorFlow 1.0 起就加入融合支持。

```python
bn = tf.contrib.layers.batch_norm(input_layer, fused=True, data_format='NCHW')
```

### 从源码构建和安装


默认情况下，TensorFlow 二进制程序已经覆盖了非常广泛的硬件种类，从而让每个人都能使用 TensorFlow。
如果用 CPU 来做训练或推理，建议编译 TensorFlow 时启用所有针对 CPU 的优化。对 CPU 上训练和推理的加速的文档
参见[编译器优化的对比](#编译器优化的对比)。

为安装 TensorFlow 的优化得最充分的版本，你需要从源码 @{$install_sources$构建和安装}。
如果需要在目标机器上构建支持不同硬件平台的 TensorFlow，你需要在交叉编译时针对目标平台启用最高级别的优化。
下面的命令展示了使用 `bazel` 针对特定平台进行编译的示例。

```python
# 此命令针对 Intel 的 Broadwell 处理器进行优化
bazel build -c opt --copt=-march="broadwell" --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

#### 环境、构建和安装技巧

*   `./configure` 命令是为了确定在构建中包含哪些计算能力。它不影响整体性能，但会影响初始启动。
    运行 TensorFlow 一次之后，编译的内核会被缓存到 CUDA 中。如果使用 docker 容器，这个数据将
    得不到缓存，因而每次 TensorFlow 启动时都会因此而变慢。最好的办法是将需要用到的 GPU 的[计算能力](http://developer.nvidia.com/cuda-gpus)
    包含进来，比如 P100 为 6.0，Titan X (Pascal) 为 6.1，Titan X (Maxwell) 为 5.2，K80 为 3.7。
*   选择一个版本的 gcc ，要求能够支持目标 CPU 能提供的所有优化。推荐的最低的 gcc 版本为 4.8.3。
    在 OS X 上，更新到最新的 Xcode 版本，并使用 Xcode 自带的那个版本的 clang。
*   安装 TensorFlow 能够支持的最新的稳定版 CUDA 平台和 cuDNN 库。

## GPU 上的优化

本节介绍针对 GPU 的优化技巧，这和[一般最佳实践](#一般最佳实践)中的内容不同。如何在多 GPU 环境下获得
最优的性能是一个有挑战性的任务。常用的方法是利用数据并行机制。基于数据并行的扩展需要将模型复制数份，它们被称之为“塔”，
然后将每个“塔”置于一个 GPU 上。每个塔会对一个不同批次的数据进行操作，然后更新变量。这些变量即我们所说的参数，是需要由
所有塔来共享的。那么每个塔是如何获得变量更新的？梯度计算又是如何影响模型的性能、扩展、以及收敛性的呢？
本节后面的部分将概述模型的塔在多个 GPU 上是如何处理那些变量的。@{$performance_models$高性能模型} 中则会更详细介绍一些更复杂的方法，用于在不同塔之间共享和更新变量。

如何最好地处理变量的更新与模型、硬件、以及硬件的配置方法等因素有关。比如，两个系统都用 NVIDIA Tesla P100s，
但是一个使用的是 PCIe 而另一个却是 [NVLink](http://www.nvidia.com/object/nvlink.html)。在这种情况下，
两者的最优方案就不可能不一样了。对于真实世界的例子，请参考 @{$performance/benchmarks$基准} 页面中关于多种平台上的最优设置的介绍。
我们对几个平台和配置进行了基准测试，下面是摘要：

*   **Tesla K80**： 如果多个 GPU 位于同一个 PCI Express 根联合体上，且相互之间能够使用 
    [NVIDIA GPUDirect](https://developer.nvidia.com/gpudirect) 技术相
    通信，则将变量均匀地分布在这些 GPU 上进行训练是最好的方法。如果不能使用 GPUDirect，则变量放在 CPU 上是最好的办法。

*   **Titan X (Maxwell 和 Pascal)、 M40、P100、及类似型号**： 对于像 ResNet 和 InceptionV3 这样的模型，将变量
    放在 CPU 上是最优选择，但是对于变量很多的模型，比如 AlexNet 和 VGG，结合 `NCCL` 使用 GPU 会更好一些。

A common approach to managing where variables are placed, is to create a method
to determine where each Op is to be placed and use that method in place of a
specific device name when calling `with tf.device():`. Consider a scenario where
a model is being trained on 2 GPUs and the variables are to be placed on the
CPU. There would be a loop for creating and placing the "towers" on each of the
2 GPUs. A custom device placement method would be created that watches for Ops
of type `Variable`, `VariableV2`, and `VarHandleOp` and indicates that they are
to be placed on the CPU. All other Ops would be placed on the target GPU.
The building of the graph would proceed as follows:

*   On the first loop a "tower" of the model would be created for `gpu:0`.
    During the placement of the Ops, the custom device placement method would
    indicate that variables are to be placed on `cpu:0` and all other Ops on
    `gpu:0`.

*   On the second loop, `reuse` is set to `True` to indicate that variables are
    to be reused and then the "tower" is created on `gpu:1`. During the
    placement of the Ops associated with the "tower", the variables that were
    placed on `cpu:0` are reused and all other Ops are created and placed on
    `gpu:1`.

The final result is all of the variables are placed on the CPU with each GPU
having a copy of all of the computational Ops associated with the model.

The code snippet below illustrates two different approaches for variable
placement: one is placing variables on the CPU; the other is placing variables
equally across the GPUs.

```python

class GpuParamServerDeviceSetter(object):
  """Used with tf.device() to place variables on the least loaded GPU.

    A common use for this class is to pass a list of GPU devices, e.g. ['gpu:0',
    'gpu:1','gpu:2'], as ps_devices.  When each variable is placed, it will be
    placed on the least loaded gpu. All other Ops, which will be the computation
    Ops, will be placed on the worker_device.
  """

  def __init__(self, worker_device, ps_devices):
    """Initializer for GpuParamServerDeviceSetter.
    Args:
      worker_device: the device to use for computation Ops.
      ps_devices: a list of devices to use for Variable Ops. Each variable is
      assigned to the least loaded device.
    """
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
      return self.worker_device

    # Gets the least loaded ps_device
    device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name

def _create_device_setter(is_cpu_ps, worker, num_gpus):
  """Create device setter object."""
  if is_cpu_ps:
    # tf.train.replica_device_setter supports placing variables on the CPU, all
    # on one GPU, or on ps_servers defined in a cluster_spec.
    return tf.train.replica_device_setter(
        worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
  else:
    gpus = ['/gpu:%d' % i for i in range(num_gpus)]
    return ParamServerDeviceSetter(worker, gpus)

# The method below is a modified snippet from the full example.
def _resnet_model_fn():
    # When set to False, variables are placed on the least loaded GPU. If set
    # to True, the variables will be placed on the CPU.
    is_cpu_ps = False

    # Loops over the number of GPUs and creates a copy ("tower") of the model on
    # each GPU.
    for i in range(num_gpus):
      worker = '/gpu:%d' % i
      # Creates a device setter used to determine where Ops are to be placed.
      device_setter = _create_device_setter(is_cpu_ps, worker, FLAGS.num_gpus)
      # Creates variables on the first loop.  On subsequent loops reuse is set
      # to True, which results in the "towers" sharing variables.
      with tf.variable_scope('resnet', reuse=bool(i != 0)):
        with tf.name_scope('tower_%d' % i) as name_scope:
          # tf.device calls the device_setter for each Op that is created.
          # device_setter returns the device the Op is to be placed on.
          with tf.device(device_setter):
            # Creates the "tower".
            _tower_fn(is_training, weight_decay, tower_features[i],
                      tower_labels[i], tower_losses, tower_gradvars,
                      tower_preds, False)

```

In the near future the above code will be for illustration purposes only as
there will be easy to use high level methods to support a wide range of popular
approaches. This
[example](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator)
will continue to get updated as the API expands and evolves to address multi-GPU
scenarios.

## CPU 上的优化

CPUs, which includes Intel® Xeon Phi™, achieve optimal performance when
TensorFlow is @{$install_sources$built from source} with all of the instructions
supported by the target CPU.

Beyond using the latest instruction sets, Intel® has added support for the
Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) to
TensorFlow. While the name is not completely accurate, these optimizations are
often simply referred to as 'MKL' or 'TensorFlow with MKL'. [TensorFlow
with Intel® MKL-DNN](#tensorflow_with_intel_mkl_dnn) contains details on the
MKL optimizations.

The two configurations listed below are used to optimize CPU performance by
adjusting the thread pools.

*   `intra_op_parallelism_threads`: Nodes that can use multiple threads to
    parallelize their execution will schedule the individual pieces into this
    pool.
*   `inter_op_parallelism_threads`: All ready nodes are scheduled in this pool.

These configurations are set via the `tf.ConfigProto` and passed to `tf.Session`
in the `config` attribute as shown in the snippet below.  For both configuration
options, if they are unset or set to 0, will default to the number of logical
CPU cores. Testing has shown that the default is effective for systems ranging
from one CPU with 4 cores to multiple CPUs with 70+ combined logical cores.
A common alternative optimization is to set the number of threads in both pools
equal to the number of physical cores rather than logical cores.

```python

  config = tf.ConfigProto()
  config.intra_op_parallelism_threads = 44
  config.inter_op_parallelism_threads = 44
  tf.session(config=config)

```

The [Comparing compiler optimizations](#comparing-compiler-optimizations)
section contains the results of tests that used different compiler
optimizations.

### 在 TensorFlow 中使用 Intel® MKL DNN

Intel® has added optimizations to TensorFlow for Intel® Xeon® and Intel® Xeon
Phi™ though the use of Intel® Math Kernel Library for Deep Neural Networks
(Intel® MKL-DNN) optimized primitives. The optimizations also provide speedups
for the consumer line of processors, e.g. i5 and i7 Intel processors. The Intel
published paper
[TensorFlow* Optimizations on Modern Intel® Architecture](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture)
contains additional details on the implementation.

> Note: MKL was added as of TensorFlow 1.2 and currently only works on Linux. It
> also does not work when also using `--config=cuda`.

In addition to providing significant performance improvements for training CNN
based models, compiling with the MKL creates a binary that is optimized for AVX
and AVX2. The result is a single binary that is optimized and compatible with
most modern (post-2011) processors.

TensorFlow can be compiled with the MKL optimizations using the following
commands that depending on the version of the TensorFlow source used.

For TensorFlow source versions after 1.3.0:

```bash
./configure
# Pick the desired options
bazel build --config=mkl -c opt //tensorflow/tools/pip_package:build_pip_package

```

For TensorFlow versions 1.2.0 through 1.3.0:

```bash
./configure
Do you wish to build TensorFlow with MKL support? [y/N] Y
Do you wish to download MKL LIB from the web? [Y/n] Y
# Select the defaults for the rest of the options.

bazel build --config=mkl --copt="-DEIGEN_USE_VML" -c opt //tensorflow/tools/pip_package:build_pip_package

```

#### MKL 调参以实现性能最优

This section details the different configurations and environment variables that
can be used to tune the MKL to get optimal performance. Before tweaking various
environment variables make sure the model is using the `NCHW` (`channels_first`)
[data format](#data-formats). The MKL is optimized for `NCHW` and Intel is
working to get near performance parity when using `NHWC`.

MKL uses the following environment variables to tune performance:

*   KMP_BLOCKTIME - Sets the time, in milliseconds, that a thread should wait,
    after completing the execution of a parallel region, before sleeping.
*   KMP_AFFINITY - Enables the run-time library to bind threads to physical
    processing units.
*   KMP_SETTINGS - Enables (true) or disables (false) the printing of OpenMP*
    run-time library environment variables during program execution.
*   OMP_NUM_THREADS - Specifies the number of threads to use.

More details on the KMP variables are on
[Intel's](https://software.intel.com/en-us/node/522775) site and the OMP
variables on
[gnu.org](https://gcc.gnu.org/onlinedocs/libgomp/Environment-Variables.html)

While there can be substantial gains from adjusting the environment variables,
which is discussed below, the simplified advice is to set the
`inter_op_parallelism_threads` equal to the number of physical CPUs and to set
the following environment variables:

*   KMP_BLOCKTIME=0
*   KMP_AFFINITY=granularity=fine,verbose,compact,1,0

Example setting MKL variables with command-line arguments:

```bash
KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,verbose,compact,1,0 \
KMP_SETTINGS=1 python your_python_script.py
```

Example setting MKL variables with python `os.environ`:

```python
os.environ["KMP_BLOCKTIME"] = str(FLAGS.kmp_blocktime)
os.environ["KMP_SETTINGS"] = str(FLAGS.kmp_settings)
os.environ["KMP_AFFINITY"]= FLAGS.kmp_affinity
if FLAGS.num_intra_threads > 0:
  os.environ["OMP_NUM_THREADS"]= str(FLAGS.num_intra_threads)

```

There are models and hardware platforms that benefit from different settings.
Each variable that impacts performance is discussed below.

*   **KMP_BLOCKTIME**: The MKL default is 200ms, which was not optimal in our
    testing. 0 (0ms) was a good default for CNN based models that were tested.
    The best performance for AlexNex was achieved at 30ms and both GoogleNet and
    VGG11 performed best set at 1ms.

*   **KMP_AFFINITY**: The recommended setting is
    `granularity=fine,verbose,compact,1,0`.

*   **OMP_NUM_THREADS**: This defaults to the number of physical cores.
    Adjusting this parameter beyond matching the number of cores can have an
    impact when using Intel® Xeon Phi™ (Knights Landing) for some models. See
    [TensorFlow* Optimizations on Modern Intel® Architecture](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture)
    for optimal settings.

*   **intra_op_parallelism_threads**: Setting this equal to the number of
    physical cores is recommended. Setting the value to 0, which is the default
    and will result in the value being set to the number of logical cores, is an
    option to try for some architectures.  This value and `OMP_NUM_THREADS`
    should be equal.

*   **inter_op_parallelism_threads**: Setting this equal to the number of
    sockets is recommended. Setting the value to 0, which is the default,
    results in the value being set to the number of logical cores.

### 编译器优化的对比

Collected below are performance results running training and inference on
different types of CPUs on different platforms with various compiler
optimizations.  The models used were ResNet-50
([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) and
InceptionV3 ([arXiv:1512.00567](https://arxiv.org/abs/1512.00567)).

For each test, when the MKL optimization was used the environment variable
KMP_BLOCKTIME was set to 0 (0ms) and KMP_AFFINITY to
`granularity=fine,verbose,compact,1,0`.

#### 推理 InceptionV3

**环境**

*   Instance Type: AWS EC2 m4.xlarge
*   CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz (Broadwell)
*   Dataset: ImageNet
*   TensorFlow Version: 1.2.0 RC2
*   Test Script: [tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/blob/mkl_experiment/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)

**每批次样本数目：1**

Command executed for the MKL test:

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=inception3 --data_format=NCHW \
--batch_size=1 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

| 优化 | 数据格式 | 图像数目/秒   | Intra 线程数 | Inter 线程数 |
:              :             : (每步时间)  :               :               :
| ------------ | ----------- | ------------ | ------------- | ------------- |
| AVX2         | NHWC        | 7.0 (142ms)  | 4             | 0             |
| MKL          | NCHW        | 6.6 (152ms)  | 4             | 1             |
| AVX          | NHWC        | 5.0 (202ms)  | 4             | 0             |
| SSE3         | NHWC        | 2.8 (361ms)  | 4             | 0             |

**每批次样本数目：32**

Command executed for the MKL test:

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=inception3 --data_format=NCHW \
--batch_size=32 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

| Optimization | Data Format | Images/Sec    | Intra threads | Inter Threads |
:              :             : (step time)   :               :               :
| ------------ | ----------- | ------------- | ------------- | ------------- |
| MKL          | NCHW        | 10.3          | 4             | 1             |
:              :             : (3,104ms)     :               :               :
| AVX2         | NHWC        | 7.5 (4,255ms) | 4             | 0             |
| AVX          | NHWC        | 5.1 (6,275ms) | 4             | 0             |
| SSE3         | NHWC        | 2.8 (11,428ms)| 4             | 0             |

#### 推理 ResNet-50

**环境**

*   Instance Type: AWS EC2 m4.xlarge
*   CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz (Broadwell)
*   Dataset: ImageNet
*   TensorFlow Version: 1.2.0 RC2
*   Test Script: [tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/blob/mkl_experiment/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)

**每批次样本数目：1**

Command executed for the MKL test:

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=resnet50 --data_format=NCHW \
--batch_size=1 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

| Optimization | Data Format | Images/Sec   | Intra threads | Inter Threads |
:              :             : (step time)  :               :               :
| ------------ | ----------- | ------------ | ------------- | ------------- |
| AVX2         | NHWC        | 8.8 (113ms)  | 4             | 0             |
| MKL          | NCHW        | 8.5 (120ms)  | 4             | 1             |
| AVX          | NHWC        | 6.4 (157ms)  | 4             | 0             |
| SSE3         | NHWC        | 3.7 (270ms)  | 4             | 0             |

**Batch Size: 32**

Command executed for the MKL test:

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=resnet50 --data_format=NCHW \
--batch_size=32 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

| Optimization | Data Format | Images/Sec    | Intra threads | Inter Threads |
:              :             : (step time)   :               :               :
| ------------ | ----------- | ------------- | ------------- | ------------- |
| MKL          | NCHW        | 12.4          | 4             | 1             |
:              :             : (2,590ms)     :               :               :
| AVX2         | NHWC        | 10.4 (3,079ms)| 4             | 0             |
| AVX          | NHWC        | 7.3 (4,4416ms)| 4             | 0             |
| SSE3         | NHWC        | 4.0 (8,054ms) | 4             | 0             |

#### 训练 InceptionV3

**环境**

*   Instance Type: Dedicated AWS EC2 r4.16xlarge (Broadwell)
*   CPU: Intel Xeon E5-2686 v4 (Broadwell) Processors
*   Dataset: ImageNet
*   TensorFlow Version: 1.2.0 RC2
*   Test Script: [tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/blob/mkl_experiment/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)

Command executed for MKL test:

```bash
python tf_cnn_benchmarks.py --device=cpu --mkl=True --kmp_blocktime=0 \
--nodistortions --model=resnet50 --data_format=NCHW --batch_size=32 \
--num_inter_threads=2 --num_intra_threads=36 \
--data_dir=<path to ImageNet TFRecords>
```

Optimization | Data Format | Images/Sec | Intra threads | Inter Threads
------------ | ----------- | ---------- | ------------- | -------------
MKL          | NCHW        | 20.8       | 36            | 2
AVX2         | NHWC        | 6.2        | 36            | 0
AVX          | NHWC        | 5.7        | 36            | 0
SSE3         | NHWC        | 4.3        | 36            | 0

ResNet and [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
were also run on this configuration but in an ad hoc manner. There were not
enough runs executed to publish a coherent table of results. The incomplete
results strongly indicated the final result would be similar to the table above
with MKL providing significant 3x+ gains over AVX2.
