# 添加一个定制的文件系统插件

## 背景

TensorFlow 框架经常用于多进程和多机环境，比如谷歌数据中心，谷歌机器学习云，亚马逊网络服务（AWS）以及实地分布集群。为了分享和保存 TensorFlow 创建的某些类型的状态，框架会假设存在可靠的、共享的文件系统。这个共享的文件系统有诸多用途，比如：

*   状态的检查点通常保存到一个分布式文件系统用来保证可靠性和容错性。
*   训练进程通过将事件文件写入一个由 TensorBoard 监听的目录与 TensorBoard 通讯。共享的文件系统允许即使 TensorBoard 运行在不同的进程和机器上，通讯仍然能够正常进行。

现实中已经有很多不同的共享或分布式文件系统的实现，所以 TensorFLow 提供实现一个定制的文件系统插件的方法，可以注册到 TensorFlow 运行时中。当 TensorFlow 运行时尝试通过 `FileSystem` 接口写入一个文件，它使用路径名的一部分来动态选择用于文件系统操作的实现。因此，为了支持定制的文件系统需要实现一个 `FileSystem` 接口，构建一个包含实现的共享对象，并在运行时加载对象到任意需要写入文件系统的进程中。

注意 TensorFlow 已经包含很多文件系统的实现，例如：

*   标准 POSIX 文件系统

    注意：NFS 文件系统通常作为一个 POSIX 接口挂载，所以标准 TensorFlow 能够运行在挂载了 NFS 的远程文件系统上。

*   HDFS - Hadoop 文件系统
*   GCS - 谷歌云存储文件系统
*   “内存映射文件”文件系统

接下来讲如何实现一个定制的文件系统。

## 实现一个定制的文件系统插件

为了实现一个定制的文件系统插件，必须执行如下操作：

*   实现 `RandomAccessFile`、`WriteableFile`、`AppendableFile` 和 `ReadOnlyMemoryRegion` 子类。
*   实现 `FileSystem` 接口作为一个子类。
*   使用合适的前缀模式注册 `FileSystem` 实现。
*   在想要写入文件系统的进程中加载文件系统插件。

### FileSystem 接口

`FileSystem` 接口是一个定义在 [file_system.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/file_system.h) 中的抽象 C++ 接口。一个 `FileSystem` 接口的实现应该实现所有在接口中定义的方法。实现需要定义操作，如创建 `RandomAccessFile`、`WritableFile` 和实现标准文件系统操作，如 `FileExists`、`IsDirectory`、`GetMatchingPaths`、`DeleteFile` 等等。这些接口的实现通常包括将函数的入参委托给一个已经存在的实现同等功能的库函数。

例如，`PosixFileSystem` 使用 POSIX `unlink()` 函数实现了 `DeleteFile`；`CreateDir` 只是简单地调用 `mkdir()`；`GetFileSize` 对文件调用 `stat()` 然后返回文件大小。类似地，对于 `HDFSFileSystem` 实现，这些调用只是简单地委托给有着相似功能的 `libHDFS` 实现，例如用于 [DeleteFile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/hadoop/hadoop_file_system.cc#L386) 的 `hdfsDelete`。

我们建议阅读这些代码示例，以了解不同文件系统实现如何调用已经存在的库。示例包括：

*   [POSIX 插件](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/posix/posix_file_system.h)
*   [HDFS 插件](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/hadoop/hadoop_file_system.h)
*   [GCS 插件](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/cloud/gcs_file_system.h)

#### 文件接口

在允许查询和操作一个文件系统中的文件和目录外，`FileSystem` 接口要求实现返回像 [RandomAccessFile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/file_system.h#L223)、`WritableFile` 这样的抽象对象的工厂方法，从而 TensorFlow 可以编码、读取和写入 `FileSystem` 实现中的文件。

为了实现一个 `RandomAccessFile`，你必须实现一个单独的 `Read()` 接口，其中实现必须提供一种从一个偏移开始读取某个特定文件的方法。

例如，下面是针对 POSIX 文件系统的 RandomAccessFile 实现，读取通过 `pread()` 随机访问 POSIX 函数实现。需要注意实现必须知道如何重试或传播来自底层文件系统的错误。

```C++
    class PosixRandomAccessFile : public RandomAccessFile {
     public:
      PosixRandomAccessFile(const string& fname, int fd)
          : filename_(fname), fd_(fd) {}
      ~PosixRandomAccessFile() override { close(fd_); }

      Status Read(uint64 offset, size_t n, StringPiece* result,
                  char* scratch) const override {
        Status s;
        char* dst = scratch;
        while (n > 0 && s.ok()) {
          ssize_t r = pread(fd_, dst, n, static_cast<off_t>(offset));
          if (r > 0) {
            dst += r;
            n -= r;
            offset += r;
          } else if (r == 0) {
            s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
          } else if (errno == EINTR || errno == EAGAIN) {
            // Retry
          } else {
            s = IOError(filename_, errno);
          }
        }
        *result = StringPiece(scratch, dst - scratch);
        return s;
      }

     private:
      string filename_;
      int fd_;
    };
```

为了实现 WritableFile 顺序写入抽象，实现者必须实现像 `Append()`、`Flush()`、`Sync()` 和 `Close()` 这些接口。

例如，下面是对 POSIX 文件系统的 WritableFile 实现。构造函数内接受 `FILE` 对象，并在对象上使用标准的 POSIX 函数来实现接口。

```C++
    class PosixWritableFile : public WritableFile {
     public:
      PosixWritableFile(const string& fname, FILE* f)
          : filename_(fname), file_(f) {}

      ~PosixWritableFile() override {
        if (file_ != NULL) {
          fclose(file_);
        }
      }

      Status Append(const StringPiece& data) override {
        size_t r = fwrite(data.data(), 1, data.size(), file_);
        if (r != data.size()) {
          return IOError(filename_, errno);
        }
        return Status::OK();
      }

      Status Close() override {
        Status result;
        if (fclose(file_) != 0) {
          result = IOError(filename_, errno);
        }
        file_ = NULL;
        return result;
      }

      Status Flush() override {
        if (fflush(file_) != 0) {
          return IOError(filename_, errno);
        }
        return Status::OK();
      }

      Status Sync() override {
        Status s;
        if (fflush(file_) != 0) {
          s = IOError(filename_, errno);
        }
        return s;
      }

     private:
      string filename_;
      FILE* file_;
    };

```

想要了解更多细节，可以查看接口文档或者实例。

### 注册和加载文件系统

一旦你已经为你定制的文件系统实现了 `FileSystem`，你需要在一个 scheme 下注册它，从而以那个 scheme 为前缀的路径会被导向你的实现。为了实现这个，调用 `REGISTER_FILE_SYSTEM`：

```
    REGISTER_FILE_SYSTEM("foobar", FooBarFileSystem);
```

当 TensorFlow 尝试操作一个路径以 `foobar://` 开始的文件，它将使用 `FooBarFileSystem` 实现。

```C++
    string filename = "foobar://path/to/file.txt";
    std::unique_ptr<WritableFile> file;

    // Calls FooBarFileSystem::NewWritableFile to return
    // a WritableFile class, which happens to be the FooBarFileSystem's
    // WritableFile implementation.
    TF_RETURN_IF_ERROR(env->NewWritableFile(filename, &file));
```

接下来，你必须构建一个包含这个实现的共享对象。[这里](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/BUILD#L244)是一个使用 bazel `cc_binary` 规则实现的实例。但是你可以使用任何构建系统实现。相似的指导可以查看 @{$adding_an_op#build-the-op-library$building the op library} 这节。

构建的结果是一个 `.so` 共享对象文件。

最后，你必须在进程中动态加载这个实现。Python 中，你可以调用 `tf.load_file_system_library(file_system_library)` 函数，传入共享对象的路径。在客户端程序中调用这个方法加载进程中的共享对象并注册实现，以用于任意经过 `FileSystem` 接口的文件操作。示例可以查看 [test_file_system.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/file_system_test.py)。

## 什么会经过接口？

TensorFlow 中几乎所有核心 C++ 文件操作都使用 `FileSystem` 接口，例如 `CheckpointWriter`、`EventsWriter` 以及许多其他功能。这意味着实现一个 `FileSystem` 实现可以让大多数 TensorFlow 程序写入你的共享文件系统。

Python 中，`gfile` 和 `file_io` 类通过 SWIG 绑定到 `FileSystem` 实现下，这意味着一旦你加载了这个文件系统库，你可以执行：

```
with gfile.Open("foobar://path/to/file.txt") as w:

  w.write("hi")
```

在执行这个之后，一个包含 ”hi“ 的文件会出现在共享文件系统的 ”/path/to/file.txt“。
