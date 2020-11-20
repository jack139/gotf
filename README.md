## 使用go运行训练好的tensorflow模型

### 安装tensorflow的C库
CPU
```
sudo tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz
```
GPU
```
sudo tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz
```

```
sudo ldconfig
```

CentOS需要设置：
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

安装测试
```
gcc examples/hello_tf.c -ltensorflow -o hello_tf
./hello_tf
```

### 安装tensorflow的go库
```
go get github.com/tensorflow/tensorflow/tensorflow/go@v1.15.4
go test github.com/tensorflow/tensorflow/tensorflow/go
```

安装测试
```
go run examples/hello_tf.go
```

### 问答测试
```
go run examples/bert.go
```

### http服务测试
```
go build
./gotf
curl -X POST http://127.0.0.1:8080/qa --data '{"c":"我是小明，我8岁了。","q":"你多大了？"}'
```
