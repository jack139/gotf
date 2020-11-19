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

### 安装tensorflow的go库
```
go get github.com/tensorflow/tensorflow/tensorflow/go@v1.15.4
go test github.com/tensorflow/tensorflow/tensorflow/go
```
