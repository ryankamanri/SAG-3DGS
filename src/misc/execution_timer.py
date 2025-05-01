import torch
import time
import psutil  # 用于统计 CPU 内存消耗

class ExecutionTimer:
    def __init__(self, desc="default", verbose=True, device=None, switch=True):
        """
        初始化计时器
        :param verbose: 是否自动打印耗时（默认True）
        :param device: 指定计算设备（默认自动检测）
        """
        self.switch = switch
        self.desc = desc
        self.verbose = verbose
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.elapsed_time = None  # 保存耗时结果（毫秒）
        self.memory_usage = None  # 保存内存消耗结果

    def __enter__(self):
        if not self.switch: 
            return self
        # CUDA时间测量需要特殊处理
        if self.device.type == 'cuda':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            # 记录初始显存使用量
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated(self.device)
        else:
            self.start_time = time.perf_counter()
            # 记录初始 CPU 内存使用量
            self.start_memory = psutil.Process().memory_info().rss  # 单位：字节
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.switch:
            return
        # 计算耗时并处理CUDA同步
        if self.device.type == 'cuda':
            self.end_event.record()
            torch.cuda.synchronize()  # 等待所有CUDA操作完成
            self.elapsed_time = self.start_event.elapsed_time(self.end_event)
            # 计算显存消耗
            self.end_memory = torch.cuda.memory_allocated(self.device)
            self.memory_usage = (self.end_memory - self.start_memory) / (1024 ** 2)  # 转为 MB
        else:
            self.elapsed_time = (time.perf_counter() - self.start_time) * 1000  # 转为毫秒
            # 计算 CPU 内存消耗
            self.end_memory = psutil.Process().memory_info().rss  # 单位：字节
            self.memory_usage = (self.end_memory - self.start_memory) / (1024 ** 2)  # 转为 MB

        if self.verbose:
            # 打印耗时
            unit = 'ms' if self.elapsed_time < 1000 else 's'
            value = self.elapsed_time if unit == 'ms' else self.elapsed_time / 1000
            print(f"Execution time({self.desc}): {value:.3f}{unit}")
            # 打印内存消耗
            memory_unit = 'MB'
            print(f"Memory usage({self.desc}): {self.memory_usage:.3f}{memory_unit}")

# 使用示例
if __name__ == "__main__":
    # 在CPU上测试
    with ExecutionTimer(desc="CPU Test"):
        x = torch.randn(1000, 1000)
        torch.mm(x, x)
    
    # 在GPU上测试（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with ExecutionTimer(desc="GPU Test", device=device):
        y = torch.randn(10000, 10000, device=device)
        torch.mm(y, y)