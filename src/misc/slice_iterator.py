class SliceIterator:
    def __init__(self, start, stop, step, slice_step=1):
        if step == 0:
            raise ValueError("step cannot be zero")
        self.start = start
        self.stop = stop
        self.step = step
        self.slice_step = slice_step
        self.current_start = start

    def __iter__(self):
        return self

    def __next__(self):
        # 检查是否完成迭代
        if self.step > 0 and self.current_start >= self.stop:
            raise StopIteration
        if self.step < 0 and self.current_start <= self.stop:
            raise StopIteration

        # 计算当前结束位置
        current_end = self.current_start + self.step

        # 根据步长方向调整结束位置
        if self.step > 0:
            current_end = min(current_end, self.stop)
        else:
            current_end = max(current_end, self.stop)

        # 创建切片对象
        if self.step > 0:
            slice_obj = slice(self.current_start, current_end, self.slice_step)
        else:
            slice_obj = slice(self.current_start, current_end, -self.slice_step)

        # 更新下一个起始位置
        self.current_start = current_end

        return slice_obj