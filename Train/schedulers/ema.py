# HE的半监督teacher模型权重EMA调度器
import torch


class EMAUpdater:
    """
    指数移动平均更新器，用于 Teacher-Student 半监督

    Teacher 参数 = α * Teacher 参数 + (1-α) * Student 参数
    α 通常接近 1（如 0.999），确保 Teacher 更新平缓，伪标签稳定
    注意：BN 的 running_mean / running_var / num_batches_tracked 属于 buffers，
          不在 parameters() 中，需在 update 里同步（teacher 直接拷贝 student）

    用法：
        ema = EMAUpdater(decay=0.999)
        ...
        ema.update(teacher, student)  # 每个 epoch 或每 N 步调用一次
    """

    def __init__(self, decay=0.999):
        """
        decay: EMA 衰减率（即公式中的 α），越大更新越缓慢
        """
        self.decay = decay

    @torch.no_grad()
    def update(self, teacher, student):
        """
        将 Student 的参数 EMA 平滑到 Teacher

        teacher: nn.Module, 不参与梯度的目标模型
        student: nn.Module, 正常训练的源模型
        """
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            # zip：把两个可迭代对象按位置配对并返回一个迭代器，对象是torch.nn.Parameter
            # EMA 公式：θ_t = α * θ_t + (1-α) * θ_s
            t_param.data.copy_(
                self.decay * t_param.data + (1.0 - self.decay) * s_param.data
            )

        # 同步 BN 统计量等 buffers
        # running_mean / running_var / num_batches_tracked 属于 buffers，不在 parameters() 里，
        # 若不同步，teacher 一直 eval() 不累积统计量，归一化会永远停留在初始值（mean=0, var=1）。
        # 这里直接 copy_ student 的 buffer：num_batches_tracked 是整型张量，无法参与浮点 EMA 平均。
        for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
            t_buf.data.copy_(s_buf.data)