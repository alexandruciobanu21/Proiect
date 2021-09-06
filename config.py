class KernelConfig:
    def __init__(self, ctx, queue, mf, local_work_group, prg):
        self.ctx = ctx
        self.queue = queue
        self.mf = mf
        self.local_work_group = local_work_group
        self.prg = prg