class Kernel:
    def __init__(self, context, queue, memF, work_group, program):
        self.context = context
        self.queue = queue
        self.memF = memF
        self.work_group = work_group
        self.program = program