from config import KernelConfig
import pyopencl as cl
import numpy as np
import cv2
from imageio import imread, imsave

def load_image(path):
        if len(imread(path).shape) == 3:
            image = cv2.cvtColor(imread(path), cv2.COLOR_BGR2HSV)
            return image.astype(np.float32)
        else:
            return imread(path).astype(np.float32)


def save_image(path, image):
      return imsave(path, image.astype(np.float32))


def config_kernel():
    ctx = cl.create_some_context()

    cpq = cl.command_queue_properties

    queue = cl.CommandQueue(ctx, properties=cpq.PROFILING_ENABLE)

    mf = cl.mem_flags

    local_work_group = None

    prg = cl.Program(ctx, open("/content/Proiect/gauss.cl").read())
    prg = prg.build()
    
    config = KernelConfig(ctx, queue, mf, local_work_group, prg)

    return config

def app(config, img):
    img_buffer = cl.Buffer(
        config.ctx, config.mf.READ_ONLY | config.mf.COPY_HOST_PTR, hostbuf=img
    )
    result_buffer = cl.Buffer(config.ctx, config.mf.WRITE_ONLY, img.nbytes)

    width_buffer = cl.Buffer(
        config.ctx,
        config.mf.READ_ONLY | config.mf.COPY_HOST_PTR,
        hostbuf=np.int32(img.shape[1]),
    )
    height_buffer = cl.Buffer(
        config.ctx,
        config.mf.READ_ONLY | config.mf.COPY_HOST_PTR,
        hostbuf=np.int32(img.shape[0]),
    )

    event1 = config.prg.Gauss(
        config.queue,
        img.shape,
        config.local_work_group,
        img_buffer,
        result_buffer,
        width_buffer,
        height_buffer,
    )
    config.queue.finish()

    # copies resulting image
    result = np.empty_like(img)
    cl.enqueue_copy(config.queue, result, result_buffer)

    return result



config = config_kernel()
img = load_image("/content/Proiect/lenna.jpg")
save_image("/content/Proiect/lenna_read.jpg", img)
result = app(config, img)
save_image("/content/Proiect/lenna_result.jpg", result)
