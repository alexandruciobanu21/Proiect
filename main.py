from config_kernel import Kernel
import pyopencl as cl
import numpy as np
import cv2
from imageio import imread, imsave


#incarcare imagine
def load_image(path):
        if len(imread(path).shape) == 3:
            image = cv2.cvtColor(imread(path), cv2.COLOR_BGR2GRAY)
            return image.astype(np.float32)
        else:
            return imread(path).astype(np.float32)

#salvare imagine
def save_image(path, image):
      return imsave(path, image.astype(np.float32))


def kernel_create():
    #configurare kernel
    context = cl.create_some_context()
    queue_prop = cl.command_queue_properties
    queue = cl.CommandQueue(context, properties=queue_prop.PROFILING_ENABLE)
    memF = cl.mem_flags
    work_group = None

    program = cl.Program(context, open("/content/Proiect/gauss.cl").read())
    program = program.build()
    kernel = Kernel(context, queue, memF, work_group, program)

    return kernel

def gauss_blur(kernel, imgIn):
  #definirea bufferelor
    imgInBuf = cl.Buffer(kernel.context, kernel.memF.READ_ONLY | kernel.memF.COPY_HOST_PTR, hostbuf=imgIn)
    imgOutBuf = cl.Buffer(kernel.context, kernel.memF.WRITE_ONLY, imgIn.nbytes)
    imgWidthBuf = cl.Buffer( kernel.context, kernel.memF.READ_ONLY | kernel.memF.COPY_HOST_PTR, hostbuf=np.int32(imgIn.shape[1]))
    imgHeightBuf = cl.Buffer( kernel.context, kernel.memF.READ_ONLY | kernel.memF.COPY_HOST_PTR, hostbuf=np.int32(imgIn.shape[0]))

    kernel.program.Gauss( kernel.queue, imgIn.shape, kernel.work_group, imgInBuf, imgOutBuf, imgWidthBuf, imgHeightBuf)
    kernel.queue.finish()
  #copiere din buffer ul de out in imaginea de out
    imgOut = np.empty_like(imgIn)
    cl.enqueue_copy(kernel.queue, imgOut, imgOutBuf)

    return imgOut


#apelarea functiilor
kernel = kernel_create()
imgIn = load_image("/content/Proiect/image.jpg")
save_image("/content/Proiect/image_read.jpg", imgIn)
imgOut = gauss_blur(kernel, imgIn)
save_image("/content/Proiect/image_result.jpg", imgOut)
