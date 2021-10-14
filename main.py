#some experimental python code testing numba and cupy
#token test
import cupy
import numpy as np

from timeit import default_timer as timer

try:
    from matplotlib.pylab import imshow, show
    have_mpl = True
except ImportError:
    have_mpl = False
from numba import jit

def main_func():
    # nvprof --print-gpu-trace python examples/stream/cufft.py

    x = cupy.array(np.random.rand(1024), dtype=float)
    expected_f = cupy.fft.fft(x)
    cupy.cuda.Device().synchronize()

    stream = cupy.cuda.stream.Stream()
    with stream:
        f = cupy.fft.fft(x)
    stream.synchronize()
    cupy.testing.assert_array_equal(f, expected_f)
    stream = cupy.cuda.stream.Stream()
    stream.use()
    f = cupy.fft.fft(x)
    stream.synchronize()
    cupy.testing.assert_array_equal(f, expected_f)

@jit(nopython=True)
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return 255

@jit(nopython=True)
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

    return image

def show_mandelbrot():
    image = np.zeros((500 * 2, 750 * 2), dtype=np.uint8)
    s = timer()
    create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
    e = timer()
    print(e - s)
    if have_mpl:
        imshow(image)
        show()

if __name__ == '__main__':
    for idx in range(1000):
        main_func()
    print("Finished CUDA FFT")
    show_mandelbrot()
    print("Finished Mandelbrot fractal plotting")




