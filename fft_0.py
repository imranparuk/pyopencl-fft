# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 02:39:51 2018

@author: imran
"""

#check out this when you have time (not used yet -> for optimization)
#http://www.bealto.com/gpu-fft_group-2.html

import pyopencl as cl
import numpy as np

#opencl float2 stuff
# thanks to -> https://stackoverflow.com/questions/21745934/how-to-use-float2-in-pyopencl
import pyopencl as cl
import pyopencl.array as cl_array


import os

from pyopencl.tools import get_test_platforms_and_devices
print(get_test_platforms_and_devices())

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

(n, m) = (2,5)

# a = np.random.randn(n, m).astype(np.float32)
# b = np.random.randn(m, p).astype(np.float32)
a = np.random.randint(100, size=(n*m))

c = np.zeros((n*m), np.float32)


z = np.fft.fft(a)

a = a.astype(np.float32)
sze = a.size;



platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)

queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer\
   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer\
   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)


prg2 = cl.Program(ctx, """
   __kernel void dft(
	__global const float *in, // complex values input
	__global float *out)                         // 1 for forward transform, -1 for backward.
{
	// Get the varying parameter of the parallel execution :
	int i = get_global_id(0);
	int length = 10;
	// In case we're executed "too much", check bounds :
	if (i >= length)
		return;
	
    int sign = 1;
	// Initialize sum and inner arguments
	float tot = 0;
	float param = (-2 * sign * i) * M_PI / (double)length;
	
	for (int k = 0; k < length; k++) {
		float value = in[k];
		
		// Compute sin and cos in a single call : 
		float c;
		float s = sincos(k * param, &c);
		
		// This adds (value.x * c - value.y * s, value.x * s + value.y * c) to the sum :
		tot += (float)(
			dot(value, (float)(c, -s)), dot(value, (float)(s, c))
		);
	}
	
	if (sign == 1) {
		// forward transform (space -> frequential)
		out[i] = tot;
	} else {
		// backward transform (frequential -> space)
		out[i] = tot / (float)length;
	}
}""").build();
        
prg2.dft(queue, c.shape, None, a_buf, c_buf)

a_mul_b = np.empty_like(c)
cl.enqueue_copy(queue, a_mul_b, c_buf)

print("matrix A:")
print(a.reshape(n, m))
print("multiplied A*B:")
print(a_mul_b.reshape(n, m))
print("python fft:")
print(z.reshape(n,m))