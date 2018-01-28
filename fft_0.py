# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 02:39:51 2018

@author: imran
"""

#check out this when you have time (not used yet -> for optimization)
#http://www.bealto.com/gpu-fft_group-2.html

#github pyopencl fft kernel adapetd from
#https://github.com/pradeepsinngh/fft-dft-opencl

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

(n, m) = (1,3)

a = np.array([4 + 1j*5, 5 + 1j*6, 6 + 1j*7]).astype(np.complex64)
b = np.random.randn(n, m).astype(np.complex64)
c = np.zeros((n*m), np.complex64)
z = np.fft.fft(a)

length = a.size;
sign = 1;


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


len_buf = np.int32(length);
   
sign_buff = np.int32(sign);

prg2 = cl.Program(ctx, """
# define len 64 //// denotes value of N point
__kernel
void dft(
	__global const float2 *in, // complex values input
	__global float2 *out,      // complex values output
	 int length,                 // number of input and output values
	 int sign)                   // sign modifier in the exponential :
	                            // 1 for forward transform, -1 for backward.
{
	// Get the varying parameter of the parallel execution :
	int i = get_global_id(0);

	// In case we're executed "too much", check bounds :
	if (i >= length)
		return;

	// Initialize sum and inner arguments
	float2 tot = 0;
	float param = (-2 * sign * i) * 3.141593f / (float)length;

	for (int k = 0; k < length; k++) {
		float2 value = in[k];

		// Compute sin and cos in a single call :
		float c;
		float s = sincos(k * param, &c);

		// This adds (value.x * c - value.y * s, value.x * s + value.y * c) to the sum :
		tot += (float2)(
			dot(value, (float2)(c, -s)),
			dot(value, (float2)(s, c))
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
    
#	hls_run_kernel("dft",x,2*len,y,2*len,length,1,sign,1);

prg2.dft(queue, c.shape, None, a_buf, c_buf, len_buf, sign_buff)

a_mul_b = np.empty_like(c)
cl.enqueue_copy(queue, a_mul_b, c_buf)

print("matrix A:")
print(a.reshape(n, m))
print("pyopencl_fft:")
print(a_mul_b.reshape(n, m))
print("python fft:")
print(z.reshape(n,m))





