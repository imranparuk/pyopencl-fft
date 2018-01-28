# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 02:39:51 2018

@author: imran
"""

#check out this when you have time (not used yet -> for optimization)
#http://www.bealto.com/gpu-fft_group-2.html

#github pyopencl fft kernel adapetd from
#https://github.com/pradeepsinngh/fft-dft-opencl

#opencl float2 stuff
# thanks to -> https://stackoverflow.com/questions/21745934/how-to-use-float2-in-pyopencl

#timeit not working? 

#When algorithm gets better, it needs to compete with other GPU FFT algo's
# (1) http://pythonhosted.org/pyfft/

import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.tools import get_test_platforms_and_devices

import datetime
from time import time

import os

print(get_test_platforms_and_devices())

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

(n, m) = (1,100)

a = np.array([4 + 1j*5, 5 + 1j*6, 6 + 1j*7]).astype(np.complex64)
b = np.random.randn(n, m).astype(np.complex64)
c = np.zeros((n*m), np.complex64)

length = b.size;
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


prg = cl.Program(ctx, open('kernel.cl').read()).build();


    
#	hls_run_kernel("dft",x,2*len,y,2*len,length,1,sign,1);

time1 = time();
prg.dft(queue, c.shape, None, b_buf, c_buf, len_buf, sign_buff)
a_mul_b = np.empty_like(c)
cl.enqueue_copy(queue, a_mul_b, c_buf)

time2 = time();
z = np.fft.fft(b);
time3 = time();

timeGPUdft = time2 - time1;
timePyFFT = time3 - time2;



print("matrix A:")
print(b.reshape(n, m))
print("pyopencl_fft:")
print(a_mul_b.reshape(n, m))
print("python fft:")
print(z.reshape(n,m))

print("Time Taken GPU DFT: " , timeGPUdft )
print("Time Taken Py FFT: ",  timePyFFT )







