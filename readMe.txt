GPU-ArraySort 2.0
License information can be found in the source code file.


this algorithm is able to sort large number of variable-sized arrays on a GPU. 
It utilizes CUDA's two layered parallelism in a very efficient manner, by
moving the excessively used data on the on-chip shared memory while keeping
the remaining in global memory. It is effectively an in-place algorithm. 

What's new?
In GPU-ArraySort 2.0 several bugs and limitations have been removed. This version
is capable of handling variable-sized arrays as well as same sized arrays while
giving much better processing speed. GPU-ArraySort 2.0 provides a 50x to 60x speed up
over simple multicore sorting of arrays on GPU. It is capable of handling all the 
primitive data types.

Compilation
Use the following command for compiling the source code:
nvcc -std=c++11 -arch=sm_35 GPU-ArraySort-2-0.cu -o out
This code has been tested on Tesla K-40c GPU by NVIDIA on a server running Ubuntu 14.04,
CUDA 7.5 and gcc version 4.8.


If you use GPU-ArraySort, please site our work using :

Bibtex:

@inproceedings{AwanSaeed2016Sort,
  title={GPU-ArraySort: A parallel, in-place algorithm for sorting large number of arrays},
  author={Awan, Muaaz and Saeed, Fahad},
  booktitle={Proceedings of Workshop on High Performance Computing for Big Data, International Conference on Parallel Processing (ICPP-2016), Philadelphia PA},
  pages={1--10},
  year={2016}
}

MLA:
Awan, Muaaz, and Fahad Saeed. "GPU-ArraySort: A parallel, in-place algorithm for sorting large number of arrays.",
Proceedings of Workshop on High Performance Computing for Big Data, International Conference on Parallel Processing 
(ICPP-2016), Philadelphia PA, (2016).




