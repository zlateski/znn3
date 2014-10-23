znn
===========
Multi-core CPU implementation of deep learning for 2D and 3D convolutional networks.



Required libraries
------------------
Currently we only support linux environments.

|Library|Ubuntu package name|
|:-----:|-------------------|
|[boost](http://www.boost.org/)|libboost-all-dev|


Compile & clean
---------------
    cd ../..
    make
    make clean

If compile is successful, an executalbe named **znn** will be generated under the directory [bin](./bin/).



Usage
-----------
#### Training
TBA

#### Forward Pass
./bin/znn [network name] [volume name] [output directory]

#### Viewing Results
MATLAB functions for preparing and viewing results are provided in the ./matlab directory


Contact
-------
* Aleksander Zlateski \<zlateski@mit.edu\>
