#pragma once

namespace zi {
namespace znn {
namespace frontiers {

inline void visit_6_helper( cube<int>& c, int x, int y, int z )
{
    c(x,y,z) = 0;
    if ( (x>0) && c(x-1,y,z) ) visit_6_helper(x-1,y,z);
    if ( (y>0) && c(x,y-1,z) ) visit_6_helper(x,y-1,z);
    if ( (z>0) && c(x,y,z-1) ) visit_6_helper(x,y,z-1);
    if ( (x<2) && c(x+1,y,z) ) visit_6_helper(x+1,y,z);
    if ( (y<2) && c(x,y+1,z) ) visit_6_helper(x,y+1,z);
    if ( (z<2) && c(x,y,z+1) ) visit_6_helper(x,y,z+1);
}

inline void visit_6_inv_helper( cube<int>& c, int x, int y, int z )
{
    c(x,y,z) = 1;
    if ( (x>0) && !c(x-1,y,z) ) visit_6_helper(x-1,y,z);
    if ( (y>0) && !c(x,y-1,z) ) visit_6_helper(x,y-1,z);
    if ( (z>0) && !c(x,y,z-1) ) visit_6_helper(x,y,z-1);
    if ( (x<2) && !c(x+1,y,z) ) visit_6_helper(x+1,y,z);
    if ( (y<2) && !c(x,y+1,z) ) visit_6_helper(x,y+1,z);
    if ( (z<2) && !c(x,y,z+1) ) visit_6_helper(x,y,z+1);
}

inline void visit_26_helper( cube<int>& c, int x, int y, int z )
{
    if ( x<0 || x>2 || y<0 || y>2 || z<0 || z>2 || !c(x,y,z) )
        return;

    c(x,y,z) = 0;
    for ( int i = -1; i < 2; ++i )
        for ( int j = -1; j < 2; ++j )
            for ( int k = -1; k < 2; ++k )
                visit_26_helper(c, x+i, y+j, z+k);
}

inline void visit_26_inv_helper( cube<int>& c, int x, int y, int z )
{
    if ( x<0 || x>2 || y<0 || y>2 || z<0 || z>2 || c(x,y,z) )
        return;

    c(x,y,z) = 1;
    for ( int i = -1; i < 2; ++i )
        for ( int j = -1; j < 2; ++j )
            for ( int k = -1; k < 2; ++k )
                visit_26_helper(c, x+i, y+j, z+k);
}


inline void

inline bool is_simple(int* b)
{
    // 1. N6 = 1, then p is simple

    int N6 = b[4] + b[10] + b[12] + b[14] + b[16] + b[22];

    if ( N6 == 1 ) return true;

    // 2. N6 = 2, then p is simple none of the two 6-neighbors are opposite

    if ( N6 == 2 )
        if (!( (b[4]&&b[22]) || (b[10]&&b[16]) || (b[12]&&b[14]) ))
            return true;

    // 3. Calculate Euler number eps. If eps != 1 then p is not simple

    int n2 = N6;
    int n1 = b[1]+b[3]+b[5]+b[7]+b[9]+b[11]+b[15]+b[17]+b[19]+b[21]+b[23]+b[25];
    int n0 = b[0]+b[2]+b[6]+b[8]+b[18]+b[20]+b[24]+b[26];

    int eps = n0 - n1 + n2;

    if ( eps != 1 ) return false;

    // 4. If an isolated 0-cell is in S then p is not simple

#define __CHECK_0_CELL(i0,i1,i2,i3,i4,i5,i6)                            \
    if ( b[i0] && !b[i1] && !b[i2] && !b[i3] && !b[i4] && !b[i5] && !b[i6] ) \
        return false

    __CHECK_0_CELL(0,1,3,4,9,10,12);
    __CHECK_0_CELL(2,1,4,5,10,11,14);
    __CHECK_0_CELL(6,3,4,7,12,15,16);
    __CHECK_0_CELL(8,4,5,7,14,16,17);
    __CHECK_0_CELL(18,19,21,22,9,10,12);
    __CHECK_0_CELL(20,19,22,23,10,11,14);
    __CHECK_0_CELL(24,21,22,25,12,15,16);
    __CHECK_0_CELL(26,22,23,25,14,16,17);

#undef __CHECK_0_CELL

    // 5. If an isolated 1-cell is in S then p is not simple

#define __CHECK_1_CELL(i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10)               \
    if ( b[i0] && !b[i1] && !b[i2] && !b[i3] && !b[i4] && !b[i5] &&     \
         !b[i6] && !b[i7] && !b[i8] && !b[i9] && !b[i10] )              \
        return false

    __CHECK_1_CELL(1,0,2,3,4,5,9,10,11,12,14);
    __CHECK_1_CELL(3,0,1,4,6,7,9,10,12,15,16);
    __CHECK_1_CELL(5,1,2,4,7,8,10,11,14,16,17);
    __CHECK_1_CELL(7,3,4,5,6,8,12,14,15,16,17);
    __CHECK_1_CELL(19,18,20,21,22,23,9,10,11,12,14);
    __CHECK_1_CELL(21,18,19,22,24,25,9,10,12,15,16);
    __CHECK_1_CELL(23,19,20,22,25,26,10,11,14,16,17);
    __CHECK_1_CELL(25,21,22,23,24,26,12,14,15,16,17);
    __CHECK_1_CELL(9,0,1,3,4,10,12,18,19,21,22);
    __CHECK_1_CELL(11,1,2,4,5,10,14,19,20,22,23);
    __CHECK_1_CELL(15,3,4,6,7,12,16,21,22,24,25);
    __CHECK_1_CELL(17,4,5,7,8,14,16,22,23,25,26);

#undef __CHECK_1_CELL

    // 6. If an isolated 2-cell is in inv(S)

    //std::cout << "EPS: " << eps << std::endl;

#define __CHECK_2_CELL(i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16) \
    if ( !b[i0] && b[i1] && b[i2] && b[i3] && b[i4] && b[i5] && b[i6]   \
         && b[i7] && b[i8] && b[i9] && b[i10] && b[i11] && b[i12]       \
         && b[i13] && b[i14] && b[i15] && b[i16] )                      \
        return false

    __CHECK_2_CELL(4,0,1,2,3,5,6,7,8,         9,10,11,12,14,15,16,17);
    __CHECK_2_CELL(22,18,19,20,21,23,24,25,26,9,10,11,12,14,15,16,17);

    __CHECK_2_CELL(12,0,3,6,9,15,18,21,24, 1,4,7,10,16,19,22,25);
    __CHECK_2_CELL(14,2,5,8,11,17,20,23,26,1,4,7,10,16,19,22,25);

    __CHECK_2_CELL(10,0,1,2,9,11,18,19,20, 3,4,5,12,14,21,22,23);
    __CHECK_2_CELL(16,6,7,8,15,17,24,25,26,3,4,5,12,14,21,22,23);

#undef __CHECK_2_CELL

    return true;

}

}}} // namespace zi::znn::frontiers
