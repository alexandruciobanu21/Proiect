__kernel void Gauss(__global const float *imgIn, __global  float *imgOut, __global const int *width, __global const int *height){
    const int w = *width;
    const int h = *height;
    float ps, pc, pd;
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    const int i = w*y + x;
    // Gaussian Kernel is:
  	// 1 2 1
	  // 2 4 2
	  // 1 2 1
    if(x == 0 || y ==0 || x==w-1 || y == h-1){
         imgOut[i] = imgIn[i] ;  
    }else{
        ps =  2*imgIn[i    -1];
        pc =  4*imgIn[i];
        pd =  2*imgIn[i    +1];
       imgOut[i]= (ps+pc+pd)/16;
    }


}
