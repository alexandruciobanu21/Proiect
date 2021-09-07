__kernel void Gauss(__global const float *imgIn, __global  float *imgOut, __global const int *width, __global const int *height){
    const int w = *width;
    const int h = *height;
    float p1, p2, p3, p4, p5, p6, p7, p8, p9;
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    const int i = w*y + x;
    // Gaussian matrix conv  Kernel  is:
  	// 1 2 1     p1 p2 p3
	  // 2 4 2  *  p4 p5 p6   * 1/16 
	  // 1 2 1     p7 p8 p9
    if(x == 0 || y ==0 || x==w-1 || y == h-1){
         imgOut[i] = imgIn[i] ;  
    }else{
        p1 = 1*imgIn[i - w-1];
        p2 = 2*imgIn[i - w];
        p3 = 1*imgIn[i - w+1];
        p4 = 2*imgIn[i    -1];
        p5 = 4*imgIn[i];
        p6 = 2*imgIn[i    +1];
        p7 = 1*imgIn[i + w-1];
        p8 = 2*imgIn[i + w];
        p9 = 1*imgIn[i + w+1];

       imgOut[i]= (p1+p2+p3+p4+p5+p6+p7+p8+p9)/16;
    }


}
