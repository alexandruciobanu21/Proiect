__kernel void Gauss(__global const float *img, __global  float *result, __global const int *width, __global const int *height){
    const int w = *width;
    const int h = *height;
    const int posx = get_global_id(1);
    const int posy = get_global_id(0);
    const int i = w*posy + posx;
    
    if(posx == 0 || posy ==0 || posx==w-1 || posy == h-1){
         result[i] = img[i] ;  
    }else{
        float pixel00, pixel01, pixel02, pixel10, pixel11, pixel12, pixel20, pixel21, pixel22;
        pixel00 =  1*img[i - w-1];
        pixel01 = 2*img[i - w];
        pixel02 =  1*img[i - w+1];
        pixel10 =  4*img[i    -1];
        pixel11 =  16*img[i];
        pixel12 =  4*img[i    +1];
        pixel20 = 1*img[i + w-1];
        pixel21 = 2*img[i + w];
        pixel22 =  1*img[i + w+1];
       // result[i] = pixel00+pixel01+pixel02+pixel10+pixel11+pixel12+pixel20+pixel21+pixel22;
       result[i]= (pixel10+pixel11+pixel12)/16;
    }

// Gaussian Kernel is:
	// 1 2 1
	// 2 4 2
	// 1 2 1


}