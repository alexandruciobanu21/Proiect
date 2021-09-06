__kernel void Gauss(__global const float *img, __global  float *result, __global const int *width, __global const int *height){
    const int w = *width;
    const int h = *height;
    const int posx = get_global_id(1);
    const int posy = get_global_id(0);
    const int i = w*posy + posx;
    // Gaussian Kernel is:
  	// 1 2 1
	  // 2 4 2
	  // 1 2 1
    if(posx == 0 || posy ==0 || posx==w-1 || posy == h-1){
         result[i] = img[i] ;  
    }else{
        float pixel1, pixel2, pixel3;
        pixel1 =  2*img[i    -1];
        pixel2 =  4*img[i];
        pixel3 =  2*img[i    +1];
       result[i]= (pixel1+pixel2+pixel3)/16;
    }


}
