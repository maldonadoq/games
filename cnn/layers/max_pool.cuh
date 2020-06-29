#pragma once

#include "../utils/blas.cuh"
#include "layer.cuh"

class MaxPool : public Layer {
 public:
	explicit MaxPool(int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
					int stride_w)
			: kernel_h(kernel_h),
				kernel_w(kernel_w),
				pad_h(pad_h),
				pad_w(pad_w),
				stride_h(stride_h),
				stride_w(stride_w) {}

	void forward();
	void backward();

 private:
	std::unique_ptr<Tensor> mask;
	int kernel_h;
	int kernel_w;
	int pad_h;
	int pad_w;
	int stride_h;
	int stride_w;
};