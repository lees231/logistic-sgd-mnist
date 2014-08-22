#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>
#include "util.h"

namespace logistic {
	class LogisticRegression
	{
		LogisticRegression(vec2d_t x, vec_t y) :
			in_size(x.size()), out_size(10), batch_size(10), x_(x), y_(y)
		{ 
			W_.resize(in_size * out_size);
			b_.resize(out_size);

			this->init_weight();
		}

		void init_weight(){
			uniform_rand(W_.begin(), W_.end(), 0, 0);
			uniform_rand(b_.begin(), b_.end(), 0, 0);
		}

		/*
			http://stackoverflow.com/questions/9906136/implementation-of-a-softmax-activation-function-for-neural-networks
			--------------------------------
			Vector y = mlp(x); // output of the neural network without softmax activation function
			double ymax = maximal component of y
			for(int f = 0; f < y.rows(); f++)
			  y(f) = exp(y(f) - ymax);
			y /= y.sum();
			--------------------------------
		*/

		vec_t softmax(vec_t &in){
			assert(in.size() > 0);
			float_t m = in[0];
			for (size_t i = 1; i < in.size(); i++)
				m = std::max(m, in[i]);

			for (auto &i : in)
				i = exp(i - m);
			
			float_t sum = std::accumulate(in.begin(), in.end(), 0);

			for (auto &i : in)
				i /= sum;
			
			return in;
		}

		float_t dot(vec_t x, vec_t w){
			assert(x.size() == w.size());
			float_t sum = 0;
			for (size_t i = 0; i < x.size(); i++){
				sum += x[i] * w[i];
			}
			return sum;
		}

	private:
		size_t in_size;
		size_t out_size;
		size_t batch_size;
		vec2d_t x_;
		vec_t y_;
		vec_t W_;
		vec_t b_;
	};


} // namespace logistic