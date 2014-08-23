#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>
#include <cstdint>
#include "util.h"
#define MAX_EPOCH 1000

#pragma once
namespace logistic {
	class LogisticRegression
	{
	public:
		LogisticRegression(){}

		LogisticRegression(vec2d_t x, vec_t y) :
			in_size(x[0].size()), in_depth(x.size()), out_size(10), batch_size(10), x_(x), y_(y), alpha(0.13)
		{ 
			W_.resize(in_size * out_size);
			b_.resize(out_size);

			//this->init_weight();
			std::cout << "build logistic regression classifer completely." << std::endl;
		}

		void init_weight(){
			std::cout << "init weight" << std::endl;
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

		inline vec_t getW_vec(int out_index){
			vec_t v;
			for (int i = 0; i < in_size; i++)
				v.push_back(W_[out_index * in_size + i]);
			return v;
		}

		vec_t logistic_single(size_t in_index){
			vec_t out(10, 0);
			for (int i = 0; i < out_size; i++){
				out[i] = dot(x_[in_index], getW_vec(i)) + b_[i];
				//disp_vec_t(getW_vec(i));
			}
			disp_vec_t(out);
			return softmax(out);
		}

		void update_weight(){
			auto g_ = grad();
			std::cout << "grad:" << g_ << std::endl;
			auto tmp = getflatten_X();
			for (size_t out = 0; out < out_size; out++){
				for (size_t in = 0; in < in_size; in++){
					/*fuck*/
					W_[out * in_size + in] += alpha * g_ * tmp[in];
					b_[out] += alpha * g_ * tmp[in];
				}
			}
			disp_vec_t(b_);
		}

		vec_t getflatten_X(){
			vec_t v(in_size, 0);
			for (size_t i = 0; i < batch_size; i++){
				for (size_t j = 0; j < in_size; j++)
					v[j] += x_[batch_y_index[i]][j];
			}
			
			for (auto i : v)
				i /= batch_size;
			
			return v;
		}

		float_t grad(){
			float_t batch_err = 0;
			for (size_t i = 0; i < batch_size; i++){
				float_t sum = 0;
				for (size_t out_index = 0; out_index < out_size; out_index++){
					sum += abs(batch_out[i][out_index] - (y_[batch_y_index[i]] == out_index? 1:0));
				}
				batch_err += (sum / out_size);
			}
			return batch_err / batch_size;
		}

		void train(){
			size_t i = 0;
			while (i < MAX_EPOCH){
				std::cout << "loop:" << i + 1 << std::endl;
				train_once();
				i++;
			}
		}

		void train_once(){
			
			for (auto i : random_choose_samples()){
				std::cout << (int)y_[i] << std::endl;
				auto out = logistic_single(i);
				//disp_vec_t(out);
				batch_out.push_back(out);
			}

			auto cost = negative_log_likelihood(batch_out);
			if (cost > INT_MAX){
				std::cout << "meet stop condition" << std::endl;
				std::cout << cost << std::endl;
				for (auto i : batch_out){
					for (auto j : i)
						std::cout << j << "\t";
					std::cout << std::endl;
				}
				getchar();
			}

			update_weight();
			
		}

		inline std::vector<uint8_t> random_choose_samples(){
			for (size_t i = 0; i < batch_size; i++){
				batch_y_index.push_back(i/*uniform_rand<int>(0, in_depth)*/);
			}
			return batch_y_index;
		}

		float_t negative_log_likelihood(vec2d_t p_y){
			float_t sum = 0.0;
			for (auto i : p_y){
				sum += log(i);
			}
			return -(sum / batch_size);
		}

		inline float_t log(vec_t p){
			float_t sum = 0;
			for (auto i : p){
				sum += std::log(i);
			}
			return sum;
		}

	private:
		size_t in_size;
		size_t in_depth;
		size_t out_size;

		size_t batch_size;
		std::vector<uint8_t> batch_y_index;
		vec2d_t batch_out;
		
		float_t alpha; //learning rate

		vec2d_t x_;
		vec_t y_;
		vec_t W_;
		vec_t b_;
	};
} // namespace logistic