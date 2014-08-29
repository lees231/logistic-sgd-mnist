#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>
#include <cstdint>
#include "util.h"
#define MAX_TRAIN_EPOCH 60000
#define MAX_TEST_EPOCH 10000

#pragma once
namespace logistic {
	class LogisticRegression
	{
	public:
		LogisticRegression(vec2d_t x, vec_t y) :
			in_size(x[0].size()), in_depth(x.size()), out_size(10), x_(x), y_(y), alpha(0.003), lambda(0.1)
		{ 
			W_.resize(in_size * out_size);
			b_.resize(out_size);
			exp_y.assign({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

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

		/*
		stochastic gradient descent with weight decay.
		
		*/
		void update_weight(){

			vec_t g;
			for (size_t i = 0; i < out_size; i++){
				float_t _ = 0 - train_y[i];
				if (abs(exp_y[i] - 1.0) < 1e-7)
					_ = 1.0 - train_y[i];
				g.push_back(_);
			}
			//disp_vec_t(g);
			for (size_t out = 0; out < out_size; out++){
				for (size_t in = 0; in < in_size; in++){
					/*fuck*/
					W_[out * in_size + in] += alpha * (g[out] * x_[train_x_index][in] 
						+ /*weight decay*/lambda * W_[out * in_size + in]);
				}
			}
		}
		
		void train(){
			size_t i = 0;
			while (i <  MAX_TRAIN_EPOCH){
				std::cout << "loop:" << i + 1 << std::endl;
				train_once();
				i++;
			}
		}

		size_t max_iter(vec_t v){
			size_t i = 0;
			float_t max = v[0];
			for (size_t j = 1; j < v.size(); j++){
				if (v[j] > max){
					max = v[j];
					i = j;
				}
			}
			return i;
		}

		void test(vec2d_t test_x, vec_t test_y){
			size_t i = 0;
			size_t err = 0;
			while (i < MAX_TEST_EPOCH){
				std::cout << "loop:" << i + 1 << std::endl;
				size_t test_x_index = uniform_rand(0, MAX_TEST_EPOCH - 1);
				//std::cout << "train_x_index:" << train_x_index << std::endl;
				//disp_vec_t(exp_y);
				//std::cout << "test:" << (int)y_[test_x_index] << std::endl;
				//disp_vec_t(exp_y);
				vec_t ptest_y;
				for (int out = 0; out < out_size; out++){
					ptest_y.push_back(dot(test_x[test_x_index], get_W(out)) + b_[out]);
				}
				//disp_vec_t(train_y);
				softmax(ptest_y);
				
				int y_index = max_iter(ptest_y);
				//std::cout << "predict:" << y_index << std::endl;
				if (y_index != (int)test_y[test_x_index]){
					err++;
				}
				i++;
			}
			std::cout << "err:" << (float)err / 100 << "%" << std::endl;
		}

		vec_t get_W(size_t index){
			vec_t v;
			for (int i = 0; i < in_size; i++){
				v.push_back(W_[index * in_size + i]);
			}
			return v;
		}

		void train_once(){
			train_x_index = uniform_rand(0, in_depth - 1);
			//std::cout << "train_x_index:" << train_x_index << std::endl;
			//disp_vec_t(exp_y);
			//std::cout << (int)y_[train_x_index] << std::endl;
			exp_y[(int)y_[train_x_index]] = 1.0;
			//disp_vec_t(exp_y);
			for (int out = 0; out < out_size; out++){
				train_y.push_back(dot(x_[train_x_index], get_W(out)) + b_[out]);
			}
			//disp_vec_t(train_y);
			softmax(train_y);
			//disp_vec_t(train_y);
			auto cost = negative_log_likelihood();
		
			update_weight();
			exp_y[(int)y_[train_x_index]] = 0.0;
			train_y.clear();
		}

		float_t decay(){
			float_t sum = 0;
			for (int out = 0; out < out_size; out++){
				for (int in = 0; in < in_size; in++){
					sum += W_[out * in_size + in];
				}
			}
			return lambda * 0.5 * sum;
		}

		float_t negative_log_likelihood(){
			return -std::log(train_y[(int)y_[train_x_index]]) + decay();
		}

	private:

		float_t dot(vec_t x, vec_t w){
			assert(x.size() == w.size());
			float_t sum = 0;
			for (size_t i = 0; i < x.size(); i++){
				sum += x[i] * w[i];
			}
			return sum;
		}
		
		size_t in_size;
		size_t in_depth;
		size_t out_size;
		
		float_t alpha; // learning rate
		float_t lambda; // weight decay
		vec2d_t x_;
		vec_t y_;
		vec_t W_;
		vec_t b_;
		/*
		vec2d_t test_x_;
		vec_t test_y_;
		*/
		size_t train_x_index;
		vec_t train_y;
		vec_t exp_y;
	};
} // namespace logistic