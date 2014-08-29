#include "logistic_regression.h"
#include "mnist_parser.h"

using namespace logistic;
int main(){
	Mnist_Parser m;
	m.load_testing();
	m.load_training();
	vec2d_t x;
	vec_t y;
	vec2d_t test_x;
	vec_t test_y;

	for (size_t i = 0; i < 60000; i++){
		x.push_back(m.train_sample[i]->image);
		y.push_back(m.train_sample[i]->label);
	}

	for (size_t i = 0; i < 10000; i++){
		test_x.push_back(m.test_sample[i]->image);
		test_y.push_back(m.test_sample[i]->label);
	}

	LogisticRegression l(x, y);
	l.train();
	l.test(test_x, test_y);
	getchar();
	return 0;
}