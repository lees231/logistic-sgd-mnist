#include "logistic_regression.h"
#include "mnist_parser.h"

using namespace logistic;
int main(){
	Mnist_Parser m;
	m.load_testing();
	vec2d_t x;
	vec_t y;
	std::cout << "start training" << std::endl;
	for (size_t i = 0; i < 100; i++){
		x.push_back(m.test_sample[i]->image);
		y.push_back(m.test_sample[i]->label);
	}

	LogisticRegression l(x, y);
	l.train();
	getchar();
	return 0;
}