#include <iostream>
#include <string>

/* Cuda Neural Network */
#include "../nn/neurals/neural_network.cuh"
#include "../nn/layers/linear_layer.cuh"
#include "../nn/layers/linear_relu.cuh"
#include "../nn/layers/linear_softmax.cuh"
#include "../nn/layers/relu_activation.cuh"
#include "../nn/utils/cce_loss.cuh"
#include "../nn/layers/softmax_activation.cuh"
#include "../nn/neurals/snake_dataset.cuh"
#include "src/result.cuh"

#define num_batches_train 175
#define batch_size 256

#define input_size 7
#define output_size 3

NeuralNetwork *nn;
Matrix *Y;

/* Snake in OpenGL */
#include <iostream>
#include <GL/glut.h>
#include "src/snake.h"

#define KEY_ESC 27
#define KEY_D 'd'	// Right
#define KEY_A 'a'	// Left
#define KEY_W 'w'	// Up
#define KEY_S 's'	// Down

float size = 500;
float unit = 10;
bool death = false;

Snake *snake_game;
Point apple;
vector<float> input;
vector<int> output;

void glDrawQuad(Point p){
	glBegin(GL_QUADS);
        glVertex2f((p.x+1)*unit, (p.y+1)*unit);
        glVertex2f(p.x*unit, (p.y+1)*unit);
		glVertex2f(p.x*unit, p.y*unit);
        glVertex2f((p.x+1)*unit, p.y*unit);
    glEnd();
}

void glDraw(){
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();
	glOrtho(0, size, 0, size, -1.0f, 1.0f);
	
	if(death){
		snake_game->reset();
		death = false;
	}
	else{
		input = snake_game->getData(apple);

		*Y = nn->forward(Matrix(Shape(1, input_size), input));
		Y->copyDeviceToHost();
		output = firstResultInt(*Y, output_size);
		//printVector(input);
		printVector(output);

		//snake_game->move(0);
		glColor3f(0,1,0);

		for(auto point:snake_game->body){
			if((point.x == apple.x) and (point.y == apple.y)){
				snake_game->grow(apple);

				apple.x = rand() % (int) size/unit;
				apple.y = rand() % (int) size/unit;				
			}
			glDrawQuad(point);
		}

		glColor3f(0,0,1);
		glDrawQuad(apple);
	}

	glutSwapBuffers();
}

void glInit(void) {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	apple.x = rand() % (int) size/unit;
	apple.y = rand() % (int) size/unit;
	snake_game = new Snake((int) size/unit);
}

void glWindowRedraw(int w, int h){
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, size, 0, size, -1.0f, 1.0f);
}

void glIdle(){
	glutPostRedisplay();
}

void glWindowKey(unsigned char key, int x, int y) {
	switch (key) {
		case KEY_ESC:{
			exit(0);
			break;
		}
		case KEY_D:{
			snake_game->move(0);
			break;
		}
		case KEY_A:{
			snake_game->move(1);
			break;
		}
		case KEY_W:{
			snake_game->move(2);
			break;
		}
		case KEY_S:{
			snake_game->move(3);
			break;
		}
		default:
			break;
	}
}

int main(int argc, char *argv[]){
	int epochs = 10;
	int epoch, batch;
	float loss;

	if(argc == 2){
		epochs = std::stoi(argv[1]);
	}

	CCELoss cce_cost;
	nn = new NeuralNetwork();
	Y = new Matrix();

	nn->addLayer(new LinearLayer("linear_1", Shape(input_size, 256)));
	nn->addLayer(new ReLUActivation("relu_1"));
	nn->addLayer(new LinearLayer("linear_2", Shape(256, output_size)));
	nn->addLayer(new softmaxActivation("softmax_output"));

	
	SnakeDataset snake_train(num_batches_train, batch_size, "data/trainX.csv", "data/trainY.csv");
	
	for (epoch = 0; epoch < epochs; epoch++){
		loss = 0.0;
		for (batch = 0; batch < num_batches_train; batch++){
			*Y = nn->forward(snake_train.getBatches().at(batch));
			nn->backprop(*Y, snake_train.getTargets().at(batch));
			loss += cce_cost.cost(*Y, snake_train.getTargets().at(batch));
		}

		if(epoch % 10 == 0){
			std::cout << "Epoch: " << epoch << ", Loss: " << loss / snake_train.getNumOfBatches() << std::endl;
		}		
	}
	std::cout << std::endl;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(size, size);
	glutInitWindowPosition(50, 50);
	glutCreateWindow("Snake Game");
	glInit();

	glutDisplayFunc(glDraw);

	glutReshapeFunc(&glWindowRedraw);
	glutKeyboardFunc(&glWindowKey);
	glutIdleFunc(&glIdle);
	glutMainLoop();
	
	delete snake_game;
	delete nn;
	delete Y;

	return 0;
}