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

Snake *snake;
Point point;
Point apple;

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
		snake->reset();
		death = false;
	}
	else{
		snake->update();

		glColor3f(0,1,0);
		for(auto point:snake->body){
			if((point.x == apple.x) and (point.y == apple.y)){
				apple.x = rand() % (int) size/unit;
				apple.y = rand() % (int) size/unit;

				snake->grow(apple);
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

	point.x = 0;
	point.y = 0;
	apple.x = rand() % (int) size/unit;
	apple.y = rand() % (int) size/unit;
	snake = new Snake(point, (int) size/unit);
}

void glWindowRedraw(int width, int height){
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, size, 0, size, -1.0f, 1.0f);
}

void glTimer(int t){
    glutPostRedisplay();
    glutTimerFunc(50, glTimer, 0);
}

void glWindowKey(unsigned char key, int x, int y) {
	switch (key) {
		case KEY_ESC:{
			exit(0);
			break;
		}
		case KEY_D:{
			death = snake->setDir(-1);
			break;
		}
		case KEY_A:{
			death = snake->setDir(1);
			break;
		}
		case KEY_W:{
			death = snake->setDir(-2);
			break;
		}
		case KEY_S:{
			death = snake->setDir(2);
			break;
		}
		default:
			break;
	}
}

int main(int argc, char *argv[]){
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(size, size);
	glutInitWindowPosition(50, 50);
	glutCreateWindow("Snake Game");
	glInit();

	glutDisplayFunc(glDraw);

	glutReshapeFunc(&glWindowRedraw);
	glutKeyboardFunc(&glWindowKey);
	glTimer(0);
	glutMainLoop();
	return 0;
}