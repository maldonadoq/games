#include <iostream>
#include <vector>
#include <stdlib.h> 
#include <fstream>
#include <math.h>

using namespace std;

class CPoint{
public:
    float x;
    float y;
    CPoint(float _x, float _y){
        this->x = _x;
        this->y = _y;
    }

    CPoint operator-(const CPoint rhs){ 
		return CPoint(x-rhs.x, y-rhs.y);
	}

	CPoint operator+(const CPoint rhs){ 
		return CPoint(x+rhs.x, y+rhs.y);
	}

    CPoint operator/(const float rhs){ 
		return CPoint(x/rhs, y/rhs);
	}

    bool operator==(const CPoint rhs){ 
		return (x == rhs.x) and (y == rhs.y);
	}

    CPoint(){ };
};

float norm(float x, float y){
    return sqrt((x*x)+(y*y));
}

void angle_between(vector<CPoint> &snake_pos, CPoint &apple_pos, float &angle, CPoint &apple_dir, CPoint &snake_dir){
    apple_dir = apple_pos - snake_pos[0];
    snake_dir = snake_pos[0] - snake_pos[1];

    float norm_apple_dir = norm(apple_dir.x, apple_dir.y);
    float norm_snake_dir = norm(snake_dir.x, snake_dir.y);

    if(norm_apple_dir == 0){
        norm_apple_dir = 10;
    }

    if(norm_snake_dir == 0){
        norm_snake_dir = 10;
    }

    apple_dir = apple_dir / norm_apple_dir;
    snake_dir = snake_dir / norm_snake_dir;

    angle = atan2(
        (apple_dir.y * snake_dir.x) - (apple_dir.x * snake_dir.y),
        (apple_dir.y * snake_dir.y) + (apple_dir.x * snake_dir.x)
    ) / M_PI;
}

void dir_vector(vector<CPoint> &snake_pos, float &angle, int tdir, int &direc, int &btn_dir){
    CPoint curr_dir = snake_pos[0] - snake_pos[1];
    CPoint left_dir(curr_dir.y, -curr_dir.x);
    CPoint right_dir(-curr_dir.y, curr_dir.x);

    CPoint new_dir = curr_dir;

    if(tdir == -1){
        new_dir = left_dir;
    }

    if(tdir == 1){
        new_dir = right_dir;
    }

    direc = tdir;

    btn_dir = 0;
    if(new_dir == CPoint(10,0)){
        btn_dir = 1;
    }
    else if(new_dir == CPoint(-10,0)){
        btn_dir = 0;
    }
    else if(new_dir == CPoint(0,10)){
        btn_dir = 2;
    }
    else{
        btn_dir = 3;
    }
}

void gen_rnd_dir(vector<CPoint> &snake_pos, float &angle, int &direc, int &btn_dir){
    int tdirec = 0;
    if(angle > 0){
        tdirec = 1;
    }
    else if(angle < 0){
        tdirec = -1;
    }
    else{
        tdirec = 0;
    }

    dir_vector(snake_pos, angle, tdirec, direc, btn_dir);
}

int dir_block(vector<CPoint> &snake_pos, CPoint &curr_dir){
    CPoint next = snake_pos[0] + curr_dir;

    bool self = false;
	bool wall = (next.x == -1) or (next.x == 500) or (next.y == -1) or (next.y == 500);

	for(auto p: snake_pos){
		if((p.x == next.x) and (p.y == next.y)){
			self = true;
			break;
		}
	}	

	return (wall or self);
}

void block_directions(vector<CPoint> &snake_pos, int &front, int &left, int &right){
    CPoint curr_dir = snake_pos[0] - snake_pos[1];
    CPoint left_dir(curr_dir.y, -curr_dir.x);
    CPoint right_dir(-curr_dir.y, curr_dir.x);

    front = dir_block(snake_pos, curr_dir);
    left = dir_block(snake_pos, left_dir);
    right = dir_block(snake_pos, right_dir);
}

void generate_snake(CPoint &snake_start, vector<CPoint> &snake_pos, CPoint &apple_pos, int &btn_dir, int &score){
    if(btn_dir == 1){
        snake_start.x += 10;
    }
    else if(btn_dir == 0){
        snake_start.x -= 10;
    }
    else if(btn_dir == 2){
        snake_start.y += 10;
    }
    else{
        snake_start.y -= 10;
    }

    if(snake_start == apple_pos){
        apple_pos = CPoint((rand()%50)*10, (rand()%50)*10);
        score += 1;

        snake_pos.insert(snake_pos.begin(), snake_start);
    }
    else{
        snake_pos.insert(snake_pos.begin(), snake_start);
        snake_pos.pop_back();
    }
}

void play(CPoint &snake_start, vector<CPoint> &snake_pos, CPoint &apple_pos, int &btn_dir, int &score){
    generate_snake(snake_start, snake_pos, apple_pos, btn_dir, score);
}

void generate_training_data_y(vector<vector<int>> &training_data_y, vector<CPoint> &snake_pos, float &angle, int &btn_dir, int &direc, int &front, int &left, int &right){
    if(direc == -1){
        if(left == 1){
            if(front == 1 and right == 0){
                dir_vector(snake_pos, angle, 1, direc, btn_dir);
                training_data_y.push_back({0,0,1});
            }
            else if(front == 0 and right == 1){
                dir_vector(snake_pos, angle, 0, direc, btn_dir);
                training_data_y.push_back({0,1,0});
            }
            else if(front == 0 and right == 0){
                dir_vector(snake_pos, angle, 1, direc, btn_dir);
                training_data_y.push_back({0,0,1});
            }
        }
        else{
            training_data_y.push_back({1,0,0});
        }
    }
    else if(direc == 0){
        if(front == 1){
            if(left == 1 and right == 0){
                dir_vector(snake_pos, angle, 1, direc, btn_dir);
                training_data_y.push_back({0,0,1});
            }
            else if(left == 0 and right == 1){
                dir_vector(snake_pos, angle, -1, direc, btn_dir);
                training_data_y.push_back({1,0,0});
            }
            else if(left == 0 and right == 0){
                dir_vector(snake_pos, angle, 1, direc, btn_dir);
                training_data_y.push_back({0,0,1});
            }
        }
        else{
            training_data_y.push_back({0,1,0});
        }
    }
    else{
        if(right == 1){
            if(left == 1 and front == 0){
                dir_vector(snake_pos, angle, 0, direc, btn_dir);
                training_data_y.push_back({0,1,0});
            }
            else if(left == 0 and front == 1){
                dir_vector(snake_pos, angle, -1, direc, btn_dir);
                training_data_y.push_back({1,0,0});
            }
            else if(left == 0 and front == 0){
                dir_vector(snake_pos, angle, -1, direc, btn_dir);
                training_data_y.push_back({1,0,0});
            }
        }
        else{
            training_data_y.push_back({0,0,1});
        }
    }
}

void generate_training_data(vector<vector<float>>& training_data_x, vector<vector<int>>& training_data_y){
    srand (time(NULL));
    int training_games = 250;
    int steps_per_game = 500;

    CPoint snake_start;
    vector<CPoint> snake_position;
    CPoint apple_position;
    int score;

    float angle;
    CPoint apple_dir_norm;
    CPoint snake_dir_norm;

    int direc;
    int btn_dir;

    int front_block;
    int left_block;
    int right_block;

    for(int i = 0; i < training_games; i++){
        snake_start = CPoint(100,100);
        snake_position = {CPoint(100,100),CPoint(90,100),CPoint(80,100)};
        apple_position = CPoint((rand()%50)*10,(rand()%50)*10);
        score = 3;

        for (int j = 0; j < steps_per_game; j++){
            angle_between(snake_position, apple_position, angle, apple_dir_norm, snake_dir_norm);
            gen_rnd_dir(snake_position, angle, direc, btn_dir);
            block_directions(snake_position, front_block, left_block, right_block);

            generate_training_data_y(training_data_y, snake_position, angle, btn_dir, direc, front_block, left_block, right_block);

            if(front_block == 1 and left_block == 1 and right_block == 1){
                break;
            }

            training_data_x.push_back({(float)left_block, (float)front_block, (float)right_block, apple_dir_norm.x, snake_dir_norm.x, apple_dir_norm.y, snake_dir_norm.y});
            play(snake_start, snake_position, apple_position, btn_dir, score);
        }
    }
}

template<class T>
void printMatrix(vector<vector<T>> vec){
    for(unsigned i=0; i<vec.size(); i++){
        for(unsigned j=0; j<vec[0].size(); j++){
            cout << vec[i][j] << " ";
        }
        cout << endl;
    }
}

template<class T>
void saveCsv(string filename, vector<vector<T>> vec){
    ofstream tfile;
    string tmp;
    tfile.open(filename);

    for(unsigned i=0; i<vec.size(); i++){
        tmp = "";
        for(unsigned j=0; j<vec[0].size(); j++){
            if(j == vec[0].size()-1){
                tfile << vec[i][j] << "\n";
            }
            else{
                tfile << vec[i][j] << " ";
            }
        }
    }

    tfile.close();
}

int main(int argc, char const *argv[]){
    vector<vector<float>> training_data_x;
    vector<vector<int>> training_data_y;

    generate_training_data(training_data_x,training_data_y);

    saveCsv("data/trainX.csv", training_data_x);
    saveCsv("data/trainY.csv", training_data_y);

    return 0;
}
