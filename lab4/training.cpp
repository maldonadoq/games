#include <iostream>
#include <vector>
#include <stdlib.h> 
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
};

float norm(float x, float y){
    return sqrt((x*x)+(y*y));
}

void generate_training_data(vector<vector<float>>& training_data_x, vector<vector<int>>& training_data_y){
    srand (time(NULL));
    int training_games = 10;
    int steps_per_game = 20;

    for(int i = 0; i < training_games; i++){
        CPoint snake_start = CPoint(100,100);
        vector<CPoint> snake_position = {CPoint(100,100),CPoint(90,100),CPoint(80,100)};
        CPoint apple_position = CPoint((rand()%50)*10,(rand()%50)*10);
        int score = 3;

        float prev_apple_distance = norm(apple_position.x-snake_position.x, apple_position.y-snake_position.y);

        for (int j = 0; j < steps_per_game; j++){
            
        }

    }
}




int main(int argc, char const *argv[])
{
    vector<vector<float>> training_data_x;
    vector<vector<int>> training_data_y;

    generate_training_data(training_data_x,training_data_y);

    
    return 0;
}
