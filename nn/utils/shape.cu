#include "shape.cuh"

Shape::Shape(size_t x, size_t y) : x(x), y(y)
{
}


std::ostream& operator<<(std::ostream& out, const Shape& s){
	out << "[" << s.x << "," << s.y  << "]";
	return out;
}