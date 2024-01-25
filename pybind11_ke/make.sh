mkdir -p release
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) ./base/Base.cpp -o ./release/base$(python3-config --extension-suffix)