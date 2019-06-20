cmake_minimum_required(VERSION 3.13)
project(PeopleCounterOpenCV)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(PeopleCounterOpenCV
        "videos/[10 sec fragment, HDconvert.com] TownCentreXVID.avi"
        videos/Video_003.avi
        buildAndRun.sh
        CMakeLists.txt
        cvui.h
        main.cpp)
