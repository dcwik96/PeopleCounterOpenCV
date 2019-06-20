#!/bin/bash
cmake -Bbuild -H.
cmake --build build --target all
./build/OpenCVTest