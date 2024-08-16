train:
	python train.py

build:
	cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
	cmake --build build

clean:
	rm -rf build

profile:
	nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop -o profile -f true python train.py

works:
	./build/inference ${PWD}/export/scatter_works/model.so

error:
	./build/inference ${PWD}/export/scatter_linear_error/model.so


.PHONY: run works error build clean

