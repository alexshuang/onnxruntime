all: cuComputeGradInput prof

cuComputeGradInput: cuComputeGradInput.cpp
	hipcc -o $@ $^

prof:
	@./prof.sh cuComputeGradInput
