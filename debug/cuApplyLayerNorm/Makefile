all: cuApplyLayerNorm prof

cuApplyLayerNorm: cuApplyLayerNorm.cpp
	hipcc -o $@ $^

prof:
	@./prof.sh cuApplyLayerNorm
