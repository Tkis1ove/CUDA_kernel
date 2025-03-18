# 基础配置
CUDA_PATH    ?= /usr/local/cuda-12.8
NVCC         := $(CUDA_PATH)/bin/nvcc
TARGET       := sum
BUILD_DIR    := build
SRC_DIR      := src

# 架构配置（根据实际GPU调整）
ARCH_FLAGS   := -gencode arch=compute_86,code=sm_86

# 文件自动查找
CU_SRCS      := $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS     := $(wildcard $(SRC_DIR)/*.cpp)
OBJS         := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.cu.o, $(CU_SRCS)) \
                $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.cpp.o, $(CPP_SRCS))

# 编译选项
CFLAGS       := -O2
INCLUDES     := -I$(CUDA_PATH)/include -I./include
LIBS         := -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcurand

# 调试模式配置
DEBUG        ?= 0
ifeq ($(DEBUG),1)
    CFLAGS   += -g -G
endif

# 构建规则
$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(ARCH_FLAGS) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(ARCH_FLAGS) $^ -o $@ $(LIBS)

.PHONY: clean run
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: $(TARGET)
	./$(TARGET)