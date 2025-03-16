CC=nvcc
CXXFLAGS += -arch=sm_86
INCLUDES  += -I./include
LDFLAGS =

#获取当前目录下的cu文件集，放在变量CUR_SOURCE中
CUR_SOURCE=${wildcard ./src/*.cu}

#将对应的c文件名转为o文件后放在下面的CUR_OBJS变量中
CUR_OBJS=${patsubst %.cu, %.o, $(CUR_SOURCE)}

EXECUTABLE=conv2dfp16


all:$(EXECUTABLE)

$(EXECUTABLE): $(CUR_OBJS)
	$(CC) $(CUR_OBJS) $(CXXFLAGS) $(LDFLAGS) -o $(EXECUTABLE)
      
	
%.o:%.cu
	$(CC) -c -w $< $(CXXFLAGS) $(INCLUDES) -o $@
	
	
clean:
	rm -f $(EXECUTABLE)
	rm -f ./src/*.o
