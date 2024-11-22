cmake_minimum_required(VERSION 3.31.0)
set(CMAKE_CXX_STANDARD 17)
project( rl_test )

set(CMAKE_VERBOSE_MAKEFILE ON)
set(Protobuf_PROTOC_EXECUTABLE "/opt/homebrew/bin/protoc")
find_package(Protobuf REQUIRED)

# 设置 proto 文件
set(PROTO_FILES message.proto)

# 生成 C++ 代码
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# 添加可执行文件
add_executable(my_executable main.cc ${PROTO_SRCS} ${PROTO_HDRS})

# 链接 protobuf 库
target_link_libraries(my_executable ${Protobuf_LIBRARIES})