cmake_minimum_required(VERSION 3.10)

project(my_project)
set(CMAKE_CXX_STANDARD 14)
# 查找 protobuf
find_package(Protobuf REQUIRED)

if (PROTOBUF_FOUND)
    message("protobuf found")
else ()
    message(FATAL_ERROR "Cannot find Protobuf")
endif ()

# 编译 proto 为 .cpp 和 .h
file(GLOB NART_PROTOS *.proto)
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${NART_PROTOS})

message("PROTO_SRCS = ${PROTO_SRCS}")
message("PROTO_HDRS = ${PROTO_HDRS}")

# 将PROTO_SRCS生成静态库nart_proto.a文件
add_library(nart_proto STATIC ${PROTO_SRCS})




target_include_directories(nart_proto
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC ${PROTOBUF_INCLUDE_DIRS}) # 引入头文件

message("PROTOBUF_INCLUDE_DIRS = ${PROTOBUF_INCLUDE_DIRS}")

add_executable(test main.cc ${PROTO_SRCS} ${PROTO_HDRS} )
target_link_libraries(test nart_proto ${PROTOBUF_LIBRARIES}) # 引入库文件